// SHRIMP: CPU constrained RRT-Connect, scalar, one problem per call (single-threaded; callers parallelise over problems).
// Needs, concatenated before it (src/rrtc.py: build_cpu_crrtc_source): CRAB's CPU robot header (N, LO, HI, World, fkcc),
// CRAB's TSR header (Tsr, ProjParams, tsr_*; crab_codegen/constraint_header.py), kdtree.hpp and rrtc.hpp (Params, Tree,
// nearest, halton, Result, RES, MAXP).
//
// Plans between two configurations ON the task-space region. Nothing here is a tolerance or a bound: the region
// (Tsr: bounds + per-row tolerance) and the settings (CParams) are inputs.
//
// Local planning follows McVAMP / CBiRRT2 (edge_mode 0), scalar:
//   * steer: the step toward the sample is projected onto the region from `lanes` candidates (step length x
//     (1 + perturb * {0, -.25, +.25, -.5, ...}); the first candidate that converges wins, lane 0 is the exact step), the
//     result must stay within 2 step lengths of where it started, then the edge is traced:
//   * trace: `lanes` waypoints along the chord, each projected; adjacent waypoints must stay within 2x the chord
//     spacing; all collision-free; then the gaps are swept at RES, projecting every substep (it must stay near its
//     interpolant and collision-free).
//   * connect: hops of <= range toward the other tree's nearest node, at most ceil(slack * distance / range).
// Ablations derived from cpRRTC (Hu et al.), all off by default:
//   * edge_mode 1: the edge is a dense chain (waypoints ~`spacing` apart) projected jointly, no gap sweep;
//     smooth: a term pulling adjacent waypoints back to the chord spacing (so no gap grows past 2x);
//     freeze:  waypoints of the satisfied prefix of the chain stop being updated;
//   * closer: a connect hop must get closer to its target by more than closer2 (squared), else the connect stops.
// Every waypoint of a successful chain is added to the tree when emit_all (else only the endpoint), so paths stay on the
// region between nodes.
#include <cfloat>

constexpr int CMAXL = 32;     // most lanes / emitted waypoints per extension
constexpr int CMAXW = 128;    // most waypoints of a dense chain (edge_mode 1)

struct CParams {
    ProjParams proj;          // projection settings (tsr header)
    int lanes;                // candidates per steer, chain waypoints per trace (McVAMP's rake), <= CMAXL
    float perturb;            // scale of the per-candidate step-length perturbation
    float slack;              // a connect may take ceil(slack * distance / range) hops
    float reached2;           // a steer target counts as reached within this squared radius
    int emit_all;             // add every chain waypoint to the tree (else only the endpoint)
    int edge_mode;            // 0 McVAMP lane chain + gap sweep, 1 cpRRTC dense chain
    int smooth, freeze;       // edge_mode 1 ablations
    int closer;               // connect ablation
    float closer2;            //   ... squared progress a hop must make
    float smooth_rate;        // step scale of the smoothing term
    float spacing;            // edge_mode 1: waypoint spacing (rad)
};

static thread_local long g_pe = 0;     // full task-space evaluations (error + Jacobian) made by projections (instrumentation)
static thread_local long g_pc = 0;     // error-only evaluations (tsr_err)

#ifndef TSC
#define TSC() 0ull
#endif
// -DSHRIMP_PROF: cycles of the pieces of a projection and of the collision check (per call, summed per thread)
static thread_local unsigned long long g_pcy[4] = {0, 0, 0, 0};     // tsr_err, tsr_eval, tsr_step, fkcc
static inline void p_err(const float* q, const Tsr& t, float* e)
{
    const unsigned long long a = TSC();
    tsr_err(q, t, e);
    g_pcy[0] += TSC() - a;
    ++g_pc;
}
static inline void p_eval(const float* q, const Tsr& t, bool hold, float* e, float* J)
{
    const unsigned long long a = TSC();
    tsr_eval(q, t, hold, e, J);
    g_pcy[1] += TSC() - a;
    ++g_pe;
}
static inline float p_step(float* q, const float* e, const float* J, const ProjParams& p)
{
    const unsigned long long a = TSC();
    const float s2 = tsr_step(q, e, J, p);
    g_pcy[2] += TSC() - a;
    return s2;
}

static inline float sqd(const float* a, const float* b)
{
    float s = 0.f;
    for (int k = 0; k < N; ++k) s += (a[k] - b[k]) * (a[k] - b[k]);
    return s;
}

// tsr_project (tsr header) with the evaluation counter. q is clamped to the joint limits first: a candidate that is
// already on the region would otherwise return unprojected, and only steps clamp (a perturbed step can leave the limits).
static bool cproject(float* q, const Tsr& t, const ProjParams& p)
{
    float q0[N], e[NR], J[NR * N];
    for (int k = 0; k < N; ++k) q0[k] = q[k] = fminf(fmaxf(q[k], LO[k]), HI[k]);
    for (int it = 0;; ++it) {
        p_err(q, t, e);                                // cheap: the Jacobian only when a step is needed
        if (tsr_satisfied(e, t)) return true;
        if (it >= p.max_iters) return false;
        p_eval(q, t, p.hold != 0, e, J);
        if (p_step(q, e, J, p) <= p.stall2) return false;
        if (p.drift2 > 0.f && sqd(q, q0) >= p.drift2) return false;
    }
}

struct Chain { float q[CMAXL][N]; int n; };     // waypoints to add to a tree, the last is the extension's endpoint

static inline bool cfree(const float* q, const World& w)
{
    ++g_fk;
    const unsigned long long a = TSC();
    const bool free = fkcc(q, w);
    g_pcy[3] += TSC() - a;
    return free;
}

// Multiplier of the lane's step perturbation: 0, -0.25, +0.25, -0.5, +0.5, ...
static inline float lane_mult(int i)
{
    if (i == 0) return 0.f;
    const float m = (float)((i + 1) / 2) * 0.25f;
    return (i % 2 == 1) ? -m : m;
}

// edge_mode 0: McVAMP's trace of the constrained local path a -> b (see the header comment).
static bool ctrace_lanes(const float* a, const float* b, const World& w, const Tsr& t, const CParams& cp, Chain& ch)
{
    const int L = std::min(std::max(cp.lanes, 1), CMAXL);
    float vec[N];
    const float dist2 = sqd(b, a);
    for (int k = 0; k < N; ++k) vec[k] = b[k] - a[k];
    ProjParams pp = cp.proj;
    pp.drift2 = 4.f * dist2;
    for (int l = 0; l < L; ++l) {
        const float pct = (float)(l + 1) / (float)L;
        for (int k = 0; k < N; ++k) ch.q[l][k] = a[k] + vec[k] * pct;
        if (!cproject(ch.q[l], t, pp)) return false;
    }
    const float spacing = std::sqrt(dist2) / (float)L;
    const float max_gap2_allowed = 4.f * spacing * spacing;
    float max_gap2 = 0.f;
    for (int l = 0; l < L; ++l) {
        const float gap2 = sqd(ch.q[l], l == 0 ? a : ch.q[l - 1]);
        if (gap2 > max_gap2_allowed) return false;
        max_gap2 = std::max(max_gap2, gap2);
    }
    for (int l = L - 1; l >= 0; --l)
        if (!cfree(ch.q[l], w)) return false;
    const float max_gap = std::sqrt(max_gap2);
    const int n_steps = std::max((int)std::ceil(max_gap / RES), 1);
    if (n_steps > 1) {
        const float substep = max_gap / (float)n_steps;
        const float max_drift2 = 4.f * substep * substep;
        ProjParams ps = cp.proj;
        ps.drift2 = 4.f * max_gap * max_gap * (float)(L * L);
        for (int i = 1; i < n_steps; ++i) {
            const float f = (float)i / (float)n_steps;
            for (int l = L - 1; l >= 0; --l) {
                const float* prev = l == 0 ? a : ch.q[l - 1];
                float lin[N], pr[N];
                for (int k = 0; k < N; ++k) pr[k] = lin[k] = ch.q[l][k] - f * (ch.q[l][k] - prev[k]);
                if (!cproject(pr, t, ps) || sqd(pr, lin) >= max_drift2 || !cfree(pr, w)) return false;
            }
        }
    }
    ch.n = L;
    return true;
}

// edge_mode 1: cpRRTC-style trace: a dense chain of waypoints on the chord, projected jointly (see the header comment).
static bool ctrace_dense(const float* a, const float* b, const World& w, const Tsr& t, const CParams& cp, Chain& ch)
{
    float vec[N];
    const float dist = std::sqrt(sqd(b, a));
    for (int k = 0; k < N; ++k) vec[k] = b[k] - a[k];
    const int n = std::min(std::max((int)std::ceil(dist / cp.spacing), 1), CMAXW);
    const float s = dist / (float)n;                    // chord spacing: the smoothing target
    float wp[CMAXW][N], nw[CMAXW][N];
    bool ok[CMAXW];
    for (int i = 0; i < n; ++i)
        for (int k = 0; k < N; ++k) wp[i][k] = a[k] + vec[k] * ((float)(i + 1) / (float)n);
    int frozen = -1;                                    // waypoints 0..frozen are satisfied and no longer updated
    for (int it = 0;; ++it) {
        bool all = true;
        for (int i = frozen + 1; i < n; ++i) {
            float e[NR], J[NR * N];
            p_err(wp[i], t, e);                         // cheap first; the Jacobian only for a waypoint that must move
            const float* prev = i == 0 ? a : wp[i - 1];
            const float d = std::max(std::sqrt(sqd(wp[i], prev)), 1e-9f);
            const float serr = d - s;                   // spacing error; a gap above 2 s counts as unsatisfied
            const bool sat = tsr_satisfied(e, t);
            ok[i] = sat && !(cp.smooth && serr > s);
            all = all && ok[i];
            for (int k = 0; k < N; ++k) nw[i][k] = wp[i][k];
            if (!sat) {
                p_eval(wp[i], t, cp.proj.hold != 0, e, J);
                p_step(nw[i], e, J, cp.proj);
            }
            if (cp.smooth)
                for (int k = 0; k < N; ++k)
                    nw[i][k] = fminf(fmaxf(nw[i][k] - cp.smooth_rate * (wp[i][k] - prev[k]) / d * serr, LO[k]), HI[k]);
        }
        if (all) break;
        if (it >= cp.proj.max_iters) return false;
        if (cp.freeze)
            while (frozen + 1 < n && ok[frozen + 1]) ++frozen;
        for (int i = frozen + 1; i < n; ++i)
            for (int k = 0; k < N; ++k) wp[i][k] = nw[i][k];
    }
    for (int i = 0; i < n; ++i)                         // without smoothing nothing bounded the gaps: check them
        if (sqd(wp[i], i == 0 ? a : wp[i - 1]) > 4.f * s * s * 1.0001f) return false;
    for (int i = n - 1; i >= 0; --i)
        if (!cfree(wp[i], w)) return false;
    const int L = std::min(std::max(cp.lanes, 1), CMAXL);
    const int stride = (n + L - 1) / L;                 // emit about `lanes` waypoints, always the last
    ch.n = 0;
    for (int i = 0; i < n; ++i)
        if ((i + 1) % stride == 0 || i == n - 1) {
            for (int k = 0; k < N; ++k) ch.q[ch.n][k] = wp[i][k];
            ++ch.n;
        }
    return true;
}

enum { TRAPPED = 0, ADVANCED = 1, REACHED = 2 };

// Extend from `from` toward `target` (at `dist`) by at most `range`, staying on the region. Fills ch.
static int csteer(const float* from, const float* target, float dist, float range, const World& w, const Tsr& t,
                  const CParams& cp, Chain& ch)
{
    const bool reach = dist < range;
    const float step = reach ? dist : range;
    const float sc = reach ? 1.f : range / dist;
    float vec[N], next[N];
    for (int k = 0; k < N; ++k) vec[k] = (target[k] - from[k]) * sc;
    ProjParams pp = cp.proj;
    pp.drift2 = 4.f * step * step + 1e-6f;
    bool found = false;
    for (int l = 0; l < cp.lanes && !found; ++l) {
        const float m = 1.f + cp.perturb * lane_mult(l);
        for (int k = 0; k < N; ++k) next[k] = from[k] + vec[k] * m;
        found = cproject(next, t, pp);
    }
    if (!found || sqd(next, from) > 4.f * step * step) return TRAPPED;
    if (!cfree(next, w)) return TRAPPED;                // the endpoint first: cheap, fails most often
    const bool traced = cp.edge_mode == 1 ? ctrace_dense(from, next, w, t, cp, ch) : ctrace_lanes(from, next, w, t, cp, ch);
    if (!traced) return TRAPPED;
    if (!cp.emit_all) {
        for (int k = 0; k < N; ++k) ch.q[0][k] = ch.q[ch.n - 1][k];
        ch.n = 1;
    }
    return reach && sqd(ch.q[ch.n - 1], target) < cp.reached2 ? REACHED : ADVANCED;
}

// Add ch's waypoints to tr as a parent chain from `parent`. Returns the last node's index, or -1 when the tree is full.
static int insert_chain(Tree& tr, int parent, const Chain& ch, int max_samples)
{
    if (tr.size() + ch.n > max_samples) return -1;
    for (int i = 0; i < ch.n; ++i) {
        tr.add(ch.q[i], parent);
        parent = tr.size() - 1;
    }
    return parent;
}

// Hops of <= range from cfg toward target, adding each to tr (parent chain from `cur`). 1 reached, 0 blocked, -1 tree
// full. cfg and cur are advanced.
static int cconnect(Tree& tr, int& cur, float* cfg, const float* target, const World& w, const Tsr& t, const CParams& cp,
                    const Params& prm)
{
    const int hops = std::max(1, (int)std::ceil(cp.slack * std::sqrt(sqd(target, cfg)) / prm.range));
    for (int i = 0; i < hops; ++i) {
        const float rem = std::sqrt(sqd(target, cfg));
        const bool last = i == hops - 1 || rem <= prm.range;
        Chain ch;
        const int st = csteer(cfg, target, rem, last ? FLT_MAX : prm.range, w, t, cp, ch);
        if (st == TRAPPED) return 0;
        const float* end = ch.q[ch.n - 1];
        if (cp.closer && sqd(end, target) >= rem * rem - cp.closer2) return 0;
        const int c = insert_chain(tr, cur, ch, prm.max_samples);
        if (c < 0) return -1;
        cur = c;
        for (int k = 0; k < N; ++k) cfg[k] = end[k];
        if (st == REACHED) return 1;
    }
    return 0;
}

// Plan start -> goal on the region. start and goal are first projected onto it (solved = -2 when that fails). On
// success writes the path (start first) to path[MAXP][N] and its length to Result.len (-1 when longer than MAXP).
// Result.solved: 1 yes, 0 out of iterations, -1 a tree filled up, -2 start or goal not projectable.
static Result plan_c(const float* start, const float* goal, const World& w, const Tsr& t, const Params& prm,
                     const CParams& cp, float* path)
{
    Result r{0, 0, 0};
    float s0[N], g0[N];
    for (int k = 0; k < N; ++k) { s0[k] = start[k]; g0[k] = goal[k]; }
    if (!cproject(s0, t, cp.proj) || !cproject(g0, t, cp.proj)) { r.solved = -2; return r; }
    Tree T[2];
    for (auto& tr : T) tr.init(prm);
    T[0].add(s0, 0);
    T[1].add(g0, 0);
    const unsigned skip = prm.seed * 9973u;
    int ta = 1;
    for (int it = 1; it <= prm.max_iters; ++it) {
        r.iters = it;
        float cfg[N];
        int tt, o, cur = 0, on;
        if (it == 1) {                                       // direct attempt: start root -> goal root
            tt = 0; o = 1; on = 0;
            for (int k = 0; k < N; ++k) cfg[k] = s0[k];
        } else {
            const float sa = (float)T[ta].size(), sb = (float)T[1 - ta].size();
            if (std::fabs(sa - sb) / sa < 1.f) ta = 1 - ta;  // VAMP's balance rule, as rrtc.hpp
            tt = ta;
            o = 1 - tt;
            for (int k = 0; k < N; ++k) cfg[k] = LO[k] + halton(skip + it, PRIMES[k]) * (HI[k] - LO[k]);
            const int nb = nearest(T[tt], cfg);
            const float* near = T[tt].at(nb);
            const float dist = std::sqrt(sqd(cfg, near));
            if (dist < 1e-6f) continue;
            Chain ch;
            if (csteer(near, cfg, dist, prm.range, w, t, cp, ch) == TRAPPED) continue;
            cur = insert_chain(T[tt], nb, ch, prm.max_samples);
            if (cur < 0) { r.solved = -1; return r; }
            for (int k = 0; k < N; ++k) cfg[k] = ch.q[ch.n - 1][k];
            on = nearest(T[o], cfg);
        }
        const int res = cconnect(T[tt], cur, cfg, T[o].at(on), w, t, cp, prm);
        if (res < 0) { r.solved = -1; return r; }
        if (res == 0) continue;
        const int ia = tt == 0 ? cur : on, ib = tt == 0 ? on : cur;
        int la = 0, lb = 0;
        for (int i = ia;; i = T[0].par[i]) { ++la; if (i == 0) break; }
        for (int i = ib;; i = T[1].par[i]) { ++lb; if (i == 0) break; }
        r.solved = 1;
        if (la + lb > MAXP) { r.len = -1; return r; }
        int k = la;
        for (int i = ia;; i = T[0].par[i]) {
            --k;
            std::copy(T[0].at(i), T[0].at(i) + N, path + (size_t)k * N);
            if (i == 0) break;
        }
        k = la;
        for (int i = ib;; i = T[1].par[i]) {
            std::copy(T[1].at(i), T[1].at(i) + N, path + (size_t)k * N);
            ++k;
            if (i == 0) break;
        }
        r.len = la + lb;
        return r;
    }
    return r;
}
