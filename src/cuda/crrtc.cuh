// SHRIMP: GPU constrained RRT-Connect in one CUDA kernel: cuda/rrtc.cuh's structure (B warps per problem, shared trees in
// global memory, many problems per launch) with every extension kept on a task-space region.
// Needs, concatenated before it (src/rrtc.py: build_crrtc_source): CRAB's GPU robot header (N, LO, HI, World, fkcc), CRAB's
// TSR header (Tsr, ProjParams, tsr_*; crab_codegen/constraint_header.py) and rrtc.cuh (Params, PState, NT, MAXP, halton,
// PRIMES, TREE_RATIO, urand, nearest, block_argmin).
//
// The warp's 32 lanes are McVAMP's SIMD lanes (cpu/crrtc.hpp is the scalar reference; same names for the same things):
//   * steer: lane l (< lanes) projects its own candidate, step x (1 + perturb * {0, -.25, +.25, ...}); the lowest lane
//     that converges (warp ballot) wins and must stay within 2 step lengths of where it started;
//   * edge: a dense chain of n = ceil(length / spacing) <= 32 waypoints on the chord, lane i holding waypoint i. Each lane
//     projects its own (edge_mode 0), the gaps must stay <= gap_slack x the chord spacing, and all waypoints are
//     collision-checked together by the lane-mask fkcc. With one thread per waypoint there is no gap sweep.
//   * connect: hops of <= range toward the other tree's nearest node, at most ceil(slack * distance / range).
// cpRRTC-derived ablations (all off by default): edge_mode 1 projects the chain jointly (Jacobi: every waypoint steps
// from the previous iterate), smooth pulls adjacent waypoints back to the chord spacing, freeze stops updating the
// satisfied prefix (a warp ballot), closer makes a connect hop get closer to its target.
// Waypoints of a chain are added to the tree in one claim (emit_all: about `lanes` of them, else the endpoint).
// Nothing here is a tolerance or a bound: the region (Tsr) and CParams are inputs.

struct CParams {
    ProjParams proj;          // projection settings (tsr header)
    int lanes;                // steer candidates; emitted chain waypoints per extension (<= NT)
    float perturb;            // scale of the per-candidate step-length perturbation
    float slack;              // a connect may take ceil(slack * distance / range) hops
    float reached2;           // a steer target counts as reached within this squared radius
    int emit_all;             // add about `lanes` chain waypoints to the tree (else only the endpoint)
    int edge_mode;            // 0 independent per-lane projection, 1 joint chain projection (cpRRTC)
    int smooth, freeze;       // edge_mode 1 ablations
    int closer;               // connect ablation
    float closer2;            //   ... squared progress a hop must make
    float smooth_rate;        // step scale of the smoothing term
    float spacing;            // waypoint spacing (rad); a chain has at most NT waypoints
    float gap_slack;          // adjacent chain waypoints must stay within gap_slack x the chord spacing
};

enum { TRAPPED = 0, ADVANCED = 1, REACHED = 2 };

struct CShared {              // per-block scratch (shared memory)
    float wp[NT][N];          // chain waypoints: lane i holds waypoint i
    float nw[NT][N];          // edge_mode 1: the next iterate
    float from[N], target[N], cfg[N];
    int emit[NT];             // chain waypoints to add to the tree (indices into wp), emit[m - 1] the endpoint
    int m, status;
};

__device__ __forceinline__ float dsqd(const float* a, const float* b)
{
    float s = 0.f;
    for (int k = 0; k < N; ++k) s += (a[k] - b[k]) * (a[k] - b[k]);
    return s;
}

// Multiplier of the lane's step perturbation: 0, -0.25, +0.25, -0.5, +0.5, ...
__device__ __forceinline__ float lane_mult(int i)
{
    if (i == 0) return 0.f;
    const float m = (float)((i + 1) / 2) * 0.25f;
    return (i % 2 == 1) ? -m : m;
}

// edge_mode 1: project the n chain waypoints jointly (see the header). The whole warp calls it; true when every waypoint
// is on the region (the result is in sc.wp). Uniform result.
__device__ bool ctrace_joint(CShared& sc, int n, float s, const Tsr& t, const CParams& cp)
{
    const int l = threadIdx.x;
    const unsigned full = 0xffffffffu;
    int frozen = -1;                                    // waypoints 0..frozen are satisfied and no longer updated
    for (int it = 0;; ++it) {
        const bool active = l < n && l > frozen;
        bool okf = true;
        float q[N], q2[N];
        if (active) {
            for (int k = 0; k < N; ++k) q[k] = q2[k] = sc.wp[l][k];
            float e[NR];
            tsr_err(q, t, e);
            const float* prev = l == 0 ? sc.from : sc.wp[l - 1];
            const float d = fmaxf(sqrtf(dsqd(q, prev)), 1e-9f);
            const float serr = d - s;                   // spacing error; a gap above 2 s counts as unsatisfied
            const bool sat = tsr_satisfied(e, t);
            okf = sat && !(cp.smooth && serr > s);
            if (!sat) {
                float J[NR * N];
                tsr_eval(q, t, cp.proj.hold != 0, e, J);
                tsr_step(q2, e, J, cp.proj);
            }
            if (cp.smooth)
                for (int k = 0; k < N; ++k)
                    q2[k] = fminf(fmaxf(q2[k] - cp.smooth_rate * (q[k] - prev[k]) / d * serr, LO[k]), HI[k]);
        }
        const unsigned bad = __ballot_sync(full, active && !okf);
        if (!bad) return true;
        if (it >= cp.proj.max_iters) return false;
        const int fn = cp.freeze ? (int)__ffs(bad) - 2 : frozen;     // lanes <= fn are satisfied: all before the first bad one
        __syncwarp();                                   // every lane has read its neighbour's waypoint
        if (active && l > fn)
            for (int k = 0; k < N; ++k) sc.wp[l][k] = q2[k];
        frozen = fn;
        __syncwarp();
    }
}

// Extend from sc.from toward sc.target (at `dist`) by at most `range`, staying on the region. The whole warp calls it.
// Fills sc.wp[0..n), sc.emit[0..m) and returns TRAPPED / ADVANCED / REACHED (uniform).
__device__ int csteer(CShared& sc, float dist, float range, const World& w, const Tsr& t, const CParams& cp)
{
    const int l = threadIdx.x;
    const unsigned full = 0xffffffffu;
    const bool reach = dist < range;
    const float step = reach ? dist : range;
    const float k0 = reach ? 1.f : range / dist;

    // candidates: lane l projects its own perturbed step; the lowest converged lane wins
    float q[N];
    for (int k = 0; k < N; ++k) q[k] = 0.f;
    bool ok = false;
    if (l < min(max(cp.lanes, 1), NT)) {
        const float m = 1.f + cp.perturb * lane_mult(l);
        for (int k = 0; k < N; ++k) q[k] = sc.from[k] + (sc.target[k] - sc.from[k]) * (k0 * m);
        ProjParams pp = cp.proj;
        pp.drift2 = 4.f * step * step + 1e-6f;
        ok = tsr_project(q, t, pp) && dsqd(q, sc.from) <= 4.f * step * step;
    }
    const unsigned bal = __ballot_sync(full, ok);
    if (!bal) return TRAPPED;
    const int win = (int)__ffs(bal) - 1;
    float next[N];
    for (int k = 0; k < N; ++k) next[k] = __shfl_sync(full, q[k], win);

    // the chain from -> next, one waypoint per lane
    const float d2 = dsqd(next, sc.from);
    const int n = min(max((int)ceilf(sqrtf(d2) / cp.spacing), 1), NT);
    const float s = sqrtf(d2) / (float)n;               // chord spacing
    const unsigned maskn = n >= 32 ? full : ((1u << n) - 1u);
    if (l < n) {
        const float a = (float)(l + 1) / (float)n;
        for (int k = 0; k < N; ++k) sc.wp[l][k] = sc.from[k] + (next[k] - sc.from[k]) * a;
    }
    __syncwarp();
    bool good = true;
    if (cp.edge_mode == 1) {
        good = ctrace_joint(sc, n, s, t, cp);
    } else {
        if (l < n) {
            float qq[N];
            for (int k = 0; k < N; ++k) qq[k] = sc.wp[l][k];
            ProjParams pp = cp.proj;
            pp.drift2 = 4.f * d2;
            good = tsr_project(qq, t, pp);
            for (int k = 0; k < N; ++k) sc.wp[l][k] = qq[k];
        }
        __syncwarp();
        good = __all_sync(full, good);
    }
    if (!good) return TRAPPED;
    bool cont = true;                                   // the projection must not tear the chain apart
    if (l < n) cont = dsqd(sc.wp[l], l == 0 ? sc.from : sc.wp[l - 1]) <= cp.gap_slack * cp.gap_slack * s * s * 1.0001f;
    if (!__all_sync(full, cont)) return TRAPPED;
    bool free = true;                                   // all waypoints together: any hit rejects the edge
    if (l < n) {
        float qq[N];
        for (int k = 0; k < N; ++k) qq[k] = sc.wp[l][k];
        free = fkcc(qq, w, maskn);
    }
    if (!__all_sync(full, free)) return TRAPPED;

    if (l == 0) {
        const int Lm = min(max(cp.lanes, 1), NT);
        const int stride = cp.emit_all ? (n + Lm - 1) / Lm : n;      // about `lanes` waypoints, always the last
        int m = 0;
        for (int i = 0; i < n; ++i)
            if ((i + 1) % stride == 0 || i == n - 1) sc.emit[m++] = i;
        sc.m = m;
        sc.status = (reach && dsqd(sc.wp[n - 1], sc.target) < cp.reached2) ? REACHED : ADVANCED;
    }
    __syncthreads();
    return sc.status;
}

// Claim sc.m slots in tree t with one atomicAdd, store the emitted waypoints as a parent chain from `parent`, publish them
// together. False when the tree is full.
__device__ bool add_chain(PState* S, int t, float* ndt, int* part, int* rdyt, const CShared& sc, int parent,
                          const Params& prm, int* s_idx, int& last)
{
    const int tid = threadIdx.x;
    const int m = sc.m;
    if (tid == 0) {
        const int idx0 = atomicAdd(&S->cnt[t], m);
        *s_idx = idx0;
        if (idx0 + m > prm.max_samples) atomicCAS(&S->solved, 0, -1);
    }
    __syncthreads();
    const int idx0 = *s_idx;
    if (idx0 + m > prm.max_samples) return false;
    if (tid < m) {
        const int idx = idx0 + tid;
        const float* src = sc.wp[sc.emit[tid]];
        for (int k = 0; k < N; ++k) ndt[(size_t)idx * N + k] = src[k];
        *(volatile int*)&part[idx] = tid == 0 ? parent : idx - 1;
    }
    __threadfence();
    __syncthreads();
    if (tid < m) *(volatile int*)&rdyt[idx0 + tid] = prm.epoch;     // only now, so no block sees half a chain
    last = idx0 + m - 1;
    return true;
}

// One thread per problem: project start and goal onto the region (solved = -2 when that fails) and set up the trees.
__global__ void k_cinit(const float* starts, const float* goals, const Tsr* tsrs, float* nodes, int* parents, int* ready,
                        PState* st, int* path_len, Params prm, CParams cp, int nP)
{
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= nP) return;
    float s[N], g[N];
    for (int k = 0; k < N; ++k) { s[k] = starts[(size_t)p * N + k]; g[k] = goals[(size_t)p * N + k]; }
    ProjParams pp = cp.proj;
    pp.drift2 = 0.f;
    const bool ok = tsr_project(s, tsrs[p], pp) && tsr_project(g, tsrs[p], pp);
    for (int t = 0; t < 2; ++t) {
        const size_t base = (size_t)(p * 2 + t) * prm.max_samples;
        const float* src = t == 0 ? s : g;
        for (int k = 0; k < N; ++k) nodes[base * N + k] = src[k];
        parents[base] = 0;
        ready[base] = prm.epoch;
        st[p].cnt[t] = 1;
    }
    st[p].solved = ok ? 0 : -2;
    st[p].iters = 0;
    path_len[p] = 0;
}

__global__ void __launch_bounds__(NT)
k_crrtc(const float* env, const float* dmin, const int* env_n, const Tsr* tsrs, float* nodes, int* parents, int* ready,
        PState* st, float* path, int* path_len, Params prm, CParams cp)
{
    const int tid = threadIdx.x;
    const int p = blockIdx.x / prm.B, bl = blockIdx.x % prm.B;
    PState* S = st + p;
    const size_t ms = prm.max_samples;
    float* nd[2] = {nodes + (size_t)(p * 2) * ms * N, nodes + (size_t)(p * 2 + 1) * ms * N};
    int* par[2] = {parents + (size_t)(p * 2) * ms, parents + (size_t)(p * 2 + 1) * ms};
    int* rdy[2] = {ready + (size_t)(p * 2) * ms, ready + (size_t)(p * 2 + 1) * ms};
    World w;
    w.sph = env + (size_t)p * (prm.MS * 4 + prm.MB * 15 + prm.MC * 7);
    w.box = w.sph + prm.MS * 4;
    w.cap = w.box + prm.MB * 15;
    const float* dm = dmin + (size_t)p * (prm.MS + prm.MB + prm.MC);
    w.smin = dm; w.bmin = dm + prm.MS; w.cmin = dm + prm.MS + prm.MB;
    w.ns = env_n[3 * p]; w.nb = env_n[3 * p + 1]; w.nc = env_n[3 * p + 2];
    const unsigned rs = prm.seed ^ ((unsigned)(prm.p0 + p) * 0x9E3779B1u) ^ ((unsigned)bl * 0x85EBCA6Bu);

    __shared__ Tsr tsr;                                  // this problem's region, read often
    __shared__ CShared sc;
    __shared__ float sd[NT / 32];
    __shared__ int si[NT / 32], s_t, s_idx, s_stop, s_cnt, prime_order[16];
    for (int i = tid; i < (int)(sizeof(Tsr) / sizeof(float)); i += NT)
        ((float*)&tsr)[i] = ((const float*)&tsrs[p])[i];

    if (tid == 0) {                                      // per-block Halton: block 0 in order, the others permuted
        for (int i = 0; i < 16; ++i) prime_order[i] = i;
        if (bl != 0)
            for (int i = 15; i > 0; --i) {
                const int j = min((int)(urand(rs, 77u, i) * (i + 1)), i);
                const int tmp = prime_order[i]; prime_order[i] = prime_order[j]; prime_order[j] = tmp;
            }
    }
    __syncthreads();

    for (int it = 1;; ++it) {
        if (tid == 0) {
            s_stop = *(volatile int*)&S->solved != 0;
            if (it > prm.max_iters) {
                atomicCAS(&S->solved, 0, -1);
                S->iters = prm.max_iters;
                s_stop = 1;
            }
            const int c0 = min(*(volatile int*)&S->cnt[0], prm.max_samples);
            const int c1 = min(*(volatile int*)&S->cnt[1], prm.max_samples);
            const float ratio = c0 / (float)(c0 + c1);       // balancing: as rrtc.cuh (pRRTC mode 1)
            if (it == 1) s_t = bl < prm.B / 2 ? 0 : 1;
            else if (abs(c0 - c1) < 1.25f * prm.B) s_t = bl < prm.B * (1.f - ratio) ? 0 : 1;
            else s_t = ratio < TREE_RATIO ? 0 : 1;
        }
        __syncthreads();
        if (s_stop) return;
        const int t = s_t, o = 1 - t;

        // direct attempt: block 0's first iteration tries the start root -> goal root connect
        const bool direct = (it == 1 && bl == 0 && t == 0);
        float bd;
        int bi = 0, cur = 0;
        if (direct) {
            if (tid < N) sc.cfg[tid] = __ldcg(&nd[0][tid]);
            __syncthreads();
        } else {
            if (tid < N) sc.target[tid] = LO[tid] + halton(it, PRIMES[prime_order[tid]]) * (HI[tid] - LO[tid]);
            if (tid == 0) s_cnt = min(*(volatile int*)&S->cnt[t], prm.max_samples);
            __syncthreads();
            nearest(nd[t], rdy[t], s_cnt, sc.target, prm.epoch, sd, si, bd, bi);
            if (tid < N) sc.from[tid] = __ldcg(&nd[t][(size_t)bi * N + tid]);
            __syncthreads();
            const float dist = sqrtf(bd);
            if (dist < 1e-6f) continue;
            if (csteer(sc, dist, prm.range, w, tsr, cp) == TRAPPED) continue;
            if (!add_chain(S, t, nd[t], par[t], rdy[t], sc, bi, prm, &s_idx, cur)) continue;
            if (tid < N) sc.cfg[tid] = sc.wp[sc.emit[sc.m - 1]][tid];
            __syncthreads();
        }

        // connect: hops toward the nearest node of the other tree (the goal root for the direct attempt)
        if (!direct) {
            if (tid == 0) s_cnt = min(*(volatile int*)&S->cnt[o], prm.max_samples);
            __syncthreads();
            nearest(nd[o], rdy[o], s_cnt, sc.cfg, prm.epoch, sd, si, bd, bi);
        }
        const int on = direct ? 0 : bi;
        if (tid < N) sc.target[tid] = __ldcg(&nd[o][(size_t)on * N + tid]);
        __syncthreads();
        const int hops = max(1, (int)ceilf(cp.slack * sqrtf(dsqd(sc.target, sc.cfg)) / prm.range));
        bool connected = false;
        for (int e = 0; e < hops; ++e) {
            const float rem = sqrtf(dsqd(sc.target, sc.cfg));
            const bool last = e == hops - 1 || rem <= prm.range;
            if (tid < N) sc.from[tid] = sc.cfg[tid];
            __syncthreads();
            const int st_ = csteer(sc, rem, last ? FLT_MAX : prm.range, w, tsr, cp);
            if (st_ == TRAPPED) break;
            if (cp.closer && dsqd(sc.wp[sc.emit[sc.m - 1]], sc.target) >= rem * rem - cp.closer2) break;
            int ni;
            if (!add_chain(S, t, nd[t], par[t], rdy[t], sc, cur, prm, &s_idx, ni)) break;
            cur = ni;
            if (tid < N) sc.cfg[tid] = sc.wp[sc.emit[sc.m - 1]][tid];
            __syncthreads();
            if (st_ == REACHED) { connected = true; break; }
        }
        if (connected && tid == 0 && atomicCAS(&S->solved, 0, 1) == 0) {
            const int ia = t == 0 ? cur : on, ib = t == 0 ? on : cur;    // node in tree 0 (start) / tree 1 (goal)
            S->ia = ia; S->ib = ib; S->iters = it;
            int la = 0, lb = 0;
            for (int i = ia;; i = *(volatile int*)&par[0][i]) { ++la; if (i == 0) break; }
            for (int i = ib;; i = *(volatile int*)&par[1][i]) { ++lb; if (i == 0) break; }
            float* P = path + (size_t)p * MAXP * N;
            if (la + lb > MAXP) {
                path_len[p] = -1;
            } else {
                int k = la;
                for (int i = ia;; i = *(volatile int*)&par[0][i]) {
                    --k;
                    for (int d = 0; d < N; ++d) P[(size_t)k * N + d] = __ldcg(&nd[0][(size_t)i * N + d]);
                    if (i == 0) break;
                }
                k = la;
                for (int i = ib;; i = *(volatile int*)&par[1][i]) {
                    for (int d = 0; d < N; ++d) P[(size_t)k * N + d] = __ldcg(&nd[1][(size_t)i * N + d]);
                    ++k;
                    if (i == 0) break;
                }
                path_len[p] = la + lb;
            }
        }
        __syncthreads();
    }
}
