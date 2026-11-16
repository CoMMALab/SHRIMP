// SHRIMP: CPU RRT-Connect, scalar, one problem per call (single-threaded; callers parallelise over problems).
// Needs CRAB's CPU robot header first (N, LO, HI, World, scalar fkcc; crab_codegen/robot_header.py).
//
// Same algorithm and conventions as cuda/rrtc.cuh so the two are comparable:
//   * iteration 1 is the direct attempt: connect the start root to the goal root in hops of <= range;
//   * then each iteration grows the SMALLER tree (ties alternate): Halton sample, nearest node (kd-tree),
//     steer by <= range, validate the edge, add the node, then connect to the nearest node of the other tree
//     in <= range hops (hop nodes are added to the growing tree);
//   * edge validation: n = ceil(len / RES) waypoints at fixed resolution (no cap), checked coarse-to-fine
//     (endpoint, then midpoints down to single steps) with exit at the first collision.
// Scalar by design: SIMD goes in edge_ok (batch waypoints) and the kd-tree's distance loop without touching plan().
// Env layout per problem (floats): sph[MS][4], box[MB][15], cap[MC][7]; counts int32 (ns, nb, nc).
#include <algorithm>
#include <cmath>
#include <vector>
// (kdtree.hpp is concatenated before this file by shrimp/rrtc.py)

constexpr float RES = 1.f / 32.f;    // edge resolution (rad, joint-space L2)
constexpr int MAXP = 512;            // max path nodes written per problem
constexpr unsigned PRIMES[16] = {3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59};
static_assert(N <= 16, "Halton bases: at most 16 joints");

struct Params {
    int max_iters, max_samples;
    float range;
    unsigned seed;
    float epsilon;          // nearest-neighbour approximation: neighbour within (1 + epsilon) of the true nearest
    float weights[N];       // per-joint weights of the nearest-neighbour metric (steering stays plain L2)
};
struct Result { int solved, iters, len; };       // solved: 1 yes, 0 out of iterations, -1 a tree filled up

static inline float halton(unsigned i, unsigned base)
{
    float f = 1.f, r = 0.f;
    while (i > 0) { f /= base; r += f * (i % base); i /= base; }
    return r;
}

// A tree's nodes (configs + RRT parent) plus a KDTree (kdtree.hpp) over them for nearest-node queries. Node ids
// are the insertion order in both.
struct Tree {
    std::vector<float> q;
    std::vector<int> par;
    KDTree<N> kd;
    int size() const { return (int)par.size(); }
    const float* at(int i) const { return q.data() + (size_t)i * N; }
    void init(const Params& prm)
    {
        q.reserve((size_t)prm.max_samples * N);
        par.reserve(prm.max_samples);
        kd.reserve(prm.max_samples);
        kd.set_epsilon(prm.epsilon);
        kd.set_weights(prm.weights);
    }
    void add(const float* c, int parent)
    {
        kd.insert(size(), c);
        q.insert(q.end(), c, c + N);
        par.push_back(parent);
    }
};

// Nearest node under the tree's (weighted, approximate) metric.
static int nearest(const Tree& t, const float* c) { float d2; return t.kd.nearest(c, d2); }

static thread_local long g_fk = 0;   // fkcc calls made by edge_ok on this thread (instrumentation)

// Is the straight edge from -> from + delta free? Waypoints j = 1..n at from + (j / n) delta, j = n first,
// then the odd multiples of the largest power of two below n, halving down to 1.
static bool edge_ok(const float* from, const float* delta, const World& w)
{
    float len2 = 0.f;
    for (int k = 0; k < N; ++k) len2 += delta[k] * delta[k];
    const int n = std::max(1, (int)std::ceil(std::sqrt(len2) / RES));
    float q[N];
    auto free_at = [&](int j) {
        const float a = (float)j / (float)n;
        for (int k = 0; k < N; ++k) q[k] = from[k] + a * delta[k];
        ++g_fk;
        return fkcc(q, w);
    };
    if (!free_at(n)) return false;
    int s = 1;
    while (s < n) s <<= 1;
    for (s >>= 1; s >= 1; s >>= 1)
        for (int j = s; j < n; j += 2 * s)
            if (!free_at(j)) return false;
    return true;
}

// Walk cfg toward `target` in hops of <= range, adding each hop node to `tr` (parent chain from `cur`).
// 1 reached, 0 blocked, -1 tree full. cfg and cur are advanced.
static int walk(Tree& tr, int& cur, float* cfg, const float* target, const World& w, const Params& prm)
{
    float d2 = 0.f;
    for (int k = 0; k < N; ++k) d2 += (target[k] - cfg[k]) * (target[k] - cfg[k]);
    const int hops = std::max(1, (int)std::ceil(std::sqrt(d2) / prm.range));
    float vec[N];
    for (int k = 0; k < N; ++k) vec[k] = (target[k] - cfg[k]) / hops;
    for (int e = 0; e < hops; ++e) {
        if (!edge_ok(cfg, vec, w)) return 0;
        if (tr.size() >= prm.max_samples) return -1;
        for (int k = 0; k < N; ++k) cfg[k] += vec[k];
        tr.add(cfg, cur);
        cur = tr.size() - 1;
    }
    return 1;
}

// Plan start -> goal. On success writes the path (start first) to path[MAXP][N] and its length to Result.len
// (-1 when longer than MAXP).
static Result plan(const float* start, const float* goal, const World& w, const Params& prm, float* path)
{
    Tree T[2];
    for (auto& t : T) t.init(prm);
    T[0].add(start, 0);
    T[1].add(goal, 0);
    const unsigned skip = prm.seed * 9973u;
    Result r{0, 0, 0};
    for (int it = 1; it <= prm.max_iters; ++it) {
        r.iters = it;
        float cfg[N];
        int t, o, cur = 0, on;
        if (it == 1) {                                       // direct attempt: start root -> goal root
            t = 0; o = 1; on = 0;
            for (int k = 0; k < N; ++k) cfg[k] = start[k];
        } else {
            const int c0 = T[0].size(), c1 = T[1].size();
            t = c0 < c1 ? 0 : c1 < c0 ? 1 : (it & 1);
            o = 1 - t;
            for (int k = 0; k < N; ++k) cfg[k] = LO[k] + halton(skip + it, PRIMES[k]) * (HI[k] - LO[k]);
            const int nb = nearest(T[t], cfg);
            const float* near = T[t].at(nb);
            float d2 = 0.f;                                  // steering uses plain L2, not the weighted metric
            for (int k = 0; k < N; ++k) d2 += (cfg[k] - near[k]) * (cfg[k] - near[k]);
            const float dist = std::sqrt(d2);
            if (dist < 1e-6f) continue;
            const float scale = std::min(1.f, prm.range / dist);
            float delta[N];
            for (int k = 0; k < N; ++k) { delta[k] = (cfg[k] - near[k]) * scale; cfg[k] = near[k] + delta[k]; }
            if (!edge_ok(near, delta, w)) continue;
            if (T[t].size() >= prm.max_samples) { r.solved = -1; return r; }
            T[t].add(cfg, nb);
            cur = T[t].size() - 1;
            on = nearest(T[o], cfg);
        }
        const int res = walk(T[t], cur, cfg, T[o].at(on), w, prm);
        if (res < 0) { r.solved = -1; return r; }
        if (res == 0) continue;
        const int ia = t == 0 ? cur : on, ib = t == 0 ? on : cur;      // node in tree 0 (start) / tree 1 (goal)
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
