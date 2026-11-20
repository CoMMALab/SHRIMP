// SHRIMP: GPU RRT-Connect in one CUDA kernel (the algorithm of pRRTC, arXiv 2503.06757, written fresh).
// Needs CRAB's robot header first (N, LO, HI, World, fkcc; see crab_codegen/robot_header.py).
//
// One planning problem = `B` thread blocks of 32 threads (one warp); many problems per launch.
//   * every block repeatedly: samples a config (Halton, per-block shuffled prime bases), finds its nearest
//     node in one tree (strided scan + warp argmin), steers toward it by <= `range`, validates the edge at a
//     fixed resolution (RES: ceil(len / RES) waypoints, one thread each, max NT), adds the node, then tries
//     to CONNECT to the nearest node of the other tree in <= `range` hops (same edge test).
//   * trees live in global memory, per problem: nodes (N floats), parents, and a `ready` epoch flag per node
//     (a node is visible only when ready == the launch's epoch, so nothing needs clearing between launches).
//     New nodes are claimed with atomicAdd; the first block that connects wins via atomicCAS(solved) and
//     walks both trees on the device to write the start->goal path.
//   * tree balancing is pRRTC's mode 1 (blocks split by size ratio while the trees are similar, else all
//     grow the smaller tree). Block 0's first iteration is the direct start->goal attempt (VAMP RRTC).
// Env layout per problem (floats): sph[MS][4], box[MB][15], cap[MC][7]; counts int32 (ns, nb, nc).

constexpr int NT = 32;          // threads per block (one warp) = max waypoints checked per edge
constexpr float RES = 1.f / 32.f;   // edge resolution (rad, joint-space L2): ceil(length / RES) waypoints
constexpr int MAXP = 512;       // max path nodes written per problem

struct Params {
    int B, max_samples, max_iters, p0, MS, MB, MC, epoch;
    float range;
    unsigned seed;
};
struct PState { int cnt[2]; int solved; int ia, ib, iters, pad; };

__device__ __forceinline__ float urand(unsigned a, unsigned b, unsigned c)
{
    unsigned x = a * 0x9E3779B1u ^ (b + 0x7F4A7C15u) * 0x85EBCA6Bu ^ (c + 0x165667B1u) * 0xC2B2AE35u;
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
    return (x >> 8) * (1.f / 16777216.f);
}

constexpr float TREE_RATIO = 0.5f;
__constant__ int PRIMES[16] = {3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59};
static_assert(N <= 16, "Halton bases: at most 16 joints");

// Halton radical inverse of i >= 1 in `base`.
__device__ __forceinline__ float halton(unsigned i, unsigned base)
{
    float f = 1.f, r = 0.f;
    while (i > 0) { f /= base; r += f * (i % base); i /= base; }
    return r;
}

// Whole block: is the straight edge from -> from + delta free? n = ceil(|delta| / RES) waypoints (<= NT),
// thread t < n checks waypoint t + 1 (the last one is the endpoint); the other threads just vote "free".
__device__ bool edge_ok(const float* from, const float* delta, const World& w)
{
    float len2 = 0.f;
    for (int k = 0; k < N; ++k) len2 += delta[k] * delta[k];
    const int n = min(NT, max(1, (int)ceilf(sqrtf(len2) / RES)));
    bool ok = true;
    if ((int)threadIdx.x < n) {
        float q[N];
        const float a = (threadIdx.x + 1) / (float)n;
        for (int k = 0; k < N; ++k) q[k] = from[k] + a * delta[k];
        const unsigned mask = n >= 32 ? 0xffffffffu : ((1u << n) - 1u);     // the lanes checking this edge
        ok = fkcc(q, w, mask);
    }
    return __syncthreads_and(ok);
}

__device__ __forceinline__ void block_argmin(float d, int i, float* sd, int* si, float& od, int& oi)
{
    for (int o = 16; o > 0; o >>= 1) {
        const float d2 = __shfl_down_sync(0xffffffffu, d, o);
        const int i2 = __shfl_down_sync(0xffffffffu, i, o);
        if (d2 < d) { d = d2; i = i2; }
    }
    __syncthreads();
    if ((threadIdx.x & 31) == 0) { sd[threadIdx.x >> 5] = d; si[threadIdx.x >> 5] = i; }
    __syncthreads();
    od = sd[0]; oi = si[0];
    for (int w = 1; w < NT / 32; ++w)
        if (sd[w] < od) { od = sd[w]; oi = si[w]; }
}

// Nearest READY node (ready == epoch) among the first `count`; result valid in every thread.
__device__ void nearest(const float* nd, const int* rdy, int count, const float* q, int epoch, float* sd,
                        int* si, float& bd, int& bi)
{
    float d = FLT_MAX;
    int b = 0;
    for (int i = threadIdx.x; i < count; i += NT) {
        if (*(volatile const int*)&rdy[i] != epoch) continue;
        float s = 0.f;
        for (int k = 0; k < N; ++k) {
            const float x = __ldcg(nd + (size_t)i * N + k) - q[k];
            s += x * x;
        }
        if (s < d) { d = s; b = i; }
    }
    block_argmin(d, b, sd, si, bd, bi);
}

// Claim a slot in tree t, store `c` with parent `parent`, publish it. False when the tree is full.
__device__ bool add_node(PState* S, int t, float* ndt, int* part, int* rdyt, const float* c, int parent,
                         const Params& prm, int* s_idx, int& out)
{
    const int tid = threadIdx.x;
    if (tid == 0) {
        const int idx = atomicAdd(&S->cnt[t], 1);
        *s_idx = idx;
        if (idx < prm.max_samples) *(volatile int*)&part[idx] = parent;
        else atomicCAS(&S->solved, 0, -1);
    }
    __syncthreads();
    const int idx = *s_idx;
    if (idx >= prm.max_samples) return false;
    if (tid < N) ndt[(size_t)idx * N + tid] = c[tid];
    __threadfence();
    __syncthreads();
    if (tid == 0) *(volatile int*)&rdyt[idx] = prm.epoch;
    out = idx;
    return true;
}

__global__ void k_init(const float* starts, const float* goals, float* nodes, int* parents, int* ready,
                       PState* st, int* path_len, Params prm, int nP)
{
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= nP) return;
    for (int t = 0; t < 2; ++t) {
        const size_t base = (size_t)(p * 2 + t) * prm.max_samples;
        const float* src = (t == 0 ? starts : goals) + (size_t)p * N;
        for (int k = 0; k < N; ++k) nodes[base * N + k] = src[k];
        parents[base] = 0;
        ready[base] = prm.epoch;
        st[p].cnt[t] = 1;
    }
    st[p].solved = 0;
    st[p].iters = 0;
    path_len[p] = 0;
}

__global__ void __launch_bounds__(NT)
k_rrtc(const float* env, const float* dmin, const int* env_n, float* nodes, int* parents, int* ready, PState* st, float* path,
       int* path_len, Params prm)
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

    __shared__ float cfg[N], near_cfg[N], stepv[N], vecs[N], sd[NT / 32];
    __shared__ int si[NT / 32], s_t, s_idx, s_stop, s_cnt, prime_order[16];

#ifdef SHRIMP_PROF       // cycles in nearest / edge_ok / add_node and in the whole block (flushed at exit)
    long long pc[3] = {0, 0, 0};
    const long long pstart = clock64();
#define PROF(k, ...) { const long long pt_ = clock64(); __VA_ARGS__; pc[k] += clock64() - pt_; }
#else
#define PROF(k, ...) __VA_ARGS__
#endif

    // per-block Halton: block 0 uses the primes in order, the others a random permutation of them
    if (tid == 0) {
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
            // tree balancing (pRRTC mode 1): first iteration split the blocks in half; while the trees are
            // within 1.25 x blocks of each other split the blocks by the size ratio; otherwise every block
            // grows the smaller tree (ratio = tree 0's share of all nodes; < TREE_RATIO -> grow tree 0)
            const float ratio = c0 / (float)(c0 + c1);
            if (it == 1) s_t = bl < prm.B / 2 ? 0 : 1;
            else if (abs(c0 - c1) < 1.25f * prm.B) s_t = bl < prm.B * (1.f - ratio) ? 0 : 1;
            else s_t = ratio < TREE_RATIO ? 0 : 1;
        }
        __syncthreads();
        if (s_stop) {
#ifdef SHRIMP_PROF
            if (tid == 0) {
                atomicAdd(&g_prof[4], (unsigned long long)pc[0]);
                atomicAdd(&g_prof[5], (unsigned long long)pc[1]);
                atomicAdd(&g_prof[6], (unsigned long long)pc[2]);
                atomicAdd(&g_prof[7], (unsigned long long)(clock64() - pstart));
                atomicAdd(&g_prof[8], 1ull);
            }
#endif
            return;
        }
        const int t = s_t, o = 1 - t;

        // Direct attempt (VAMP RRTC's first move, as prrax): block 0's first iteration skips sampling and
        // tries to connect the start root straight to the goal root; easy problems get a straight path.
        const bool direct = (it == 1 && bl == 0 && t == 0);             // block 0 has t == 0 on iteration 1 (B >= 2)
        float bd;
        int bi = 0, cur = 0;
        if (direct) {
            if (tid < N) cfg[tid] = __ldcg(&nd[0][tid]);                // the start root
            __syncthreads();
        } else {
            if (tid < N) cfg[tid] = LO[tid] + halton(it, PRIMES[prime_order[tid]]) * (HI[tid] - LO[tid]);
            if (tid == 0) s_cnt = min(*(volatile int*)&S->cnt[t], prm.max_samples);
            __syncthreads();
            PROF(0, nearest(nd[t], rdy[t], s_cnt, cfg, prm.epoch, sd, si, bd, bi));
            if (tid < N) near_cfg[tid] = __ldcg(&nd[t][(size_t)bi * N + tid]);
            __syncthreads();
            const float dist = sqrtf(bd);
            if (dist < 1e-6f) continue;
            const float scale = fminf(1.f, prm.range / dist);
            if (tid < N) {
                cfg[tid] = near_cfg[tid] + (cfg[tid] - near_cfg[tid]) * scale;
                stepv[tid] = cfg[tid] - near_cfg[tid];                  // the edge vector
            }
            __syncthreads();
            bool ok1, ok2;
            PROF(1, ok1 = edge_ok(near_cfg, stepv, w));
            if (!ok1) continue;
            PROF(2, ok2 = add_node(S, t, nd[t], par[t], rdy[t], cfg, bi, prm, &s_idx, cur));
            if (!ok2) continue;
        }

        // connect: walk toward the nearest node of the other tree (the goal root for the direct attempt)
        // in <= range hops
        if (!direct) {
            if (tid == 0) s_cnt = min(*(volatile int*)&S->cnt[o], prm.max_samples);
            __syncthreads();
            PROF(0, nearest(nd[o], rdy[o], s_cnt, cfg, prm.epoch, sd, si, bd, bi));
        }
        const int on = direct ? 0 : bi;
        if (tid < N) near_cfg[tid] = __ldcg(&nd[o][(size_t)on * N + tid]);   // the target
        __syncthreads();
        float d2 = 0.f;
        for (int k = 0; k < N; ++k) d2 += (near_cfg[k] - cfg[k]) * (near_cfg[k] - cfg[k]);
        const int hops = max(1, (int)ceilf(sqrtf(d2) / prm.range));
        if (tid < N) {
            vecs[tid] = (near_cfg[tid] - cfg[tid]) / hops;
            stepv[tid] = vecs[tid];                                     // the hop's edge vector
        }
        __syncthreads();
        bool connected = true;
        for (int e = 0; e < hops; ++e) {
            bool ok1, ok2;
            PROF(1, ok1 = edge_ok(cfg, stepv, w));
            if (!ok1) { connected = false; break; }
            if (tid < N) cfg[tid] += vecs[tid];
            int ni;
            PROF(2, ok2 = add_node(S, t, nd[t], par[t], rdy[t], cfg, cur, prm, &s_idx, ni));
            if (!ok2) { connected = false; break; }
            cur = ni;
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
