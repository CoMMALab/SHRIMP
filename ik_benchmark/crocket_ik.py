"""Baseline: crocket's traced IK on the hard RBY1 targets (crocket env; GPU + nvcc).
Run: python ik_benchmark/crocket_ik.py [--chain arm|full|bimanual|mbm] [--method lm|sqp|lmmulti]
                                       [--seeds 8,32,128] [--max-iter 60] [--warm]

Methods:
  lm       crocket's `ik_lm_step` in its accept/reject loop (benchmarks/ik/crocket_ik.py); thread
           and warp tiers. ONE tool only.
  sqp      GRiM sqp_ik's active-set formulation traced by crocket (benchmarks/ik/crocket_ik_sqp.py);
           mixed and f32 precisions. ONE tool only (its QP has 6 rows).
  lmmulti  my Levenberg-Marquardt loop around crocket's `ik_geometry` kernel (stacked pose residual
           and Jacobian of any number of frames), weights 50/10 and accept/reject like `lm`; the
           only crocket method that handles bimanual. Thread tier.
  lmcoll   lmmulti plus collision: hinge rows for every robot sphere vs the problem's obstacles
           (MBM spheres/boxes/capsules) and vs other links' spheres (SRDF pairs), gradients from
           ik_geometry's sphere centres + joint twists. Stops when the pose is met AND the smallest
           gap is >= 1 mm. Thread tier.
The kernels are crocket's; the driver differs from its benchmarks:
  * Only the chain's joints exist: a copy of the URDF marks every other joint `fixed` (crocket
    merges those), otherwise the steps would also move joints outside the problem.
  * One launch per target (S seeds, one thread each) for latency, plus one launch over all targets
    for throughput. Both are CUDA-event times over back-to-back launches (no host sync or Python).
  * All S solutions per target are saved; report.py counts a target solved if ANY seed passes.
"""

import argparse
import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

import problem

os.environ.setdefault("CROCKET_GRID_PATH", str(problem.REPO / "external/pyroffi/external/GRiM/external/GRiD"))
sys.path.insert(0, str(problem.REPO / "external/crocket/python"))

import crocket  # noqa: E402
from crocket import cuda  # noqa: E402

BUILD = problem.REPO / "ik_benchmark" / "build"
ARCH = os.environ.get("CROCKET_ARCH", "-arch=native")
POS_W, ORI_W = 50.0, 10.0
COLL_W, COLL_MARGIN, DONE_GAP = 2500.0, 0.005, 0.001   # lmcoll: hinge weight, hinge margin (m), min gap to stop
WARPS = 4
N_INNER = 10

# ---- shared: the host program (times latency per target and a full batch) -------------------
# TS = floats of target per problem (7 per tool); defined per build before this text.
MAIN = r"""
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

int main(int argc, char** argv)
{
    // <mode> <P> <S> <seeds.f32> <targets.f32> <out.f32> <latency_ms.f32>
    g_mode = argv[1];
    const int P = atoi(argv[2]), S = atoi(argv[3]), K = 5;
    std::vector<float> seeds((size_t)P * S * N), targets((size_t)P * TS), out((size_t)P * S * N), lat(P);
    FILE* f = fopen(argv[4], "rb");
    if (!f || fread(seeds.data(), sizeof(float), seeds.size(), f) != seeds.size()) return 1;
    fclose(f);
    f = fopen(argv[5], "rb");
    if (!f || fread(targets.data(), sizeof(float), targets.size(), f) != targets.size()) return 1;
    fclose(f);
    float *ds, *dt, *dout;
    cudaMalloc(&ds, seeds.size() * sizeof(float));
    cudaMalloc(&dt, targets.size() * sizeof(float));
    cudaMalloc(&dout, out.size() * sizeof(float));
    cudaMemcpy(ds, seeds.data(), seeds.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dt, targets.data(), targets.size() * sizeof(float), cudaMemcpyHostToDevice);

    for (int i = 0; i < 2; ++i) launch(P * S, S, ds, dt, dout);
    cudaDeviceSynchronize();
    cudaEvent_t a, e;
    cudaEventCreate(&a);
    cudaEventCreate(&e);

    std::vector<float> batch;
    for (int rep = 0; rep < 5; ++rep) {
        cudaEventRecord(a);
        for (int k = 0; k < K; ++k) launch(P * S, S, ds, dt, dout);
        cudaEventRecord(e);
        cudaEventSynchronize(e);
        float ms;
        cudaEventElapsedTime(&ms, a, e);
        batch.push_back(ms / K);
    }
    for (int p = 0; p < P; ++p) {  // one target (S seeds) per launch
        cudaEventRecord(a);
        for (int k = 0; k < K; ++k) launch(S, S, ds + (size_t)p * S * N, dt + (size_t)p * TS, dout + (size_t)p * S * N);
        cudaEventRecord(e);
        cudaEventSynchronize(e);
        float ms;
        cudaEventElapsedTime(&ms, a, e);
        lat[p] = ms / K;
    }
    launch(P * S, S, ds, dt, dout);  // results from the full batch
    cudaDeviceSynchronize();
    if (cudaGetLastError() != cudaSuccess) return 2;
    cudaMemcpy(out.data(), dout, out.size() * sizeof(float), cudaMemcpyDeviceToHost);
    f = fopen(argv[6], "wb");
    fwrite(out.data(), sizeof(float), out.size(), f);
    fclose(f);
    f = fopen(argv[7], "wb");
    fwrite(lat.data(), sizeof(float), lat.size(), f);
    fclose(f);
    std::nth_element(batch.begin(), batch.begin() + batch.size() / 2, batch.end());
    printf("%.6f\n", batch[batch.size() / 2]);
    return 0;
}
"""

# ---- LM: crocket's ik_lm_step inside an accept/reject loop (benchmarks/ik/crocket_ik.py) ----
LM_LOOP = r"""
namespace r = crocket::robots::robot;
#include <cstring>
constexpr int N = r::n_q;
constexpr float EPS_POS2 = 1e-4f * 1e-4f, EPS_ORI2 = 1e-3f * 1e-3f;
static const char* g_mode = "thread";

#define SOLVE(STEP, SYNC)                                                                    \
    float lam = LAM0;                                                                        \
    STEP(q, cur);                                                                            \
    for (int it = 0; it < MAX_ITER; ++it) {                                                  \
        if (cur[N + 1] < EPS_POS2 && cur[N + 2] < EPS_ORI2) break;                           \
        STEP(cur, nxt);                                                                      \
        if (nxt[N] < cur[N]) {                                                               \
            for (int i = 0; i < N; ++i) q[i] = cur[i];                                       \
            SYNC;                                                                            \
            float* t = cur; cur = nxt; nxt = t;                                              \
            lam = fmaxf(lam / 3.f, 1e-7f);                                                   \
        } else {                                                                             \
            lam = fminf(lam * 4.f, 1e7f);                                                    \
            STEP(q, cur);                                                                    \
        }                                                                                    \
    }

__global__ void k_thread(const float* __restrict__ seeds, const float* __restrict__ targets,
                         float* __restrict__ out, int n_seeds, int S)
{
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_seeds) return;
    const float* target = targets + 7 * (b / S);
    float q[N], buf_a[N + 3], buf_b[N + 3];
    float *cur = buf_a, *nxt = buf_b;
    for (int i = 0; i < N; ++i) q[i] = seeds[(size_t)b * N + i];
#define STEP_T(x, y) r::ik_lm_step_thread<float>(x, target, &lam, y, y + N, y + N + 1, y + N + 2, 1)
    SOLVE(STEP_T, (void)0)
    for (int i = 0; i < N; ++i) out[(size_t)b * N + i] = q[i];
}

__global__ void k_warp(const float* __restrict__ seeds, const float* __restrict__ targets,
                       float* __restrict__ out, int n_seeds, int S)
{
    // Shared memory: the step's constant tables (once per block), then per warp: its scratch and
    // q / cur / nxt. Every warp fills the tables itself (identical writes), as crocket's harness does.
    extern __shared__ float smem[];
    constexpr int TB = r::ik_lm_step_warp_tables<float>();
    const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int b = blockIdx.x * WARPS + w;
    if (b >= n_seeds) return;
    float* tables = smem;
    float* scratch = smem + TB + w * (r::ik_lm_step_warp_smem + 3 * (N + 3));
    float *q = scratch + r::ik_lm_step_warp_smem, *cur = q + N + 3, *nxt = cur + N + 3;
    const float* target = targets + 7 * (b / S);
    r::ik_lm_step_warp_load_tables<float>(tables, lane, 32);
    for (int i = lane; i < N; i += 32) q[i] = seeds[(size_t)b * N + i];
    __syncwarp();
#define STEP_W(x, y) do { r::ik_lm_step_warp<float>(x, target, &lam, y, y + N, y + N + 1, y + N + 2, scratch, tables, 1); __syncwarp(); } while (0)
    SOLVE(STEP_W, __syncwarp())
    for (int i = lane; i < N; i += 32) out[(size_t)b * N + i] = q[i];
}

void launch(int n, int S, const float* ds, const float* dt, float* dout)
{
    if (!strcmp(g_mode, "warp")) {
        const size_t smem = (r::ik_lm_step_warp_tables<float>() +
                             WARPS * (r::ik_lm_step_warp_smem + 3 * (N + 3))) * sizeof(float);
        static bool set = false;
        if (!set) { cudaFuncSetAttribute(k_warp, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem); set = true; }
        k_warp<<<(n + WARPS - 1) / WARPS, WARPS * 32, smem>>>(ds, dt, dout, n, S);
    } else {
        k_thread<<<(n + 63) / 64, 64>>>(ds, dt, dout, n, S);
    }
}
"""

# ---- SQP: GRiM sqp_ik's formulation traced by crocket (benchmarks/ik/crocket_ik_sqp.py) ----
SQP_LOOP = r"""
namespace r = crocket::robots::robot;
#include <cstring>
constexpr int N = r::n_q;
static const char* g_mode = "mixed";

template <class T, class Q>
__global__ void k_sqp(const float* __restrict__ seeds, const float* __restrict__ targets,
                      float* __restrict__ out, int n_seeds, int S)
{
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_seeds) return;
    T target[7];
    for (int i = 0; i < 7; ++i) target[i] = targets[7 * (b / S) + i];
    T cfg[N], best_cfg[N], delta[N], trial[N], lam = T(LAM0), cost, pe2, oe2, best_err;
    for (int i = 0; i < N; ++i) best_cfg[i] = cfg[i] = seeds[(size_t)b * N + i];
    r::ik_pose_cost_thread<T>(cfg, target, &best_err, 1);
    const T alphas[5] = {1, 0.5, 0.25, 0.1, 0.025};
    for (int it = 0; it < MAX_ITER; ++it) {
        T Jw[6 * NC], rr[6];
        r::ik_ls_system_thread<T>(cfg, target, Jw, rr, &cost, &pe2, &oe2, 1);
        if (pe2 < T(EPS_POS * EPS_POS) && oe2 < T(EPS_ORI * EPS_ORI)) break;

        Q Jq[6 * NC], rq[6], lq = Q(lam), H[NC * NC], g[NC], sc[NC], zero[NC], p[NC], pb[NC], act[NC];
        for (int i = 0; i < 6 * NC; ++i) Jq[i] = Jw[i];
        for (int i = 0; i < 6; ++i) rq[i] = rr[i];
        r::ik_sqp_system_thread<Q>(Jq, rq, &lq, H, g, sc, 1);

        T lb[NC], ub[NC], ps[NC];
        for (int i = 0; i < NC; ++i) {
            lb[i] = (T(LO[COLS[i]]) - cfg[COLS[i]]) * T(sc[i]);
            ub[i] = (T(HI[COLS[i]]) - cfg[COLS[i]]) * T(sc[i]);
            zero[i] = 0;
        }
        r::ik_masked_solve_thread<Q>(H, g, zero, zero, p, 1);
        for (int i = 0; i < NC; ++i) ps[i] = fmin(fmax(T(p[i]), lb[i]), ub[i]);

        for (int k = 0; k < N_INNER; ++k) {
            for (int i = 0; i < NC; ++i) {
                const bool a = ps[i] <= lb[i] + T(1e-8) || ps[i] >= ub[i] - T(1e-8);
                act[i] = a;
                pb[i] = a ? Q(ps[i]) : Q(0);
            }
            r::ik_masked_solve_thread<Q>(H, g, pb, act, p, 1);
            for (int i = 0; i < NC; ++i)
                ps[i] = act[i] > Q(0.5) ? T(pb[i]) : fmin(fmax(T(p[i]), lb[i]), ub[i]);
        }
        for (int i = 0; i < N; ++i) delta[i] = 0;
        for (int i = 0; i < NC; ++i) delta[COLS[i]] = ps[i] / T(sc[i]);

        const T pn = sqrt(pe2), on = sqrt(oe2);
        const T R = (pn > T(1e-2) || on > T(0.6)) ? T(0.38) : (pn > T(1e-3) || on > T(0.25)) ? T(0.22)
                  : (pn > T(2e-4) || on > T(0.08)) ? T(0.12) : T(0.05);
        T dn = 0;
        for (int i = 0; i < N; ++i) dn += delta[i] * delta[i];
        dn = sqrt(dn);
        if (dn > R) for (int i = 0; i < N; ++i) delta[i] *= R / (dn + T(1e-18));
        T best_a_err = T(1e30);
        int best_a = 0;
        for (int a = 0; a < 5; ++a) {
            for (int i = 0; i < N; ++i) trial[i] = fmin(fmax(cfg[i] + alphas[a] * delta[i], T(LO[i])), T(HI[i]));
            T c;
            r::ik_pose_cost_thread<T>(trial, target, &c, 1);
            if (c < best_a_err) { best_a_err = c; best_a = a; }
        }
        for (int i = 0; i < N; ++i) trial[i] = fmin(fmax(cfg[i] + alphas[best_a] * delta[i], T(LO[i])), T(HI[i]));
        if (best_a_err < cost * T(1 - 1e-4)) {
            for (int i = 0; i < N; ++i) cfg[i] = trial[i];
            lam = fmax(lam * T(0.5), T(1e-10));
        } else {
            lam = fmin(lam * T(3), T(1e6));
        }
        if (best_a_err < best_err) {
            best_err = best_a_err;
            for (int i = 0; i < N; ++i) best_cfg[i] = trial[i];
        }
    }
    for (int i = 0; i < N; ++i) out[(size_t)b * N + i] = (float)best_cfg[i];
}

void launch(int n, int S, const float* ds, const float* dt, float* dout)
{
    if (!strcmp(g_mode, "f32")) k_sqp<float, float><<<(n + 31) / 32, 32>>>(ds, dt, dout, n, S);
    else k_sqp<float, double><<<(n + 31) / 32, 32>>>(ds, dt, dout, n, S);  // GRiM's own split
}
"""

# ---- LM-multi: my loop around crocket's ik_geometry (stacked residual + Jacobian of NF frames) ----
# Dual-form damped LS like ik_lm_step: dq = -Jw^T (Jw Jw^T + lam I)^-1 rw, clamp to the limits,
# accept if the weighted cost drops (lam / 3) else reject (lam * 4). Needs: NF, TS = 7 * NF,
# POS_W, ORI_W, LAM0, MAX_ITER, __constant__ LO[N], HI[N].
MULTI_LOOP = r"""
namespace r = crocket::robots::robot;
#include <cstring>
constexpr int N = r::n_q;
constexpr int M = 6 * NF;
constexpr float EPS_POS2 = 1e-4f * 1e-4f, EPS_ORI2 = 1e-3f * 1e-3f;
static const char* g_mode = "thread";

// cost = 0.5 |W r|^2; done = every frame's position and orientation error under the tolerances.
__device__ void eval_cost(const float* rr, float& cost, bool& done)
{
    cost = 0.f;
    done = true;
    for (int f = 0; f < NF; ++f) {
        const float p2 = rr[6 * f] * rr[6 * f] + rr[6 * f + 1] * rr[6 * f + 1] + rr[6 * f + 2] * rr[6 * f + 2];
        const float o2 = rr[6 * f + 3] * rr[6 * f + 3] + rr[6 * f + 4] * rr[6 * f + 4] + rr[6 * f + 5] * rr[6 * f + 5];
        cost += 0.5f * (POS_W * POS_W * p2 + ORI_W * ORI_W * o2);
        done = done && p2 < EPS_POS2 && o2 < EPS_ORI2;
    }
}

// Solves the SPD system A z = y (M x M, row-major) in place by Cholesky; the solution is left in y.
__device__ void chol_solve(float* A, float* y)
{
    for (int j = 0; j < M; ++j) {
        float d = A[j * M + j];
        for (int k = 0; k < j; ++k) d -= A[j * M + k] * A[j * M + k];
        d = sqrtf(fmaxf(d, 1e-20f));
        A[j * M + j] = d;
        for (int i = j + 1; i < M; ++i) {
            float s = A[i * M + j];
            for (int k = 0; k < j; ++k) s -= A[i * M + k] * A[j * M + k];
            A[i * M + j] = s / d;
        }
    }
    for (int i = 0; i < M; ++i) {
        float s = y[i];
        for (int k = 0; k < i; ++k) s -= A[i * M + k] * y[k];
        y[i] = s / A[i * M + i];
    }
    for (int i = M - 1; i >= 0; --i) {
        float s = y[i];
        for (int k = i + 1; k < M; ++k) s -= A[k * M + i] * y[k];
        y[i] = s / A[i * M + i];
    }
}

__global__ void k_multi(const float* __restrict__ seeds, const float* __restrict__ targets,
                        float* __restrict__ out, int n_seeds, int S)
{
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_seeds) return;
    const float* tgt = targets + TS * (b / S);
    float q[N], trial[N], r0[M], rt[M], J[M * N], A[M * M], y[M];
    for (int i = 0; i < N; ++i) q[i] = seeds[(size_t)b * N + i];

    float lam = LAM0, cost, cost_t;
    bool done, done_t;
    r::ik_geometry_thread<float>(q, tgt, nullptr, r0, J, nullptr, nullptr, nullptr, 1);
    eval_cost(r0, cost, done);
    for (int it = 0; it < MAX_ITER && !done; ++it) {
        for (int i = 0; i < M; ++i) {
            const float wi = (i % 6 < 3) ? POS_W : ORI_W;
            for (int j = 0; j <= i; ++j) {
                const float wj = (j % 6 < 3) ? POS_W : ORI_W;
                float s = 0.f;
                for (int k = 0; k < N; ++k) s += J[i * N + k] * J[j * N + k];
                A[i * M + j] = A[j * M + i] = wi * wj * s;
            }
            A[i * M + i] += lam;
            y[i] = wi * r0[i];
        }
        chol_solve(A, y);
        for (int k = 0; k < N; ++k) {
            float dq = 0.f;
            for (int i = 0; i < M; ++i) dq += ((i % 6 < 3) ? POS_W : ORI_W) * J[i * N + k] * y[i];
            trial[k] = fminf(fmaxf(q[k] - dq, LO[k]), HI[k]);
        }
        r::ik_geometry_value_thread<float>(trial, tgt, nullptr, rt, nullptr, nullptr, 1);
        eval_cost(rt, cost_t, done_t);
        if (cost_t < cost) {
            for (int k = 0; k < N; ++k) q[k] = trial[k];
            r::ik_geometry_thread<float>(q, tgt, nullptr, r0, J, nullptr, nullptr, nullptr, 1);
            eval_cost(r0, cost, done);
            lam = fmaxf(lam / 3.f, 1e-7f);
        } else {
            lam = fminf(lam * 4.f, 1e7f);
        }
    }
    for (int i = 0; i < N; ++i) out[(size_t)b * N + i] = q[i];
}

void launch(int n, int S, const float* ds, const float* dt, float* dout)
{
    k_multi<<<(n + 63) / 64, 64>>>(ds, dt, dout, n, S);
}
"""


# ---- LM-coll: lmmulti + collision hinge rows (world obstacles and self pairs) ----------------
# crocket's ik_geometry gives the world centres `cw` of all robot spheres and every joint's world twist
# `tw` ([w, v_o] each); a centre c moves with n . (v_o + w x c) along the unit direction n per unit of
# that joint's column, which is the gradient of any sphere gap. Residual per colliding sphere pair or
# sphere-obstacle pair: h = max(0, margin - gap), cost 0.5 * COLL_W * h^2, Gauss-Newton rows added to the
# primal normal equations (J^T W^2 J + sum COLL_W a a^T + lam I) dq = J^T W^2 r + sum -COLL_W h a.
# Per-problem obstacles ride after the poses in that problem's target block:
# [n_sph, n_box, n_cap, sph (MAX_S x 4), box (MAX_B x 15), cap (MAX_C x 7)]
# (sphere: x y z r; box: centre, R row-major, half extents; capsule: p1, p2, r).
# Needs: NF, TS, N_SPH, N_JOINT, N_PAIR, MAX_S/B/C, POS_W, ORI_W, COLL_W, COLL_MARGIN, DONE_GAP, LAM0, MAX_ITER,
# __constant__ LO/HI[N], RAD[N_SPH], ANC[N_SPH] (bit j: joint j moves the sphere), JQ[N_JOINT], PAIR_I/J[N_PAIR + 1].
COLL_LOOP = r"""
namespace r = crocket::robots::robot;
#include <cstring>
constexpr int N = r::n_q;
constexpr int M = 6 * NF;
constexpr float EPS_POS2 = 1e-4f * 1e-4f, EPS_ORI2 = 1e-3f * 1e-3f;
static const char* g_mode = "thread";

__device__ void eval_cost(const float* rr, float& cost, bool& done)
{
    cost = 0.f;
    done = true;
    for (int f = 0; f < NF; ++f) {
        const float p2 = rr[6 * f] * rr[6 * f] + rr[6 * f + 1] * rr[6 * f + 1] + rr[6 * f + 2] * rr[6 * f + 2];
        const float o2 = rr[6 * f + 3] * rr[6 * f + 3] + rr[6 * f + 4] * rr[6 * f + 4] + rr[6 * f + 5] * rr[6 * f + 5];
        cost += 0.5f * (POS_W * POS_W * p2 + ORI_W * ORI_W * o2);
        done = done && p2 < EPS_POS2 && o2 < EPS_ORI2;
    }
}

// Cholesky solve of the SPD system A z = y (N x N, row-major, lower triangle used); z is left in y.
__device__ void chol_solve(float* A, float* y)
{
    for (int j = 0; j < N; ++j) {
        float d = A[j * N + j];
        for (int k = 0; k < j; ++k) d -= A[j * N + k] * A[j * N + k];
        d = sqrtf(fmaxf(d, 1e-20f));
        A[j * N + j] = d;
        for (int i = j + 1; i < N; ++i) {
            float s = A[i * N + j];
            for (int k = 0; k < j; ++k) s -= A[i * N + k] * A[j * N + k];
            A[i * N + j] = s / d;
        }
    }
    for (int i = 0; i < N; ++i) {
        float s = y[i];
        for (int k = 0; k < i; ++k) s -= A[i * N + k] * y[k];
        y[i] = s / A[i * N + i];
    }
    for (int i = N - 1; i >= 0; --i) {
        float s = y[i];
        for (int k = i + 1; k < N; ++k) s -= A[k * N + i] * y[k];
        y[i] = s / A[i * N + i];
    }
}

// d(centre . n)/d(joint j's coordinate): n . (v_o + w x c) = n . v_o + w . (c x n)
__device__ float motion(const float* tw, int j, const float* c, const float* n)
{
    const float* t = tw + 6 * j;
    return t[3] * n[0] + t[4] * n[1] + t[5] * n[2] + t[0] * (c[1] * n[2] - c[2] * n[1]) +
           t[1] * (c[2] * n[0] - c[0] * n[2]) + t[2] * (c[0] * n[1] - c[1] * n[0]);
}

__device__ void sphere_grad(const float* tw, unsigned mask, const float* c, const float* n, float sgn, float* a)
{
    for (int j = 0; j < N_JOINT; ++j)
        if ((mask >> j) & 1u) a[JQ[j]] += sgn * motion(tw, j, c, n);
}

// One gap d (negative = penetrating) with unit direction n = d(gap)/d(centre i) (and -n for centre j).
// Adds the hinge's cost; with H/g non-null also its Gauss-Newton rows. Tracks the smallest gap.
__device__ float hinge(float d, unsigned mi, const float* ci, unsigned mj, const float* cj, const float* n,
                       const float* tw, float* H, float* g, float& mg)
{
    mg = fminf(mg, d);
    if (d >= COLL_MARGIN) return 0.f;
    const float h = COLL_MARGIN - d;
    if (H) {
        float a[N];
        for (int i = 0; i < N; ++i) a[i] = 0.f;
        if (mi) sphere_grad(tw, mi, ci, n, 1.f, a);
        if (mj) sphere_grad(tw, mj, cj, n, -1.f, a);
        for (int i = 0; i < N; ++i) {
            g[i] -= COLL_W * h * a[i];
            for (int j = 0; j < N; ++j) H[i * N + j] += COLL_W * a[i] * a[j];
        }
    }
    return 0.5f * COLL_W * h * h;
}

__device__ float coll_terms(const float* cw, const float* tw, const float* obs, float* H, float* g, float& mg)
{
    float cost = 0.f;
    mg = 1e9f;
    const int ns = (int)obs[0], nb = (int)obs[1], nc = (int)obs[2];
    const float* sph = obs + 3;
    const float* box = sph + MAX_S * 4;
    const float* cap = box + MAX_B * 15;
    for (int s = 0; s < N_SPH; ++s) {
        const float* c = cw + 3 * s;
        const unsigned m = ANC[s];
        for (int k = 0; k < ns; ++k) {
            const float* o = sph + 4 * k;
            const float v[3] = {c[0] - o[0], c[1] - o[1], c[2] - o[2]};
            const float dist = sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
            const float inv = dist > 1e-9f ? 1.f / dist : 0.f;
            const float n[3] = {dist > 1e-9f ? v[0] * inv : 0.f, dist > 1e-9f ? v[1] * inv : 0.f,
                                dist > 1e-9f ? v[2] * inv : 1.f};
            cost += hinge(dist - RAD[s] - o[3], m, c, 0u, nullptr, n, tw, H, g, mg);
        }
        for (int k = 0; k < nb; ++k) {
            const float* b = box + 15 * k;
            const float dp[3] = {c[0] - b[0], c[1] - b[1], c[2] - b[2]};
            float l[3], o[3], o2 = 0.f;
            for (int a = 0; a < 3; ++a) {
                l[a] = b[3 + a] * dp[0] + b[6 + a] * dp[1] + b[9 + a] * dp[2];   // R^T (c - p)
                o[a] = fmaxf(fabsf(l[a]) - b[12 + a], 0.f);
                o2 += o[a] * o[a];
            }
            float nl[3] = {0.f, 0.f, 0.f}, d;
            if (o2 > 1e-12f) {
                const float dist = sqrtf(o2);
                d = dist - RAD[s];
                for (int a = 0; a < 3; ++a) nl[a] = copysignf(o[a], l[a]) / dist;
            } else {  // centre inside the box: leave through the nearest face
                int ax = 0;
                float best = -1e30f;
                for (int a = 0; a < 3; ++a) {
                    const float e = fabsf(l[a]) - b[12 + a];
                    if (e > best) { best = e; ax = a; }
                }
                d = best - RAD[s];
                nl[ax] = copysignf(1.f, l[ax]);
            }
            float n[3];
            for (int i = 0; i < 3; ++i) n[i] = b[3 + 3 * i] * nl[0] + b[4 + 3 * i] * nl[1] + b[5 + 3 * i] * nl[2];
            cost += hinge(d, m, c, 0u, nullptr, n, tw, H, g, mg);
        }
        for (int k = 0; k < nc; ++k) {
            const float* p = cap + 7 * k;
            const float seg[3] = {p[3] - p[0], p[4] - p[1], p[5] - p[2]};
            const float len2 = seg[0] * seg[0] + seg[1] * seg[1] + seg[2] * seg[2];
            const float t = fminf(fmaxf(((c[0] - p[0]) * seg[0] + (c[1] - p[1]) * seg[1] + (c[2] - p[2]) * seg[2]) /
                                        fmaxf(len2, 1e-12f), 0.f), 1.f);
            const float v[3] = {c[0] - p[0] - t * seg[0], c[1] - p[1] - t * seg[1], c[2] - p[2] - t * seg[2]};
            const float dist = sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
            const float n[3] = {dist > 1e-9f ? v[0] / dist : 0.f, dist > 1e-9f ? v[1] / dist : 0.f,
                                dist > 1e-9f ? v[2] / dist : 1.f};
            cost += hinge(dist - RAD[s] - p[6], m, c, 0u, nullptr, n, tw, H, g, mg);
        }
    }
    for (int p = 0; p < N_PAIR; ++p) {
        const int i = PAIR_I[p], j = PAIR_J[p];
        const float* ci = cw + 3 * i;
        const float* cj = cw + 3 * j;
        const float v[3] = {ci[0] - cj[0], ci[1] - cj[1], ci[2] - cj[2]};
        const float dist = sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
        const float n[3] = {dist > 1e-9f ? v[0] / dist : 0.f, dist > 1e-9f ? v[1] / dist : 0.f,
                            dist > 1e-9f ? v[2] / dist : 1.f};
        cost += hinge(dist - RAD[i] - RAD[j], ANC[i], ci, ANC[j], cj, n, tw, H, g, mg);
    }
    return cost;
}

// Pose + collision cost at q; full also fills the Jacobian and twists (the state we step from).
__device__ void eval_state(const float* qv, const float* tgt, const float* obs, bool full, float* rr, float* Jm,
                           float* cwv, float* twv, float& cost, bool& done)
{
    if (full) r::ik_geometry_thread<float>(qv, tgt, nullptr, rr, Jm, cwv, nullptr, twv, 1);
    else r::ik_geometry_value_thread<float>(qv, tgt, nullptr, rr, cwv, nullptr, 1);
    float pose_cost, mg;
    bool pose_done;
    eval_cost(rr, pose_cost, pose_done);
    const float cc = coll_terms(cwv, nullptr, obs, nullptr, nullptr, mg);
    cost = pose_cost + cc;
    done = pose_done && mg >= DONE_GAP;
}

__global__ void k_coll(const float* __restrict__ seeds, const float* __restrict__ targets,
                       float* __restrict__ out, int n_seeds, int S)
{
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_seeds) return;
    const float* tgt = targets + TS * (b / S);
    const float* obs = tgt + 7 * NF;
    float q[N], trial[N], r0[M], rt[M], J[M * N], A[N * N], y[N], cw[3 * N_SPH], cwt[3 * N_SPH], tw[6 * N_JOINT];
    for (int i = 0; i < N; ++i) q[i] = seeds[(size_t)b * N + i];

    float lam = LAM0, cost, cost_t, mg;
    bool done, done_t;
    eval_state(q, tgt, obs, true, r0, J, cw, tw, cost, done);
    for (int it = 0; it < MAX_ITER && !done; ++it) {
        for (int a = 0; a < N; ++a) {
            y[a] = 0.f;
            for (int c = 0; c <= a; ++c) {
                float s = 0.f;
                for (int i = 0; i < M; ++i) {
                    const float w = (i % 6 < 3) ? POS_W : ORI_W;
                    s += w * w * J[i * N + a] * J[i * N + c];
                }
                A[a * N + c] = A[c * N + a] = s;
            }
            for (int i = 0; i < M; ++i) {
                const float w = (i % 6 < 3) ? POS_W : ORI_W;
                y[a] += w * w * J[i * N + a] * r0[i];
            }
        }
        coll_terms(cw, tw, obs, A, y, mg);                  // adds the hinge rows to A and y
        for (int a = 0; a < N; ++a) A[a * N + a] += lam;
        chol_solve(A, y);
        for (int k = 0; k < N; ++k) trial[k] = fminf(fmaxf(q[k] - y[k], LO[k]), HI[k]);
        eval_state(trial, tgt, obs, false, rt, nullptr, cwt, nullptr, cost_t, done_t);
        if (cost_t < cost) {
            for (int k = 0; k < N; ++k) q[k] = trial[k];
            eval_state(q, tgt, obs, true, r0, J, cw, tw, cost, done);
            lam = fmaxf(lam / 3.f, 1e-7f);
        } else {
            lam = fminf(lam * 4.f, 1e7f);
        }
    }
    for (int i = 0; i < N; ++i) out[(size_t)b * N + i] = q[i];
}

void launch(int n, int S, const float* ds, const float* dt, float* dout)
{
    k_coll<<<(n + 63) / 64, 64>>>(ds, dt, dout, n, S);
}
"""


def obstacle_block(t) -> tuple:
    """(P, 3 + 4 MS + 15 MB + 7 MC) float32 per-problem obstacle blocks and (MS, MB, MC); counts first."""
    P = len(t["target"])
    if "n_sph" not in t.files:                                   # no world: zero counts, 1-slot padding
        return np.zeros((P, 3 + 4 + 15 + 7), np.float32), (1, 1, 1)
    sph, box, cap = t["sph"], t["box"], t["cap"]
    head = np.stack([t["n_sph"], t["n_box"], t["n_cap"]], axis=1)
    block = np.concatenate([head, sph.reshape(P, -1), box.reshape(P, -1), cap.reshape(P, -1)], axis=1)
    return block.astype(np.float32), (sph.shape[1], box.shape[1], cap.shape[1])


def chain_urdf(chain: str) -> Path:
    """Copy of the URDF where every movable joint outside the chain is `fixed` (value 0)."""
    keep = set(problem.CHAINS[chain].joints)
    robot = problem.CHAINS[chain].robot
    tree = ET.parse(problem.common.ROBOTS[robot].urdf_path)
    for joint in tree.getroot().findall("joint"):
        if joint.get("type") in ("revolute", "prismatic", "continuous") and joint.get("name") not in keep:
            joint.set("type", "fixed")
    BUILD.mkdir(parents=True, exist_ok=True)
    path = BUILD / f"{robot}_{chain}_chain.urdf"
    tree.write(path)
    return path


def compile_cu(src: str, stem: str) -> Path:
    cu, exe = BUILD / f"{stem}.cu", BUILD / stem
    if not exe.exists() or not cu.exists() or cu.read_text() != src:
        cu.write_text(src)
        subprocess.check_call(["nvcc", "-O3", "-std=c++17", ARCH, str(cu), "-o", str(exe)])
    return exe


def build_lm(robot, tools, max_iter: int, chain: str) -> Path:
    opts = {"frame": tools[0], "pos_weight": POS_W, "ori_weight": ORI_W}
    src = crocket.generate(robot, {"ik_lm_step": opts},
                           variants=[cuda.Variant("thread"), cuda.Variant("warp", 32)], name="robot")
    src += (f"\n#define LAM0 5e-3f\n#define MAX_ITER {max_iter}\n#define WARPS {WARPS}\n#define TS 7\n"
            + LM_LOOP + MAIN)
    return compile_cu(src, f"crocket_lm_{chain}_{max_iter}")


def build_sqp(robot, tools, max_iter: int, chain: str) -> Path:
    frame = tools[0]
    opts = {"frame": frame, "pos_weight": POS_W, "ori_weight": ORI_W}
    # Chain columns: where a traced step's delta can be nonzero (zero elsewhere by construction).
    probe = crocket.trace(robot, "ik_ls_step", frame=frame).evaluate(
        q=np.random.default_rng(0).uniform(-1, 1, (4, robot.n_q)),
        target=np.tile([1.0, 0, 0, 0, 0.3, 0.1, 0.5], (4, 1)), lam=np.full((4, 1), 0.05))["delta"]
    cols = np.flatnonzero(np.abs(probe).sum(0))
    nc = {"n": len(cols)}
    src = crocket.generate(robot, {"ik_ls_system": opts, "ik_pose_cost": opts, "ik_sqp_system": nc,
                                   "ik_masked_solve": nc},
                           variants=[cuda.Variant("thread")], name="robot")
    owner = {j.q_index: j for j in robot.joints if not j.mimic}
    lim = [owner[c].limits for c in range(len(owner))]
    arr = lambda xs: "{" + ", ".join(f"{x:.9g}" for x in xs) + "}"  # noqa: E731
    src += (f"\n#define LAM0 5e-3\n#define MAX_ITER {max_iter}\n#define EPS_POS 1e-4\n#define EPS_ORI 1e-3\n"
            f"#define N_INNER {N_INNER}\n#define NC {len(cols)}\n#define TS 7\n"
            f"__constant__ int COLS[] = {{{', '.join(map(str, cols))}}};\n"
            f"__constant__ float LO[] = {arr([l for l, _ in lim])};\n"
            f"__constant__ float HI[] = {arr([h for _, h in lim])};\n"
            + SQP_LOOP + MAIN)
    return compile_cu(src, f"crocket_sqp_{chain}_{max_iter}")


def build_multi(robot, tools, max_iter: int, chain: str, lo_c: np.ndarray, hi_c: np.ndarray) -> Path:
    opts = {"frames": tuple(tools), "cols": tuple(range(robot.n_q))}
    src = crocket.generate(robot, {"ik_geometry": opts, "ik_geometry_value": opts},
                           variants=[cuda.Variant("thread")], name="robot")
    arr = lambda xs: "{" + ", ".join(f"{x:.9g}" for x in xs) + "}"  # noqa: E731
    src += (f"\n#define NF {len(tools)}\n#define TS (7 * NF)\n#define LAM0 5e-3f\n#define MAX_ITER {max_iter}\n"
            f"#define POS_W {POS_W}f\n#define ORI_W {ORI_W}f\n"
            f"__constant__ float LO[] = {arr(lo_c)};\n__constant__ float HI[] = {arr(hi_c)};\n"
            + MULTI_LOOP + MAIN)
    return compile_cu(src, f"crocket_multi_{chain}_{max_iter}")


def build_coll(robot, tools, max_iter: int, chain: str, lo_c: np.ndarray, hi_c: np.ndarray, maxes: tuple) -> Path:
    spec = problem.common.ROBOTS[problem.CHAINS[chain].robot]
    spheres = problem.urdf_spheres(spec)
    if len(robot.joints) > 32:
        raise SystemExit("the collision loop keeps one 32-bit joint mask per sphere")
    masks = []                                          # bit j: crocket joint j moves sphere i
    for link, _, _ in spheres:
        j, m = robot.frame(link).parent, 0
        while j >= 0:
            m |= 1 << j
            j = robot.joints[j].parent
        masks.append(m)
    disabled = problem.srdf_disabled(spec)
    pairs = [(i, j) for i in range(len(spheres)) for j in range(i + 1, len(spheres))
             if spheres[i][0] != spheres[j][0] and frozenset((spheres[i][0], spheres[j][0])) not in disabled
             and masks[i] != masks[j]]                  # equal masks = rigidly attached: the gap never changes
    if len(pairs) > 8000:
        raise SystemExit(f"{len(pairs)} self pairs do not fit the constant-memory tables")
    print(f"collision loop: {len(spheres)} spheres, {len(pairs)} self pairs, joint order "
          f"{[j.name for j in robot.joints]}")
    opts = {"frames": tuple(tools), "cols": tuple(range(robot.n_q)),
            "wspheres": tuple((link, c) for link, c, _ in spheres)}
    src = crocket.generate(robot, {"ik_geometry": opts, "ik_geometry_value": opts},
                           variants=[cuda.Variant("thread")], name="robot")
    arr = lambda xs, fmt="{:.9g}": "{" + ", ".join(fmt.format(x) for x in xs) + "}"  # noqa: E731
    ms, mb, mc = maxes
    src += (f"\n#define NF {len(tools)}\n#define LAM0 5e-3f\n#define MAX_ITER {max_iter}\n"
            f"#define POS_W {POS_W}f\n#define ORI_W {ORI_W}f\n#define COLL_W {COLL_W}f\n"
            f"#define COLL_MARGIN {COLL_MARGIN}f\n#define DONE_GAP {DONE_GAP}f\n"
            f"#define N_SPH {len(spheres)}\n#define N_JOINT {len(robot.joints)}\n#define N_PAIR {len(pairs)}\n"
            f"#define MAX_S {ms}\n#define MAX_B {mb}\n#define MAX_C {mc}\n"
            "#define TS (7 * NF + 3 + MAX_S * 4 + MAX_B * 15 + MAX_C * 7)\n"
            f"__constant__ float LO[] = {arr(lo_c)};\n__constant__ float HI[] = {arr(hi_c)};\n"
            f"__constant__ float RAD[] = {arr([s[2] for s in spheres])};\n"
            f"__constant__ unsigned ANC[] = {arr(masks, '{}u')};\n"
            f"__constant__ int JQ[] = {arr([j.q_index for j in robot.joints], '{}')};\n"
            f"__constant__ unsigned short PAIR_I[] = {arr([p[0] for p in pairs] or [0], '{}')};\n"
            f"__constant__ unsigned short PAIR_J[] = {arr([p[1] for p in pairs] or [0], '{}')};\n"
            + COLL_LOOP + MAIN)
    return compile_cu(src, f"crocket_coll_{chain}_{max_iter}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chain", choices=sorted(problem.CHAINS), default="arm")
    parser.add_argument("--method", choices=("lm", "sqp", "lmmulti", "lmcoll"), default="lm")
    parser.add_argument("--seeds", default="8,32,128")
    parser.add_argument("--max-iter", type=int, default=60)
    parser.add_argument("--set", default="", choices=("", "coll"), dest="tset")
    parser.add_argument("--warm", action="store_true", help="mbm: seed 0 of each target = the problem's start config")
    args = parser.parse_args()

    t = np.load(problem.target_path(args.chain, args.tset))
    joints = [str(j) for j in t["joints"]]
    tools = [str(x) for x in t["tools"]]
    lo, hi = t["lo"], t["hi"]
    if args.method not in ("lmmulti", "lmcoll") and len(tools) > 1:
        raise SystemExit(f"--method {args.method} handles one tool; chain {args.chain} has {len(tools)}. "
                         "Use --method lmmulti or lmcoll.")
    # per tool [qw qx qy qz x y z] (crocket's order), flattened per target
    tg = t["target"]                                              # (P, tools, 7) as [x y z qw qx qy qz]
    target = np.concatenate([tg[..., 3:], tg[..., :3]], axis=-1).reshape(len(tg), -1).astype(np.float32)
    P = len(target)
    maxes = (1, 1, 1)
    if args.method == "lmcoll":                       # obstacles ride after each problem's poses
        block, maxes = obstacle_block(t)
        target = np.ascontiguousarray(np.concatenate([target, block], axis=1), np.float32)

    robot =crocket.Robot(str(chain_urdf(args.chain)), name="robot")
    names = list(robot.joint_names)
    assert sorted(names) == sorted(joints), (names, joints)
    to_crocket = [joints.index(n) for n in names]       # crocket column k <- our column to_crocket[k]
    to_ours = [names.index(j) for j in joints]          # our column k <- crocket column to_ours[k]
    if args.method == "lm":
        exe, modes = build_lm(robot, tools, args.max_iter, args.chain), ("thread", "warp")
    elif args.method == "sqp":
        exe, modes = build_sqp(robot, tools, args.max_iter, args.chain), ("mixed", "f32")
    elif args.method == "lmcoll":
        exe = build_coll(robot, tools, args.max_iter, args.chain, lo[to_crocket], hi[to_crocket], maxes)
        modes = ("thread",)
    else:
        exe = build_multi(robot, tools, args.max_iter, args.chain, lo[to_crocket], hi[to_crocket])
        modes = ("thread",)

    out_dir = problem.REPO / "ik_benchmark" / "results"
    out_dir.mkdir(exist_ok=True)
    rng = np.random.default_rng(1234)
    for S in [int(s) for s in args.seeds.split(",")]:
        seeds = rng.uniform(lo, hi, (P, S, len(joints)))
        if args.warm:
            seeds[:, 0] = t["start"]          # seed 0 = the problem's start configuration
        for mode in modes:
            with tempfile.TemporaryDirectory() as d:
                sp, tp, op, lp = (Path(d) / n for n in ("seeds.f32", "targets.f32", "out.f32", "lat.f32"))
                np.ascontiguousarray(seeds[..., to_crocket], np.float32).tofile(sp)
                target.tofile(tp)
                batch_ms = float(subprocess.check_output(
                    [str(exe), mode, str(P), str(S), str(sp), str(tp), str(op), str(lp)], text=True))
                q_all = np.fromfile(op, np.float32).reshape(P, S, len(joints))[..., to_ours]
                latency_ms = np.fromfile(lp, np.float32)
            name = f"crocket_{args.method}_{mode}_{S}{'_warm' if args.warm else ''}"
            fname = f"{args.chain}_{name}{'_' + args.tset if args.tset else ''}"   # chains/sets don't overwrite each other
            np.savez(out_dir / f"{fname}.npz", q_all=q_all.astype(np.float64), seconds=latency_ms / 1e3,
                     batch_seconds_per_problem=batch_ms / 1e3 / P, chain=args.chain, tset=args.tset,
                     name=name)
            print(f"{name}: latency median {np.median(latency_ms):.3f} ms/target, "
                  f"batch {batch_ms / P * 1e3:.2f} us/target")


if __name__ == "__main__":
    main()
