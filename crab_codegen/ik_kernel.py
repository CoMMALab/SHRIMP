"""Fused CUDA IK kernel from CRAB's own symbolic FK: one thread per (target, seed) runs a whole
Levenberg-Marquardt solve, calling SymForce-generated device functions for the pose and the analytic
Jacobian (the same C++ crab_codegen.codegen emits for the CPU JIT, with `__host__ __device__` added and
compiled by nvcc with Eigen). Counterpart of crocket_ik.py's `lmmulti`: same algorithm and weights
(pose residual [p - p*, log(R R*^T)], damped least squares, lam /3 on accept, *4 on reject, joint-limit
clamp, stop at 1e-4 m / 1e-3 rad), solved in dual form (an M x M system, M = 6 tools), as crocket's.

build_ik_source() returns the .cu text up to `launch()`; the host program (ik_benchmark/cuda_host.py)
is appended by the caller. float32. Compile with: nvcc -O3 -std=c++17 --expt-relaxed-constexpr
-I<eigen3 include dir>.

Layout of one problem's target block (TS = 7 * tools floats): per tool [qw qx qy qz x y z].
Seeds / outputs are in the chain's joint order (our order); QI maps them to CRAB's q index.
"""

import numpy as np
import symforce.symbolic as sf

from crab_codegen.codegen import build_and_generate_function
from crab_kinematics.derived import q_to_ee_jacobian_analytic, q_to_ee_pose

POS_W, ORI_W = 50.0, 10.0


def q_to_pose_and_jacobian(robot, q):
    """[pose (7); analytic Jacobian (6 nq, row-major rows v; w)] as one column, so common
    subexpressions are shared between the two."""
    return sf.Matrix(q_to_ee_pose(robot, q).to_flat_list() + q_to_ee_jacobian_analytic(robot, q).to_flat_list())


def device_source(src: str) -> str:
    """Mark the generated function(s) callable from device code."""
    if "template <typename Scalar>" not in src:
        raise ValueError("unexpected SymForce output: no `template <typename Scalar>` found")
    return src.replace("template <typename Scalar>", "template <typename Scalar>\n__host__ __device__")


LOOP = r"""
constexpr float EPS_POS2 = 1e-4f * 1e-4f, EPS_ORI2 = 1e-3f * 1e-3f;
static const char* g_mode = "thread";

__device__ void eval_pj(const float* q, float* pose, float* J)
{
    Eigen::Matrix<float, NQ, 1> qf = Eigen::Matrix<float, NQ, 1>::Zero();
    for (int i = 0; i < N; ++i) qf[QI[i]] = q[i];
@PJ@
}

__device__ void eval_p(const float* q, float* pose)
{
    Eigen::Matrix<float, NQ, 1> qf = Eigen::Matrix<float, NQ, 1>::Zero();
    for (int i = 0; i < N; ++i) qf[QI[i]] = q[i];
@P@
}

// Weighted cost 0.5 |W r|^2 and "done" (every tool within tolerance) of the pose residual
// [p - p*, log(R R*^T)]; pose is [qx qy qz qw x y z] per tool, tgt is [qw qx qy qz x y z].
__device__ void residual(const float* pose, const float* tgt, float* rr, float& cost, bool& done)
{
    cost = 0.f;
    done = true;
    for (int f = 0; f < NF; ++f) {
        const float* p = pose + 7 * f;
        const float* t = tgt + 7 * f;
        rr[6 * f] = p[4] - t[4];
        rr[6 * f + 1] = p[5] - t[5];
        rr[6 * f + 2] = p[6] - t[6];
        const float ax = p[0], ay = p[1], az = p[2], aw = p[3];     // q_cur * conj(q_target)
        const float bx = -t[1], by = -t[2], bz = -t[3], bw = t[0];
        float ex = aw * bx + ax * bw + ay * bz - az * by;
        float ey = aw * by - ax * bz + ay * bw + az * bx;
        float ez = aw * bz + ax * by - ay * bx + az * bw;
        float ew = aw * bw - ax * bx - ay * by - az * bz;
        if (ew < 0.f) { ex = -ex; ey = -ey; ez = -ez; ew = -ew; }
        const float s = sqrtf(ex * ex + ey * ey + ez * ez);
        const float scale = s > 1e-7f ? 2.f * atan2f(s, ew) / s : 2.f / fmaxf(ew, 1e-7f);
        rr[6 * f + 3] = ex * scale;
        rr[6 * f + 4] = ey * scale;
        rr[6 * f + 5] = ez * scale;
        const float p2 = rr[6 * f] * rr[6 * f] + rr[6 * f + 1] * rr[6 * f + 1] + rr[6 * f + 2] * rr[6 * f + 2];
        const float o2 = rr[6 * f + 3] * rr[6 * f + 3] + rr[6 * f + 4] * rr[6 * f + 4] + rr[6 * f + 5] * rr[6 * f + 5];
        cost += 0.5f * (POS_W * POS_W * p2 + ORI_W * ORI_W * o2);
        done = done && p2 < EPS_POS2 && o2 < EPS_ORI2;
    }
}

// Cholesky solve of the SPD system A z = y (K x K, row-major, lower triangle used); z is left in y.
template <int K>
__device__ void chol_solve(float* A, float* y)
{
    for (int j = 0; j < K; ++j) {
        float d = A[j * K + j];
        for (int k = 0; k < j; ++k) d -= A[j * K + k] * A[j * K + k];
        d = sqrtf(fmaxf(d, 1e-20f));
        A[j * K + j] = d;
        for (int i = j + 1; i < K; ++i) {
            float s = A[i * K + j];
            for (int k = 0; k < j; ++k) s -= A[i * K + k] * A[j * K + k];
            A[i * K + j] = s / d;
        }
    }
    for (int i = 0; i < K; ++i) {
        float s = y[i];
        for (int k = 0; k < i; ++k) s -= A[i * K + k] * y[k];
        y[i] = s / A[i * K + i];
    }
    for (int i = K - 1; i >= 0; --i) {
        float s = y[i];
        for (int k = i + 1; k < K; ++k) s -= A[k * K + i] * y[k];
        y[i] = s / A[i * K + i];
    }
}

__global__ void k_lm(const float* __restrict__ seeds, const float* __restrict__ targets,
                     float* __restrict__ out, int n_seeds, int S)
{
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_seeds) return;
    const float* tgt = targets + TS * (b / S);
    float q[N], trial[N], pose[7 * NF], pose_t[7 * NF], r0[M], rt[M], J[M * N], A[M * M], y[M];
    for (int i = 0; i < N; ++i) q[i] = seeds[(size_t)b * N + i];

    float lam = LAM0, cost, cost_t;
    bool done, done_t;
    eval_pj(q, pose, J);
    residual(pose, tgt, r0, cost, done);
    for (int it = 0; it < MAX_ITER && !done; ++it) {
        // dual form (M < N): (W J J^T W + lam I) z = W r, dq = J^T W z
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
        chol_solve<M>(A, y);
        for (int k = 0; k < N; ++k) {
            float dq = 0.f;
            for (int i = 0; i < M; ++i) dq += ((i % 6 < 3) ? POS_W : ORI_W) * J[i * N + k] * y[i];
            trial[k] = fminf(fmaxf(q[k] - dq, LO[k]), HI[k]);
        }
        eval_p(trial, pose_t);
        residual(pose_t, tgt, rt, cost_t, done_t);
        if (cost_t < cost) {
            for (int k = 0; k < N; ++k) q[k] = trial[k];
            eval_pj(q, pose, J);
            residual(pose, tgt, r0, cost, done);
            lam = fmaxf(lam / 3.f, 1e-7f);
        } else {
            lam = fminf(lam * 4.f, 1e7f);
        }
    }
    for (int i = 0; i < N; ++i) out[(size_t)b * N + i] = q[i];
}

// FK-only benchmark (mode "fk"): MAX_ITER x (pose + Jacobian, then pose) per thread and nothing else.
// q is perturbed by the running sum so the compiler can't hoist the evaluations out of the loop.
__global__ void k_fk(const float* __restrict__ seeds, const float* __restrict__ targets,
                     float* __restrict__ out, int n_seeds, int S)
{
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_seeds) return;
    float q[N], pose[7 * NF], pose_t[7 * NF], J[M * N];
    for (int i = 0; i < N; ++i) q[i] = seeds[(size_t)b * N + i];
    float acc = 0.f;
    for (int it = 0; it < MAX_ITER; ++it) {
        eval_pj(q, pose, J);
        eval_p(q, pose_t);
        for (int i = 0; i < 7 * NF; ++i) acc += pose[i] + pose_t[i];     // consume EVERY output: no dead code
        for (int i = 0; i < M * N; ++i) acc += J[i];
        q[it % N] += 1e-6f * acc;
    }
    for (int i = 0; i < N; ++i) out[(size_t)b * N + i] = q[i];
}

void launch(int n, int S, const float* ds, const float* dt, float* dout)
{
    if (!strcmp(g_mode, "fk")) k_fk<<<(n + 63) / 64, 64>>>(ds, dt, dout, n, S);
    else k_lm<<<(n + 63) / 64, 64>>>(ds, dt, dout, n, S);
}
"""


def build_ik_source(robots: list, q_index: list, lo: np.ndarray, hi: np.ndarray, max_iter: int) -> str:
    """robots: one finalized Robot per tool (frames to reach, in target order); q_index[i]: CRAB q index
    of the chain's i-th joint; lo/hi: the chain's limits in the chain's order."""
    nq, dof, nf = robots[0].nq, len(q_index), len(robots)
    pieces, pj_blocks, p_blocks = ["#include <Eigen/Dense>\n#include <cmath>\n#include <cstring>\n"], [], []
    for i, rb in enumerate(robots):
        pj = build_and_generate_function(rb, q_to_pose_and_jacobian, f"ik_pj_{i}")
        p = build_and_generate_function(rb, q_to_ee_pose, f"ik_p_{i}")
        pieces += [device_source(pj["source"]), device_source(p["source"])]
        pj_blocks.append(
            f"    {{\n        const Eigen::Matrix<float, {pj['n_out']}, 1> o = sym::{pj['func_name']}<float>(qf);\n"
            f"        for (int k = 0; k < 7; ++k) pose[{7 * i} + k] = o[k];\n"
            f"        for (int row = 0; row < 6; ++row)\n"
            f"            for (int c = 0; c < N; ++c) J[({6 * i} + row) * N + c] = o[7 + row * NQ + QI[c]];\n    }}")
        p_blocks.append(
            f"    {{\n        const Eigen::Matrix<float, 7, 1> o = sym::{p['func_name']}<float>(qf);\n"
            f"        for (int k = 0; k < 7; ++k) pose[{7 * i} + k] = o[k];\n    }}")
    arr = lambda xs, fmt="{:.9g}": "{" + ", ".join(fmt.format(x) for x in xs) + "}"  # noqa: E731
    consts = (f"\nconstexpr int N = {dof};\nconstexpr int NQ = {nq};\nconstexpr int NF = {nf};\n"
              f"constexpr int M = 6 * NF;\n#define TS (7 * NF)\n#define LAM0 5e-3f\n#define MAX_ITER {max_iter}\n"
              f"#define POS_W {POS_W}f\n#define ORI_W {ORI_W}f\n"
              f"__constant__ int QI[] = {arr(q_index, '{}')};\n"
              f"__constant__ float LO[] = {arr(lo)};\n__constant__ float HI[] = {arr(hi)};\n")
    loop = LOOP.replace("@PJ@", "\n".join(pj_blocks)).replace("@P@", "\n".join(p_blocks))
    return "\n".join(pieces) + consts + loop
