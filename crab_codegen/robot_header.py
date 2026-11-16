"""CRAB's robot header for GPU planners (CUDA, one translation unit, nvcc + Eigen).

build_robot_header() emits the device-side description of one robot that a planner library (SHRIMP) includes.
Contract (what a planner may rely on; keep it stable, improve the inside freely, e.g. a per-link early-exit
chain-form generator behind the same `fkcc`):

    constexpr int N, NQ, NS, NPAIR   chain DOF / CRAB q size / collision spheres / self-collision pairs
    __constant__ int QI[N]           chain joint i -> CRAB q index
    __constant__ float LO[N], HI[N]  joint limits in the chain's order
    struct World                     one problem's obstacles, read-only views into device memory:
                                       sph[ns][4]  x y z r
                                       box[nb][15] center(3), R row-major(9) (world-from-box), half extents(3)
                                       cap[nc][7]  p1(3), p2(3), r      (segment p1-p2 swept by r)
    bool fkcc(const float* q, const World& w)
                                     true = q (chain order, radians) is collision-free: self-collision
                                     (SRDF-disabled and rigidly attached link pairs excluded) and obstacles

Everything is float32; compile with: nvcc -O3 -std=c++17 --expt-relaxed-constexpr -I<eigen3>.
"""

import numpy as np

from crab_codegen.codegen import build_and_generate_function
from crab_codegen.ik_kernel import device_source
from crab_kinematics.derived import forward_kinematics_spheres

WORLD = r"""
struct World {
    const float* sph; int ns;
    const float* box; int nb;
    const float* cap; int nc;
};

__device__ __forceinline__ bool env_hit(float x, float y, float z, float r, const World& w)
{
    for (int i = 0; i < w.ns; ++i) {
        const float* s = w.sph + 4 * i;
        const float dx = x - s[0], dy = y - s[1], dz = z - s[2], rr = r + s[3];
        if (dx * dx + dy * dy + dz * dz < rr * rr) return true;
    }
    for (int i = 0; i < w.nb; ++i) {
        const float* b = w.box + 15 * i;
        const float dx = x - b[0], dy = y - b[1], dz = z - b[2];
        const float lx = dx * b[3] + dy * b[6] + dz * b[9];             // R^T (c - p), R row-major
        const float ly = dx * b[4] + dy * b[7] + dz * b[10];
        const float lz = dx * b[5] + dy * b[8] + dz * b[11];
        const float ox = fmaxf(fabsf(lx) - b[12], 0.f), oy = fmaxf(fabsf(ly) - b[13], 0.f),
                    oz = fmaxf(fabsf(lz) - b[14], 0.f);
        if (ox * ox + oy * oy + oz * oz < r * r) return true;
    }
    for (int i = 0; i < w.nc; ++i) {
        const float* c = w.cap + 7 * i;
        const float sx = c[3] - c[0], sy = c[4] - c[1], sz = c[5] - c[2];
        const float len2 = fmaxf(sx * sx + sy * sy + sz * sz, 1e-12f);
        const float s = fminf(fmaxf(((x - c[0]) * sx + (y - c[1]) * sy + (z - c[2]) * sz) / len2, 0.f), 1.f);
        const float dx = x - (c[0] + s * sx), dy = y - (c[1] + s * sy), dz = z - (c[2] + s * sz);
        const float rr = r + c[6];
        if (dx * dx + dy * dy + dz * dz < rr * rr) return true;
    }
    return false;
}

// true = q is collision-free (self + obstacles). q is in the chain's joint order.
__device__ __noinline__ bool fkcc(const float* q, const World& w)
{
    Eigen::Matrix<float, NQ, 1> qf = Eigen::Matrix<float, NQ, 1>::Zero();
    for (int i = 0; i < N; ++i) qf[QI[i]] = q[i];
    const Eigen::Matrix<float, NS * 4, 1> o = sym::@FK@<float>(qf);
    for (int s = 0; s < NS; ++s)
        if (env_hit(o[4 * s], o[4 * s + 1], o[4 * s + 2], o[4 * s + 3], w)) return false;
    for (int k = 0; k < NPAIR; ++k) {
        const int i = PI[k], j = PJ[k];
        const float dx = o[4 * i] - o[4 * j], dy = o[4 * i + 1] - o[4 * j + 1], dz = o[4 * i + 2] - o[4 * j + 2];
        const float rs = o[4 * i + 3] + o[4 * j + 3];
        if (dx * dx + dy * dy + dz * dz < rs * rs) return false;
    }
    return true;
}
"""


def build_robot_header(robot, q_index, lo: np.ndarray, hi: np.ndarray, pair_i, pair_j) -> str:
    """robot: finalized CRAB Robot; q_index[i]: CRAB q index of the chain's i-th joint; lo/hi: chain limits;
    pair_i/pair_j: self-collision sphere pairs (indices in crab_codegen.sphere_order's order)."""
    if len(pair_i) > 14000:
        raise ValueError(f"{len(pair_i)} self pairs do not fit in constant memory; move PI/PJ to global memory")
    g = build_and_generate_function(robot, forward_kinematics_spheres, "crab_spheres")
    arr = lambda xs, fmt="{:.9g}": "{" + ", ".join(fmt.format(x) for x in xs) + "}"  # noqa: E731
    npair = max(len(pair_i), 1)
    consts = (f"\nconstexpr int N = {len(q_index)};\nconstexpr int NQ = {robot.nq};\n"
              f"constexpr int NS = {g['n_out'] // 4};\nconstexpr int NPAIR = {len(pair_i)};\n"
              f"__constant__ int QI[] = {arr(q_index, '{}')};\n"
              f"__constant__ float LO[] = {arr(lo)};\n__constant__ float HI[] = {arr(hi)};\n"
              f"__constant__ unsigned short PI[{npair}] = {arr(list(pair_i) or [0], '{}')};\n"
              f"__constant__ unsigned short PJ[{npair}] = {arr(list(pair_j) or [0], '{}')};\n")
    head = "#include <Eigen/Dense>\n#include <cfloat>\n#include <cmath>\n#include <cstring>\n"
    return "\n".join([head, device_source(g["source"]), consts, WORLD.replace("@FK@", g["func_name"])])
