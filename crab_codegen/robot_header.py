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
    bool fkcc(const float* q, const World& w, unsigned lane_mask)
                                     true = q (chain order, radians) is collision-free: self-collision
                                     (SRDF-disabled and rigidly attached link pairs excluded) and obstacles.
                                     The lanes of lane_mask call it together; any lane's hit makes all return
                                     false (warp-wide early exit, see fkcc's comment)

Everything is float32; compile with: nvcc -O3 -std=c++17 --expt-relaxed-constexpr -I<eigen3>.
"""

import numpy as np
import symforce.symbolic as sf

from crab_codegen.codegen import build_and_generate_function
from crab_codegen.ik_kernel import device_source
from crab_codegen.sphere_order import link_bounding_spheres
from crab_kinematics.derived import forward_kinematics_spheres
from crab_kinematics.fk import forward_kinematics

WORLD = r"""
struct World {
    const float* sph; int ns;
    const float* box; int nb;
    const float* cap; int nc;
};

@FN@ bool env_hit(float x, float y, float z, float r, const World& w)
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
"""

GPU_FKCC = r"""
constexpr int ENV_VOTE = 5;     // spheres between warp votes in the obstacle loop (~100+ instructions)
constexpr int PAIR_VOTE = 16;    // self pairs between warp votes (~160 instructions)

// true = q is collision-free (self + obstacles). q is in the chain's joint order.
// Called by the lanes of `lane_mask` TOGETHER (each with its own q), e.g. one edge's waypoints: after every
// ENV_VOTE spheres / PAIR_VOTE pairs the lanes vote (__any_sync), and as soon as any lane has hit, ALL lanes
// return false (an edge with one bad waypoint is bad). Spheres are tested from the LAST (distal) to the
// first; pairs are stored most-distal first. A lane that has hit stops testing but keeps voting.
// With a single-lane mask it is a plain per-config check.
__device__ __noinline__ bool fkcc(const float* q, const World& w, unsigned lane_mask)
{
    Eigen::Matrix<float, NQ, 1> qf = Eigen::Matrix<float, NQ, 1>::Zero();
    for (int i = 0; i < N; ++i) qf[QI[i]] = q[i];
    const Eigen::Matrix<float, NS * 4, 1> o = sym::@FK@<float>(qf);
    bool hit = false;
    for (int s = NS - 1; s >= 0; --s) {
        if (!hit) hit = env_hit(o[4 * s], o[4 * s + 1], o[4 * s + 2], o[4 * s + 3], w);
        if ((NS - 1 - s) % ENV_VOTE == ENV_VOTE - 1 && __any_sync(lane_mask, hit)) return false;
    }
    if (__any_sync(lane_mask, hit)) return false;
    for (int k = 0; k < NPAIR; ++k) {
        if (!hit) {
            const int i = PI[k], j = PJ[k];
            const float dx = o[4 * i] - o[4 * j], dy = o[4 * i + 1] - o[4 * j + 1], dz = o[4 * i + 2] - o[4 * j + 2];
            const float rs = o[4 * i + 3] + o[4 * j + 3];
            hit = dx * dx + dy * dy + dz * dz < rs * rs;
        }
        if (k % PAIR_VOTE == PAIR_VOTE - 1 && __any_sync(lane_mask, hit)) return false;
    }
    return !hit;
}
"""


CPU_FKCC = r"""
// true = q (chain order, radians) is collision-free (self + obstacles). Scalar, one config: returns at the
// first hit; spheres tested from the LAST (distal) to the first, pairs most-distal first.
// Broad phase after CRICKET/VAMP: each link has one bounding sphere (BR[g] radius, world centre computed by the FK
// after the spheres, at o[NS * 4 + 3 g]). A link's spheres are tested only if its bounding sphere hits the
// environment; the sphere pairs of an allowed link pair only if the two bounding spheres overlap.
#ifdef SHRIMP_PROF      // -DSHRIMP_PROF: cycles per phase (FK, obstacles, self pairs), summed per thread
#include <x86intrin.h>
static thread_local unsigned long long g_cyc[3] = {0, 0, 0};
#define TSC() __rdtsc()
#else
#define TSC() 0ull
#endif
static inline bool fkcc(const float* q, const World& w)
{
    const unsigned long long t0 = TSC();
    Eigen::Matrix<float, NQ, 1> qf = Eigen::Matrix<float, NQ, 1>::Zero();
    for (int i = 0; i < N; ++i) qf[QI[i]] = q[i];
    const Eigen::Matrix<float, NS * 4 + NG * 3, 1> o = sym::@FK@<float>(qf);
    const unsigned long long t1 = TSC();
    bool hit = false;
    for (int g = NG - 1; g >= 0 && !hit; --g) {                // links, tip -> root
        const float* b = &o[NS * 4 + 3 * g];
        if (!env_hit(b[0], b[1], b[2], BR[g], w)) continue;    // bounding sphere clear: skip the link's spheres
        for (int s = GS[g + 1] - 1; s >= GS[g] && !hit; --s)
            hit = env_hit(o[4 * s], o[4 * s + 1], o[4 * s + 2], o[4 * s + 3], w);
    }
    const unsigned long long t2 = TSC();
    for (int k = 0; k < NLP && !hit; ++k) {                    // allowed link pairs, most distal first
        const float* a = &o[NS * 4 + 3 * LPA[k]];
        const float* b = &o[NS * 4 + 3 * LPB[k]];
        const float dx = a[0] - b[0], dy = a[1] - b[1], dz = a[2] - b[2], rb = BR[LPA[k]] + BR[LPB[k]];
        if (dx * dx + dy * dy + dz * dz >= rb * rb) continue;  // bounding spheres apart: skip the sphere pairs
        for (int p = LPS[k]; p < LPS[k + 1] && !hit; ++p) {
            const int i = PI[p], j = PJ[p];
            const float ex = o[4 * i] - o[4 * j], ey = o[4 * i + 1] - o[4 * j + 1], ez = o[4 * i + 2] - o[4 * j + 2];
            const float rs = o[4 * i + 3] + o[4 * j + 3];
            hit = ex * ex + ey * ey + ez * ez < rs * rs;
        }
    }
#ifdef SHRIMP_PROF
    const unsigned long long t3 = TSC();
    g_cyc[0] += t1 - t0; g_cyc[1] += t2 - t1; g_cyc[2] += t3 - t2;
#endif
    return !hit;
}
"""


def _spheres_bounded(robot, q):
    """q -> flat [x y z r] per sphere (as forward_kinematics_spheres), then the world centre (x y z) of each
    link's bounding sphere (sphere_order.link_bounding_spheres' order): a fixed point in the link frame."""
    primitive_xyzr, link_poses = forward_kinematics(robot, q)
    flat = [v for s in primitive_xyzr for v in s]
    for b in link_bounding_spheres(robot):
        c = link_poses[robot.link_name_to_index[b["link_name"]]] * sf.V3(*[float(x) for x in b["center"]])
        flat += [c[0], c[1], c[2]]
    return sf.Matrix(flat)


def _header(robot, q_index, lo, hi, pair_i, pair_j, gpu: bool) -> str:
    if gpu and len(pair_i) > 14000:
        raise ValueError(f"{len(pair_i)} self pairs do not fit in constant memory; move PI/PJ to global memory")
    arr = lambda xs, fmt="{:.9g}": "{" + ", ".join(fmt.format(x) for x in xs) + "}"  # noqa: E731
    extra = ""
    if gpu:
        # most distal pairs first (the sphere order is root -> tip), so the pair loop meets likely hits early
        order = sorted(range(len(pair_i)), key=lambda k: (max(pair_i[k], pair_j[k]), min(pair_i[k], pair_j[k])),
                       reverse=True)
        pair_i, pair_j = [pair_i[k] for k in order], [pair_j[k] for k in order]
        g = build_and_generate_function(robot, forward_kinematics_spheres, "crab_spheres")
        ns = g["n_out"] // 4
    else:
        # CPU: pairs grouped by allowed link pair (most distal first); one bounding-sphere test gates each group
        bsph = link_bounding_spheres(robot)
        ns = sum(b["count"] for b in bsph)
        gof = [gi for gi, b in enumerate(bsph) for _ in range(b["count"])]
        key = lambda k: (max(gof[pair_i[k]], gof[pair_j[k]]), min(gof[pair_i[k]], gof[pair_j[k]]),  # noqa: E731
                         max(pair_i[k], pair_j[k]), min(pair_i[k], pair_j[k]))
        order = sorted(range(len(pair_i)), key=key, reverse=True)
        pair_i, pair_j = [pair_i[k] for k in order], [pair_j[k] for k in order]
        lpa, lpb, lps = [], [], []
        for p in range(len(pair_i)):
            lp = (max(gof[pair_i[p]], gof[pair_j[p]]), min(gof[pair_i[p]], gof[pair_j[p]]))
            if not lpa or (lpa[-1], lpb[-1]) != lp:
                lpa.append(lp[0]); lpb.append(lp[1]); lps.append(p)
        lps.append(len(pair_i))
        g = build_and_generate_function(robot, _spheres_bounded, "crab_spheres_bs")
        extra = (f"constexpr int NG = {len(bsph)};\nstatic const int GS[] = "
                 f"{arr([b['first'] for b in bsph] + [ns], '{}')};\n"
                 f"static const float BR[] = {arr([b['radius'] for b in bsph] or [0.0])};\n"
                 f"constexpr int NLP = {len(lpa)};\n"
                 f"static const int LPA[] = {arr(lpa or [0], '{}')};\nstatic const int LPB[] = {arr(lpb or [0], '{}')};\n"
                 f"static const int LPS[] = {arr(lps, '{}')};\n")
    npair = max(len(pair_i), 1)
    q = "__constant__" if gpu else "static const"
    consts = (f"\nconstexpr int N = {len(q_index)};\nconstexpr int NQ = {robot.nq};\n"
              f"constexpr int NS = {ns};\nconstexpr int NPAIR = {len(pair_i)};\n" + extra +
              f"{q} int QI[] = {arr(q_index, '{}')};\n"
              f"{q} float LO[] = {arr(lo)};\n{q} float HI[] = {arr(hi)};\n"
              f"{q} unsigned short PI[{npair}] = {arr(list(pair_i) or [0], '{}')};\n"
              f"{q} unsigned short PJ[{npair}] = {arr(list(pair_j) or [0], '{}')};\n")
    head = "#include <Eigen/Dense>\n#include <cfloat>\n#include <cmath>\n#include <cstring>\n"
    fn = "__device__ __forceinline__" if gpu else "static inline"
    world = WORLD.replace("@FN@", fn) + (GPU_FKCC if gpu else CPU_FKCC)
    return "\n".join([head, device_source(g["source"]) if gpu else g["source"], consts,
                      world.replace("@FK@", g["func_name"])])


def build_robot_header(robot, q_index, lo: np.ndarray, hi: np.ndarray, pair_i, pair_j) -> str:
    """GPU (CUDA) header. robot: finalized CRAB Robot; q_index[i]: CRAB q index of the chain's i-th joint;
    lo/hi: chain limits; pair_i/pair_j: self-collision sphere pairs (indices in crab_codegen.sphere_order's order)."""
    return _header(robot, q_index, lo, hi, pair_i, pair_j, gpu=True)


def _fk_step(robot, q, parent, joint_name):
    """One traversal step of crab_kinematics.fk.forward_kinematics as a SymForce function: parent link pose
    (flat: R row-major 9, t 3) and q -> child link pose (same flat 12) followed by the child's collision sphere
    centres (x y z each). Same math: child = parent * origin * joint_transform."""
    j = robot.joints[joint_name]
    p = parent
    Rp = sf.Matrix([[p[0], p[1], p[2]], [p[3], p[4], p[5]], [p[6], p[7], p[8]]])
    tp = sf.V3(p[9], p[10], p[11])
    Ro, to = j.origin.R.to_rotation_matrix(), j.origin.t
    if j.type == "revolute":
        Rj, tj = sf.Rot3.from_angle_axis(q[j.id], j.axis).to_rotation_matrix(), sf.V3.zero()
    elif j.type == "prismatic":
        Rj, tj = sf.Matrix.eye(3), j.axis * q[j.id]
    elif j.type == "fixed":
        Rj, tj = sf.Matrix.eye(3), sf.V3.zero()
    else:
        raise ValueError(f"Unsupported joint type: {j.type}")
    R = Rp * Ro * Rj
    t = tp + Rp * (to + Ro * tj)
    out = [R[r, c] for r in range(3) for c in range(3)] + [t[0], t[1], t[2]]
    for prim in robot.links[j.child].primitives:
        c = t + R * prim.pose.t
        out += [c[0], c[1], c[2]]
    return sf.Matrix(out)


def _header_chain(robot, q_index, lo, hi, pair_i, pair_j) -> str:
    """CPU header whose fkcc interleaves FK with the collision tests: FK is generated one traversal step at a
    time (_fk_step), and after each step the child link's spheres are tested against the obstacles and against
    every earlier sphere (the pairs whose later sphere belongs to this link); the first hit returns false before
    the rest of the chain's FK is computed. Order is therefore root -> tip."""
    slot, stages, srcs, radii, base = {robot.root.name: 0}, [], [], [], 0
    for k, jn in enumerate(robot.traversal_order):
        j = robot.joints[jn]
        g = build_and_generate_function(robot, lambda r, q, parent, jn=jn: _fk_step(r, q, parent, jn),
                                        f"crab_fk_step_{k}", extra_inputs=[("parent", 12)])
        prims = robot.links[j.child].primitives
        slot[j.child] = k + 1
        stages.append((slot[j.parent], g["func_name"], len(prims), base, g["n_out"]))
        radii += [float(p.radius) for p in prims]
        srcs.append(g["source"])
        base += len(prims)
    ns = base
    order = sorted(range(len(pair_i)), key=lambda k: (max(pair_i[k], pair_j[k]), min(pair_i[k], pair_j[k])))
    pair_i, pair_j = [pair_i[k] for k in order], [pair_j[k] for k in order]
    pmax = [max(a, b) for a, b in zip(pair_i, pair_j)]

    body = ["static inline bool fkcc(const float* q, const World& w)", "{",
            "    Eigen::Matrix<float, NQ, 1> qf = Eigen::Matrix<float, NQ, 1>::Zero();",
            "    for (int i = 0; i < N; ++i) qf[QI[i]] = q[i];",
            f"    float pose[{12 * (len(stages) + 1)}] = {{1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0}};   // slot 0: root",
            f"    float o[{3 * max(ns, 1)}];"]
    for k, (ps, fn, nsph, b0, nout) in enumerate(stages):
        body += ["    {",
                 f"        const Eigen::Matrix<float, 12, 1> pp = Eigen::Map<const Eigen::Matrix<float, 12, 1>>(pose + {12 * ps});",
                 f"        const Eigen::Matrix<float, {nout}, 1> r = sym::{fn}<float>(qf, pp);",
                 f"        std::memcpy(pose + {12 * (k + 1)}, r.data(), 12 * sizeof(float));"]
        for s in range(nsph):
            g = b0 + s
            body += [f"        o[{3 * g}] = r[{12 + 3 * s}]; o[{3 * g + 1}] = r[{13 + 3 * s}]; o[{3 * g + 2}] = r[{14 + 3 * s}];",
                     f"        if (env_hit(o[{3 * g}], o[{3 * g + 1}], o[{3 * g + 2}], SR[{g}], w)) return false;"]
        a = sum(1 for m in pmax if m < b0)
        e = sum(1 for m in pmax if m < b0 + nsph)
        if e > a:
            body += [f"        for (int k = {a}; k < {e}; ++k) {{",
                     "            const int i = PI[k], j = PJ[k];",
                     "            const float dx = o[3 * i] - o[3 * j], dy = o[3 * i + 1] - o[3 * j + 1], dz = o[3 * i + 2] - o[3 * j + 2];",
                     "            const float rs = SR[i] + SR[j];",
                     "            if (dx * dx + dy * dy + dz * dz < rs * rs) return false;",
                     "        }"]
        body.append("    }")
    body += ["    return true;", "}"]

    arr = lambda xs, fmt="{:.9g}": "{" + ", ".join(fmt.format(x) for x in xs) + "}"  # noqa: E731
    npair = max(len(pair_i), 1)
    consts = (f"\nconstexpr int N = {len(q_index)};\nconstexpr int NQ = {robot.nq};\n"
              f"constexpr int NS = {ns};\nconstexpr int NPAIR = {len(pair_i)};\n"
              f"static const int QI[] = {arr(q_index, '{}')};\n"
              f"static const float LO[] = {arr(lo)};\nstatic const float HI[] = {arr(hi)};\n"
              f"static const unsigned short PI[{npair}] = {arr(list(pair_i) or [0], '{}')};\n"
              f"static const unsigned short PJ[{npair}] = {arr(list(pair_j) or [0], '{}')};\n"
              f"static const float SR[{max(ns, 1)}] = {arr(radii or [0.0])};\n")
    head = "#include <Eigen/Dense>\n#include <cfloat>\n#include <cmath>\n#include <cstring>\n"
    return "\n".join([head, *srcs, consts, WORLD.replace("@FN@", "static inline"), "\n".join(body)])


def build_cpu_robot_header(robot, q_index, lo: np.ndarray, hi: np.ndarray, pair_i, pair_j,
                           interleave: bool = False) -> str:
    """CPU (plain C++17 + Eigen) twin of build_robot_header: same constants and World; scalar
    `bool fkcc(const float* q, const World& w)` (no lane mask). interleave=True tests each link right after its FK
    step (root -> tip, exit at the first hit, FK of the rest skipped); False computes the full FK first, then
    tests spheres tip -> root (the original form)."""
    if interleave:
        return _header_chain(robot, q_index, lo, hi, pair_i, pair_j)
    return _header(robot, q_index, lo, hi, pair_i, pair_j, gpu=False)
