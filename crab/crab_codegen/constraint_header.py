"""CRAB's task-space-region (TSR) constraint header: error + Jacobian kernel (generated, see
crab_kinematics.constraints) and the numerics a planner needs to stay on the region: the bounds hinge, one damped
least-squares (Levenberg-Marquardt) step, and the projection loop. Planners (SHRIMP) include it, never the other way.

Concatenate it AFTER the robot header (build_robot_header / build_cpu_robot_header), which defines N, NQ, QI, LO, HI:

    src = build_robot_header(...) + build_tsr_header(robot, q_index, gpu=True) + planner

Contract (all float32; q is in the chain's joint order, like fkcc):

    constexpr int NE, NR             end-effector frames, error rows (6 per frame: x y z, then a rotation vector)
    struct Tsr                       one region: r_te[7 NE], w_tr[7 NE] ([qw qx qy qz x y z]), lb[NR], ub[NR], tol[NR]
                                     (host-side layout: pack_tsr). A row is satisfied inside [lb, ub]; tol is the slack
                                     at which a row counts as satisfied. Bounds and tolerance come from the region,
                                     never from the planner (an equality row is lb == ub with a tol).
    struct ProjParams                projection settings the caller chooses (iterations, damping, rate, stall, drift)
    tsr_eval(q, t, hold, e, J)       e: hinged error (0 inside the bounds); J: NR x N raw Jacobian, row-major. hold = 1
                                     keeps rows already inside their bounds in J (they then act as "hold this value"),
                                     0 zeroes them so projection may move them freely (McVAMP's hold_satisfied_rows)
    tsr_err(q, t, e)                 e only (a separate generated kernel: FK and the rotation log, no Jacobian): the
                                     cheap test of whether q is on the region, several times cheaper than tsr_eval
    tsr_satisfied(e, t)              every |e| <= tol
    tsr_step(q, e, J, p)             one LM step in place, clamped to the joint limits; returns the squared step length
    tsr_project(q, t, p)             step until satisfied; false on the iteration cap, a stall or too much drift
    tsr_check(q, t)                  q satisfies the region

The error and its Jacobian are of the raw se(3) displacement; see crab_kinematics.constraints.
"""

import numpy as np

from crab_codegen.codegen import build_and_generate_function
from crab_codegen.ik_kernel import device_source
from crab_kinematics.constraints import tsr_error, tsr_error_and_jacobian

TSR = r"""
#ifdef __CUDACC__
#define TSR_FN __host__ __device__ __forceinline__
#else
#define TSR_FN static inline
#endif

constexpr int NE = @NE@;
constexpr int NR = 6 * NE;

struct Tsr {
    float r_te[7 * NE];      // offset frame in the end-effector frame, per end-effector
    float w_tr[7 * NE];      // reference frame in the world, per end-effector
    float lb[NR], ub[NR];    // bounds on [translation(3); rotation vector(3)] of the offset frame in the reference frame
    float tol[NR];           // a row counts as satisfied within [lb - tol, ub + tol]
};

struct ProjParams {
    int max_iters;           // projection iteration cap
    float lambda;            // LM damping
    float rate;              // step scale (1 = the full LM step)
    float stall2;            // a step shorter than this (squared) means projection has stalled
    float drift2;            // abandon when q has moved this far (squared) from where projection started; <= 0: never
    int hold;                // 1: rows inside their bounds stay in the Jacobian (see tsr_eval)
};

// Hinged error e (NR) and raw Jacobian J (NR x N, row-major) of the region at q.
TSR_FN void tsr_eval(const float* q, const Tsr& t, bool hold, float* e, float* J)
{
    Eigen::Matrix<float, NQ, 1> qf = Eigen::Matrix<float, NQ, 1>::Zero();
    for (int i = 0; i < N; ++i) qf[QI[i]] = q[i];
    const Eigen::Matrix<float, 7 * NE, 1> rte = Eigen::Map<const Eigen::Matrix<float, 7 * NE, 1>>(t.r_te);
    const Eigen::Matrix<float, 7 * NE, 1> wtr = Eigen::Map<const Eigen::Matrix<float, 7 * NE, 1>>(t.w_tr);
    const Eigen::Matrix<float, NR + NR * NQ, 1> o = sym::@FN@<float>(qf, rte, wtr);
    for (int r = 0; r < NR; ++r) {
        const float v = o[r];
        e[r] = v < t.lb[r] ? v - t.lb[r] : (v > t.ub[r] ? v - t.ub[r] : 0.f);     // NaN compares false: masked to 0
        const bool live = hold || e[r] != 0.f;
        for (int c = 0; c < N; ++c) J[r * N + c] = live ? o[NR + r * NQ + QI[c]] : 0.f;
    }
}

// Hinged error e (NR) of the region at q: tsr_eval without the Jacobian (its own generated kernel).
TSR_FN void tsr_err(const float* q, const Tsr& t, float* e)
{
    Eigen::Matrix<float, NQ, 1> qf = Eigen::Matrix<float, NQ, 1>::Zero();
    for (int i = 0; i < N; ++i) qf[QI[i]] = q[i];
    const Eigen::Matrix<float, 7 * NE, 1> rte = Eigen::Map<const Eigen::Matrix<float, 7 * NE, 1>>(t.r_te);
    const Eigen::Matrix<float, 7 * NE, 1> wtr = Eigen::Map<const Eigen::Matrix<float, 7 * NE, 1>>(t.w_tr);
    const Eigen::Matrix<float, NR, 1> o = sym::@FNE@<float>(qf, rte, wtr);
    for (int r = 0; r < NR; ++r) {
        const float v = o[r];
        e[r] = v < t.lb[r] ? v - t.lb[r] : (v > t.ub[r] ? v - t.ub[r] : 0.f);
    }
}

TSR_FN bool tsr_satisfied(const float* e, const Tsr& t)
{
    for (int r = 0; r < NR; ++r)
        if (fabsf(e[r]) > t.tol[r]) return false;
    return true;
}

// Cholesky solve of the SPD system A z = y (K x K, row-major, lower triangle used); z is left in y.
template <int K>
TSR_FN void crab_chol_solve(float* A, float* y)
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

// One damped least-squares step in the task-space (dual) form: (J J^T + lambda I) z = e, q -= rate J^T z, clamped to
// the joint limits. The 6 NE x 6 NE system is small, so it is solved in place. Returns the squared step length.
TSR_FN float tsr_step(float* q, const float* e, const float* J, const ProjParams& p)
{
    float A[NR * NR], z[NR];
    for (int i = 0; i < NR; ++i) {
        for (int j = 0; j <= i; ++j) {
            float s = 0.f;
            for (int k = 0; k < N; ++k) s += J[i * N + k] * J[j * N + k];
            A[i * NR + j] = A[j * NR + i] = s;
        }
        A[i * NR + i] += p.lambda;
        z[i] = e[i];
    }
    crab_chol_solve<NR>(A, z);
    float s2 = 0.f;
    for (int k = 0; k < N; ++k) {
        float dq = 0.f;
        for (int i = 0; i < NR; ++i) dq += J[i * N + k] * z[i];
        const float nq = fminf(fmaxf(q[k] - p.rate * dq, LO[k]), HI[k]);
        const float d = nq - q[k];
        s2 += d * d;
        q[k] = nq;
    }
    return s2;
}

// Project q onto the region: true once every row is satisfied (q holds the result); false when the iteration cap is
// hit, a step stalls, or q drifts beyond drift2 from its start (q then holds the last iterate).
TSR_FN bool tsr_project(float* q, const Tsr& t, const ProjParams& p)
{
    float q0[N], e[NR], J[NR * N];
    for (int k = 0; k < N; ++k) q0[k] = q[k] = fminf(fmaxf(q[k], LO[k]), HI[k]);     // a candidate may start outside
    for (int it = 0;; ++it) {
        tsr_err(q, t, e);                                  // cheap: the Jacobian only when a step is needed
        if (tsr_satisfied(e, t)) return true;
        if (it >= p.max_iters) return false;
        tsr_eval(q, t, p.hold != 0, e, J);
        if (tsr_step(q, e, J, p) <= p.stall2) return false;
        if (p.drift2 > 0.f) {
            float d2 = 0.f;
            for (int k = 0; k < N; ++k) d2 += (q[k] - q0[k]) * (q[k] - q0[k]);
            if (d2 >= p.drift2) return false;
        }
    }
}

TSR_FN bool tsr_check(const float* q, const Tsr& t)
{
    float e[NR];
    tsr_err(q, t, e);
    return tsr_satisfied(e, t);
}
"""


def build_tsr_header(robot, eef_names=None, gpu: bool = True) -> str:
    """The TSR header (see the module docstring) for robot's end-effectors (robot.end_effectors, or eef_names).
    Append it to the robot header of the same robot; gpu=True marks the generated kernel __host__ __device__."""
    names = list(eef_names or robot.end_effectors)
    ne = len(names)
    g = build_and_generate_function(
        robot, lambda r, q, r_te, w_tr: tsr_error_and_jacobian(r, q, r_te, w_tr, names), "crab_tsr",
        extra_inputs=[("r_te", 7 * ne), ("w_tr", 7 * ne)])
    ge = build_and_generate_function(                      # the same error without the Jacobian
        robot, lambda r, q, r_te, w_tr: tsr_error(r, q, r_te, w_tr, names), "crab_tsr_err",
        extra_inputs=[("r_te", 7 * ne), ("w_tr", 7 * ne)])
    srcs = [device_source(x["source"]) if gpu else x["source"] for x in (g, ge)]
    return "\n".join([*srcs, TSR.replace("@NE@", str(ne)).replace("@FNE@", ge["func_name"])
                      .replace("@FN@", g["func_name"])])


def pack_tsr(r_te, w_tr, lb, ub, tol=1e-3) -> np.ndarray:
    """Host-side float32 layout of the C++ `Tsr`: r_te[7 NE], w_tr[7 NE], lb[NR], ub[NR], tol[NR].
    r_te / w_tr: (NE, 7) [qw qx qy qz x y z]; lb / ub: (NE, 6) or (6,) per frame [x y z rx ry rz] in the reference
    frame; tol: scalar or per row (NE * 6) slack at which a row counts as satisfied. The region owns its tolerance:
    pass it explicitly for anything tighter or looser than the default."""
    r_te, w_tr = np.atleast_2d(np.asarray(r_te, np.float32)), np.atleast_2d(np.asarray(w_tr, np.float32))
    ne = r_te.shape[0]
    if r_te.shape != (ne, 7) or w_tr.shape != (ne, 7):
        raise ValueError("r_te and w_tr must be (n_eef, 7) [qw qx qy qz x y z]")
    rows = lambda a: np.broadcast_to(np.asarray(a, np.float32), (ne, 6)).reshape(-1)  # noqa: E731
    lb, ub, tol = rows(lb), rows(ub), np.broadcast_to(np.asarray(tol, np.float32), (6 * ne,))
    if (lb > ub).any():
        raise ValueError("lower bound above upper bound")
    return np.concatenate([r_te.reshape(-1), w_tr.reshape(-1), lb, ub, tol]).astype(np.float32)
