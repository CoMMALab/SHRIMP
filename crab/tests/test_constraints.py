"""TSR constraint kernel checks (CRAB phase 1). Run from crab/: `python -m tests.test_constraints`

The kernel is JIT-compiled (float64) from crab_kinematics.constraints.tsr_error_and_jacobian and evaluated through
the same compiled function everywhere, so this checks: the error vanishes on the region's own reference pose, the
generated Jacobian matches central finite differences of the generated error (layout and autodiff), and a rotated
reference gives a rotation row equal to the applied rotation vector. The C++ hinge / LM step / projection loop
(crab_codegen.constraint_header) are exercised by the SHRIMP planners' validators, not here.
"""

import numpy as np

import crab_model  # noqa: F401 -- must run before anything imports symforce.symbolic

from crab_codegen.fixtures import load_panda_robot
from crab_kinematics.constraints import tsr_error, tsr_error_and_jacobian
from crab_kinematics.derived import q_to_ee_pose
from crusty_jit import build_simple_jit_function


def quat_mul(a, b):
    """wxyz quaternion product a * b."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw])


def axis_angle_quat(v):
    """Rotation vector -> wxyz quaternion."""
    ang = np.linalg.norm(v)
    return np.array([1.0, 0, 0, 0]) if ang < 1e-12 else np.concatenate([[np.cos(ang / 2)], np.sin(ang / 2) * v / ang])


def main() -> None:
    robot = load_panda_robot()
    nq = robot.nq
    ne = len(robot.end_effectors)
    nr = 6 * ne
    pose_fn = build_simple_jit_function(robot, q_to_ee_pose, name="tsr_test_pose")
    kern = build_simple_jit_function(
        robot, tsr_error_and_jacobian, name="tsr_test_err_jac",
        extra_inputs=[("r_te", 7 * ne), ("w_tr", 7 * ne)])
    kern_err = build_simple_jit_function(
        robot, tsr_error, name="tsr_test_err", extra_inputs=[("r_te", 7 * ne), ("w_tr", 7 * ne)])
    ident = np.tile([1.0, 0, 0, 0, 0, 0, 0], ne)

    def err_jac(q, r_te, w_tr):
        out = np.asarray(kern(q, r_te, w_tr), dtype=np.float64).reshape(-1)
        return out[:nr], out[nr:].reshape(nr, nq)

    rng = np.random.default_rng(0)
    for trial in range(10):
        q = rng.uniform(-1.0, 1.0, nq)
        s = np.asarray(pose_fn(q), dtype=np.float64).reshape(-1)         # Pose3 storage: qx qy qz qw x y z
        ref = np.concatenate([[s[3], s[0], s[1], s[2]], s[4:7]])          # wxyz + xyz
        w_tr = np.tile(ref, ne)

        e0, _ = err_jac(q, ident, w_tr)
        assert np.abs(e0).max() < 1e-6, f"error on the reference pose itself: {e0}"
        # the error-only kernel (what the planners call first) equals the error rows of the full kernel
        e_only = np.asarray(kern_err(q, ident, np.tile(np.concatenate([quat_mul(ref[:4], axis_angle_quat([0.1, -0.2, 0.3])),
                                                                         ref[4:] + 0.02]), ne)), dtype=np.float64).reshape(-1)
        e_full, _ = err_jac(q, ident, np.tile(np.concatenate([quat_mul(ref[:4], axis_angle_quat([0.1, -0.2, 0.3])),
                                                               ref[4:] + 0.02]), ne))
        assert np.allclose(e_only, e_full, atol=1e-12), f"error-only kernel {e_only} != full kernel {e_full}"

        # reference rotated by a known rotation vector: the rotation rows are its negative (r_T_e = R_applied^-1)
        phi = rng.uniform(-0.3, 0.3, 3)
        shifted = np.concatenate([quat_mul(ref[:4], axis_angle_quat(phi)), ref[4:]])
        e1, _ = err_jac(q, ident, np.tile(shifted, ne))
        assert np.allclose(e1[3:6], -phi, atol=1e-6), f"rotation row {e1[3:6]} != {-phi}"

        # Jacobian against central finite differences of the same kernel, at a reference away from q's own pose
        w_off = np.tile(np.concatenate([quat_mul(ref[:4], axis_angle_quat(phi)), ref[4:] + 0.05]), ne)
        _, J = err_jac(q, ident, w_off)
        h = 1e-6
        Jn = np.zeros_like(J)
        for j in range(nq):
            dq = np.zeros(nq)
            dq[j] = h
            Jn[:, j] = (err_jac(q + dq, ident, w_off)[0] - err_jac(q - dq, ident, w_off)[0]) / (2 * h)
        assert np.allclose(J, Jn, atol=1e-5), f"Jacobian mismatch (trial {trial}): max {np.abs(J - Jn).max():.2e}"
    print("TSR error / Jacobian checks: OK")


if __name__ == "__main__":
    main()
