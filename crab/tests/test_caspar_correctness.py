"""Checks the GPU path (crusty_cuda) against the CPU JIT for the same symbolic function.
Needs a CUDA GPU, nvcc, torch, and SymForce with caspar (the crab env); run:
`python -m tests.test_caspar_correctness`
"""

import os

os.environ.setdefault("CRAB_PRECISION", "float32")  # must precede importing crab_model

import numpy as np

import crab_model  # noqa: F401

from crab_codegen.fixtures import load_panda_robot
from crab_kinematics.derived import q_to_ee_pose
from crusty_cuda import build_cuda_function
from crusty_jit import build_simple_jit_function


def main(n: int = 256) -> None:
    robot = load_panda_robot()
    cpu = build_simple_jit_function(robot, q_to_ee_pose, name="q_to_ee")
    gpu = build_cuda_function(robot, q_to_ee_pose, name="q_to_ee")

    lo = np.array([j.limits[0] for j in robot.joints.values() if j.is_actuated])
    hi = np.array([j.limits[1] for j in robot.joints.values() if j.is_actuated])
    q = np.random.default_rng(0).uniform(lo, hi, size=(n, robot.nq))

    got = gpu(q).cpu().numpy()                   # (n, 7) float32: [qx, qy, qz, qw, x, y, z]
    ref = np.stack([cpu(row) for row in q])      # float64

    pos_err = np.max(np.linalg.norm(got[:, 4:] - ref[:, 4:], axis=1))
    quat_dot = np.abs(np.sum(got[:, :4] * ref[:, :4], axis=1))
    rot_err = np.max(2.0 * np.arccos(np.clip(quat_dot, 0.0, 1.0)))
    assert pos_err < 1e-4, f"position error {pos_err:.2e} m"
    assert rot_err < 1e-3, f"rotation error {rot_err:.2e} rad"
    print(f"q_to_ee: GPU matches CPU ({n} samples): pos err {pos_err:.2e} m, rot err {rot_err:.2e} rad")


if __name__ == "__main__":
    main()
