"""CRAB's GPU backend (crusty_cuda, built on SymForce's Caspar) FK.
Run in the crab env: python baseline_comparisons/fk_caspar.py [--robot panda|rby1]

One CUDA thread per configuration, float32. Data stays in Caspar's structure-of-arrays layout,
so layout conversion is outside the timer (as it would be in a pipeline that stays on the GPU).
Per-config times use the shared amortized protocol (common.amortized_us); the batch-1
wall-clock (synchronized, includes Python/pybind launch overhead) is recorded separately.
"""

import os
import sys
import time

import numpy as np

os.environ.setdefault("CRAB_PRECISION", "float32")  # must precede importing crab_model

import common
sys.path.insert(0, str(common.REPO))

import crab_model  # noqa: F401
import torch

from crab_io.urdf import load_urdf, urdf_to_robot
from crab_kinematics.derived import q_to_ee_pose
from crusty_cuda import build_cuda_function, kernel_time_us


def main() -> None:
    spec = common.parse_robot()
    robot = urdf_to_robot(load_urdf(str(spec.urdf_path)), [spec.link]).finalize()
    assert robot.nq == spec.dof, (robot.nq, spec.dof)
    q_index = [robot.joints[n].id for n in spec.joints]

    def to_robot_q(q: np.ndarray) -> np.ndarray:  # sampled joint order -> CRAB q order
        out = np.zeros((q.shape[0], robot.nq))
        out[:, q_index] = q
        return out

    t0 = time.perf_counter()
    kernel = build_cuda_function(robot, q_to_ee_pose, name=f"q_to_ee_{spec.name}")
    build_seconds = time.perf_counter() - t0  # trace + CUDA codegen + nvcc (incremental if cached)
    print(f"crusty_cuda build {build_seconds:.1f}s")

    per_config = {}
    wall_batch1 = None
    for b in common.BATCH_SIZES:
        q_cas = kernel.to_caspar(to_robot_q(common.sample_configs(spec, b, seed=b)))
        out = kernel.run_caspar(q_cas, [], b)
        per_config[b] = kernel_time_us(lambda: kernel.run_caspar(q_cas, [], b, out=out), kernel.name) / b
        if b == 1:
            def call():
                kernel.run_caspar(q_cas, [], 1, out=out)
                torch.cuda.synchronize()
            wall_batch1 = common.median_us(call, n_calls=1000)

    res = kernel(to_robot_q(common.accuracy_configs(spec))).cpu().numpy()  # [qx,qy,qz,qw,x,y,z]
    poses = [common.to_pose_row(o[4:7], [o[3], o[0], o[1], o[2]]) for o in res]

    common.save(spec, "caspar", "gpu", "float32", wall_batch1, per_config, poses,
                note="generated kernel's GPU time from the torch profiler (excludes Python/pybind "
                     "launch overhead; wall b=1 includes it); data pre-converted to caspar layout; "
                     "build_seconds = trace + codegen + nvcc (delete ~/.cache/crusty/cuda for cold)",
                build_seconds=build_seconds)


if __name__ == "__main__":
    main()
