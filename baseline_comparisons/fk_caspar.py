"""CRAB's GPU backend (crusty_cuda, built on SymForce's Caspar) FK.
Run in the crab env: python baseline_comparisons/fk_caspar.py [--robot panda|rby1]

One CUDA thread per configuration, float32. Data stays in Caspar's structure-of-arrays layout,
so layout conversion is outside the timer (as it would be in a pipeline that stays on the GPU).
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
from crusty_cuda import build_cuda_function


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

    def timed(batch: int, n_calls: int) -> float:
        q_cas = kernel.to_caspar(to_robot_q(common.sample_configs(spec, batch, seed=batch)))
        out = kernel.run_caspar(q_cas, [], batch)

        def call():
            kernel.run_caspar(q_cas, [], batch, out=out)
            torch.cuda.synchronize()

        return common.median_us(call, n_calls=n_calls)

    per_config = {}
    for b in common.BATCH_SIZES:
        per_config[b] = timed(b, n_calls=200 if b <= 4096 else 50) / b
    latency = per_config[1]

    res = kernel(to_robot_q(common.accuracy_configs(spec))).cpu().numpy()  # [qx,qy,qz,qw,x,y,z]
    poses = [common.to_pose_row(o[4:7], [o[3], o[0], o[1], o[2]]) for o in res]

    common.save(spec, "caspar", "gpu", "float32", latency, per_config, poses,
                note="crusty_cuda kernel launch only (data pre-converted to caspar layout), "
                     "synchronized per call; build_seconds = trace + codegen + nvcc "
                     "(delete ~/.cache/crusty/cuda for a cold build)",
                build_seconds=build_seconds)


if __name__ == "__main__":
    main()
