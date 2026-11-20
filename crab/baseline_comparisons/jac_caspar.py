"""CRAB's GPU path (crusty_cuda / Caspar): end-effector position Jacobian (3 x nq).
Run in the crab env: python baseline_comparisons/jac_caspar.py [--robot panda|rby1] [--method symbolic|analytic]
Same amortized protocol as fk_caspar.py; analytic results are saved as caspar_analytic."""

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
from crab_kinematics.derived import q_to_ee_position_jacobian, q_to_ee_position_jacobian_analytic
from crusty_cuda import build_cuda_function, kernel_time_us


def main() -> None:
    args = common.parse_args()
    spec = args.spec
    fn, suffix = ((q_to_ee_position_jacobian_analytic, "_analytic") if args.method == "analytic"
                  else (q_to_ee_position_jacobian, ""))
    robot = urdf_to_robot(load_urdf(str(spec.urdf_path)), [spec.link]).finalize()
    assert robot.nq == spec.dof, (robot.nq, spec.dof)
    q_index = [robot.joints[n].id for n in spec.joints]

    def to_robot_q(q: np.ndarray) -> np.ndarray:
        out = np.zeros((q.shape[0], robot.nq))
        out[:, q_index] = q
        return out

    t0 = time.perf_counter()
    kernel = build_cuda_function(robot, fn, name=f"ee_jac_{spec.name}{suffix}")
    build_seconds = time.perf_counter() - t0
    print(f"crusty_cuda jacobian ({args.method}) build {build_seconds:.1f}s")

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

    res = kernel(to_robot_q(common.accuracy_configs(spec))).cpu().numpy()  # (N, 3 * nq) row-major
    rows = common.jac_to_spec_order(res.reshape(-1, 3, robot.nq), q_index)

    common.save(spec, f"caspar{suffix}", "gpu", "float32", wall_batch1, per_config, rows.tolist(),
                note=f"generated kernel's GPU time from the torch profiler (excludes Python/pybind "
                     f"launch overhead; wall b=1 includes it); {args.method} Jacobian; "
                     "--use_fast_math build",
                build_seconds=build_seconds, task="jac")


if __name__ == "__main__":
    main()
