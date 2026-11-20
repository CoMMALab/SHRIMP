"""CRAB CPU JIT: end-effector position Jacobian (3 x nq).
Run in the crab env: python baseline_comparisons/jac_crab.py [--robot panda|rby1] [--method symbolic|analytic]
symbolic: symbolic differentiation of FK. analytic: the geometric formula (saved as crab_analytic).
Build time (derivation + codegen + clang -O3) is the scaling number to watch."""

import sys
import time

import numpy as np

import common
sys.path.insert(0, str(common.REPO))

import crab_model  # noqa: F401 -- must precede any symforce.symbolic import
from crab_io.urdf import load_urdf, urdf_to_robot
from crab_kinematics.derived import q_to_ee_position_jacobian, q_to_ee_position_jacobian_analytic
from crusty_jit import build_simple_jit_function


def main() -> None:
    args = common.parse_args()
    spec = args.spec
    fn, suffix = ((q_to_ee_position_jacobian_analytic, "_analytic") if args.method == "analytic"
                  else (q_to_ee_position_jacobian, ""))
    robot = urdf_to_robot(load_urdf(str(spec.urdf_path)), [spec.link]).finalize()
    assert robot.nq == spec.dof, (robot.nq, spec.dof)
    q_index = [robot.joints[n].id for n in spec.joints]

    def to_robot_q(q: np.ndarray) -> np.ndarray:
        out = np.zeros(robot.nq)
        out[q_index] = q
        return out

    t0 = time.perf_counter()
    kernel = build_simple_jit_function(robot, fn, name=f"ee_jac_{spec.name}{suffix}")
    build_seconds = time.perf_counter() - t0
    print(f"crab jacobian ({args.method}) build {build_seconds:.1f}s")

    q = to_robot_q(np.zeros(spec.dof))
    latency = common.median_us(lambda: kernel(q), n_calls=10_000)

    rows = []
    for cfg in common.accuracy_configs(spec):
        jac = np.asarray(kernel(to_robot_q(cfg))).reshape(1, 3, robot.nq)
        rows.append(common.jac_to_spec_order(jac, q_index)[0])

    common.save(spec, f"crab{suffix}", "cpu", "float64", latency, {1: latency}, [r.tolist() for r in rows],
                note=f"includes Python binder overhead; {args.method} position Jacobian",
                build_seconds=build_seconds, task="jac")


if __name__ == "__main__":
    main()
