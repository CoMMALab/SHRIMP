"""CRAB FK. Run in the CRAB env: python baseline_comparisons/fk_crab.py [--robot panda|rby1]"""

import sys
import time

import numpy as np

import common
sys.path.insert(0, str(common.REPO))

import crab_model  # noqa: F401 -- must precede any symforce.symbolic import
from crab_io.urdf import load_urdf, urdf_to_robot
from crab_kinematics.derived import q_to_ee_pose
from crusty_jit import build_simple_jit_function


def main() -> None:
    spec = common.parse_robot()
    urdf = load_urdf(str(spec.urdf_path))
    robot = urdf_to_robot(urdf, [spec.link]).finalize()
    assert robot.nq == spec.dof, (robot.nq, spec.dof)

    # Joint.id is the q index; map the sampled joint order onto it by name.
    q_index = [robot.joints[n].id for n in spec.joints]
    assert sorted(q_index) == list(range(robot.nq)), q_index

    def to_robot_q(q: np.ndarray) -> np.ndarray:
        out = np.zeros(robot.nq)
        out[q_index] = q
        return out

    t0 = time.perf_counter()
    kernel = build_simple_jit_function(robot, q_to_ee_pose, name=f"q_to_ee_{spec.name}")
    build_seconds = time.perf_counter() - t0  # codegen + JIT compile (disk-cached after the first run!)

    q = to_robot_q(np.zeros(spec.dof))
    latency = common.median_us(lambda: kernel(q), n_calls=10_000)

    # CPU, one config per call: no batching, so per-config time equals batch-1 latency.
    per_config = {1: latency}

    poses = []
    for cfg in common.accuracy_configs(spec):
        out = np.asarray(kernel(to_robot_q(cfg)))  # [qx, qy, qz, qw, x, y, z] (SymForce Pose3 storage)
        poses.append(common.to_pose_row(out[4:7], [out[3], out[0], out[1], out[2]]))

    common.save(spec, "crab", "cpu", "float64", latency, per_config, poses,
                note="includes Python binder overhead (ctypes buffer build + np array out)",
                build_seconds=build_seconds)


if __name__ == "__main__":
    main()
