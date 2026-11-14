"""CRAB FK of panda_hand. Run in the CRAB env: python baseline_comparisons/fk_crab.py"""

import sys

import numpy as np

import common
sys.path.insert(0, str(common.REPO))

import crab_model  # noqa: F401 -- must precede any symforce.symbolic import
from crab_io.urdf import load_urdf, urdf_to_robot
from crab_kinematics.derived import q_to_ee_pose
from crusty_jit import build_simple_jit_function


def main() -> None:
    urdf = load_urdf(str(common.REPO / "robots/panda/panda_spherized.urdf"))
    robot = urdf_to_robot(urdf, [common.LINK]).finalize()
    kernel = build_simple_jit_function(robot, q_to_ee_pose, name="q_to_ee_panda_hand")

    q = np.zeros(7)
    latency = common.median_us(lambda: kernel(q), n_calls=10_000)

    # CPU, one config per call: no batching, so per-config time equals batch-1 latency.
    per_config = {1: latency}

    poses = []
    for cfg in common.accuracy_configs():
        out = np.asarray(kernel(cfg))  # [qx, qy, qz, qw, x, y, z] (SymForce Pose3 storage)
        poses.append(common.to_pose_row(out[4:7], [out[3], out[0], out[1], out[2]]))

    common.save("crab", "cpu", "float64", latency, per_config, poses,
                note="includes Python binder overhead (ctypes buffer build + np array out)")


if __name__ == "__main__":
    main()
