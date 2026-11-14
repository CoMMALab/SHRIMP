"""cuRobo FK of panda_hand. Run in the curobo env: python baseline_comparisons/fk_curobo.py"""

import numpy as np
import torch

import common
from curobo.kinematics import Kinematics, KinematicsCfg
from curobo.types import JointState


def main() -> None:
    kin = Kinematics(KinematicsCfg.from_robot_yaml_file("franka.yml"))
    assert kin.get_dof() == 7, kin.get_dof()
    dev = "cuda"

    def run(q: torch.Tensor):
        js = JointState.from_position(q, joint_names=kin.joint_names)
        return kin.compute_kinematics(js)

    def timed(batch: int, n_calls: int) -> float:
        lo = torch.tensor(common.PANDA_LO, dtype=torch.float32, device=dev)
        hi = torch.tensor(common.PANDA_HI, dtype=torch.float32, device=dev)
        q = lo + (hi - lo) * torch.rand(batch, 7, device=dev, dtype=torch.float32)
        js = JointState.from_position(q, joint_names=kin.joint_names)  # built outside the timer

        def call():
            kin.compute_kinematics(js)
            torch.cuda.synchronize()

        return common.median_us(call, n_calls=n_calls)

    per_config = {}
    for b in common.BATCH_SIZES:
        per_config[b] = timed(b, n_calls=200 if b <= 4096 else 50) / b
    latency = per_config[1]

    q_acc = torch.tensor(common.accuracy_configs(), dtype=torch.float32, device=dev)
    pose = run(q_acc).tool_poses.get_link_pose(common.LINK)
    pos = pose.position.detach().cpu().numpy().reshape(-1, 3)
    quat = pose.quaternion.detach().cpu().numpy().reshape(-1, 4)  # wxyz
    poses = [common.to_pose_row(p, qt) for p, qt in zip(pos, quat)]

    common.save("curobo", "gpu", "float32", latency, per_config, poses,
                note="compute_kinematics only (JointState built outside timer); synchronized per call")


if __name__ == "__main__":
    main()
