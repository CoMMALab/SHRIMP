"""cuRobo FK. Run in the curobo env: python baseline_comparisons/fk_curobo.py [--robot panda|rby1]
Loads the shared URDF directly (KinematicsCfg.from_basic_urdf: FK only, no collision).
Per-config times use the shared amortized protocol; batch-1 wall-clock is recorded separately."""

import numpy as np
import torch

import common
from curobo.kinematics import Kinematics, KinematicsCfg
from curobo.types import JointState


def main() -> None:
    spec = common.parse_robot()
    frames = [spec.link, *spec.extra_frames]  # cuRobo drops joints not on a requested frame's path
    kin = Kinematics(KinematicsCfg.from_basic_urdf(str(spec.urdf_path), spec.base_link, frames))
    names = list(kin.joint_names)
    dev = "cuda"
    print(f"curobo dof={kin.get_dof()} joints={len(names)}")

    def make_js(q7: np.ndarray) -> JointState:
        full = common.embed(spec, q7, names)
        q = torch.tensor(full, dtype=torch.float32, device=dev)
        return JointState.from_position(q, joint_names=names)

    per_config = {}
    wall_batch1 = None
    for b in common.BATCH_SIZES:
        js = make_js(common.sample_configs(spec, b, seed=b))  # built outside the timer
        per_config[b] = common.amortized_us(
            lambda: kin.compute_kinematics(js), lambda _: torch.cuda.synchronize()) / b
        if b == 1:
            def call():
                kin.compute_kinematics(js)
                torch.cuda.synchronize()
            wall_batch1 = common.median_us(call, n_calls=1000)

    state = kin.compute_kinematics(make_js(common.accuracy_configs(spec)))
    pose = state.tool_poses.get_link_pose(spec.link)
    pos = pose.position.detach().cpu().numpy().reshape(-1, 3)
    quat = pose.quaternion.detach().cpu().numpy().reshape(-1, 4)  # wxyz
    poses = [common.to_pose_row(p, qt) for p, qt in zip(pos, quat)]

    common.save(spec, "curobo", "gpu", "float32", wall_batch1, per_config, poses,
                note=f"compute_kinematics only (JointState built outside timer); amortized over "
                     f"back-to-back calls; computes {len(frames)} tool frame(s): {frames}")


if __name__ == "__main__":
    main()
