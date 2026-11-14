"""cuRobo FK. Run in the curobo env: python baseline_comparisons/fk_curobo.py [--robot panda|rby1]
Loads the shared URDF directly (KinematicsCfg.from_basic_urdf: FK only, no collision)."""

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

    def timed(batch: int, n_calls: int) -> float:
        js = make_js(common.sample_configs(spec, batch, seed=batch))  # built outside the timer

        def call():
            kin.compute_kinematics(js)
            torch.cuda.synchronize()

        return common.median_us(call, n_calls=n_calls)

    per_config = {}
    for b in common.BATCH_SIZES:
        per_config[b] = timed(b, n_calls=200 if b <= 4096 else 50) / b
    latency = per_config[1]

    state = kin.compute_kinematics(make_js(common.accuracy_configs(spec)))
    pose = state.tool_poses.get_link_pose(spec.link)
    pos = pose.position.detach().cpu().numpy().reshape(-1, 3)
    quat = pose.quaternion.detach().cpu().numpy().reshape(-1, 4)  # wxyz
    poses = [common.to_pose_row(p, qt) for p, qt in zip(pos, quat)]

    common.save(spec, "curobo", "gpu", "float32", latency, per_config, poses,
                note=f"compute_kinematics only (JointState built outside timer); synchronized per call; "
                     f"computes {len(frames)} tool frame(s): {frames}")


if __name__ == "__main__":
    main()
