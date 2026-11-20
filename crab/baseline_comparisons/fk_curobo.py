"""cuRobo FK. Run in the curobo env: python baseline_comparisons/fk_curobo.py [--robot panda|rby1]
Loads the shared URDF directly (KinematicsCfg.from_basic_urdf: FK only, no collision).
Per-config times are GPU kernel time from the torch profiler, summed over all kernels one call
launches (same idea as the Caspar scripts); batch-1 wall-clock is recorded separately."""

import sys

import numpy as np
import torch

import common
sys.path.insert(0, str(common.REPO))

from crusty_cuda.profile import kernel_time_us  # torch only; no SymForce needed
from curobo.kinematics import Kinematics, KinematicsCfg
from curobo.types import JointState

torch.set_grad_enabled(False)


def main() -> None:
    spec = common.parse_robot()
    frames = [spec.link, *spec.extra_frames]  # cuRobo drops joints not on a requested frame's path
    cfg = KinematicsCfg.from_basic_urdf(str(spec.urdf_path), spec.base_link, frames)
    kin = Kinematics(cfg, compute_spheres=False)  # no spheres: FK only, like the others
    names = list(kin.joint_names)
    print(f"curobo dof={kin.get_dof()} joints={len(names)}")

    def make_js(q: np.ndarray) -> JointState:
        full = common.embed(spec, q, names)
        return JointState.from_position(torch.tensor(full, dtype=torch.float32, device="cuda"),
                                        joint_names=names)

    per_config = {}
    wall_batch1 = None
    for b in common.BATCH_SIZES:
        js = make_js(common.sample_configs(spec, b, seed=b))  # built outside the timer
        per_config[b] = kernel_time_us(lambda: kin.compute_kinematics(js)) / b
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
                note=f"GPU time of everything one compute_kinematics call launches (torch profiler; "
                     f"excludes Python overhead; wall b=1 includes it); computes {len(frames)} tool "
                     f"frame(s): {frames}")


if __name__ == "__main__":
    main()
