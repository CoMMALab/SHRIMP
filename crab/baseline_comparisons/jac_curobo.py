"""cuRobo: end-effector position Jacobian (3 x dof) from Kinematics(compute_jacobian=True).
Run in the curobo env: python baseline_comparisons/jac_curobo.py [--robot panda|rby1]

cuRobo's Jacobian layout isn't documented where I looked, so the script validates it: it
finite-differences cuRobo's own FK positions, and picks the linear rows ([0:3] or [3:6]) that
match; it raises if neither does. Timing is GPU kernel time from the torch profiler, summed over
all kernels one call launches (so it includes cuRobo's FK, which the Jacobian needs)."""

import sys

import numpy as np
import torch

import common
sys.path.insert(0, str(common.REPO))

from crusty_cuda.profile import kernel_time_us  # torch only; no SymForce needed
from curobo.kinematics import Kinematics, KinematicsCfg
from curobo.types import JointState

torch.set_grad_enabled(False)
FD_EPS = 1e-3
FD_TOL = 5e-2  # float32 central differences; only has to tell the row blocks apart


def main() -> None:
    spec = common.parse_robot()
    frames = [spec.link, *spec.extra_frames]
    cfg = KinematicsCfg.from_basic_urdf(str(spec.urdf_path), spec.base_link, frames)
    kin = Kinematics(cfg, compute_jacobian=True, compute_spheres=False)
    names = list(kin.joint_names)
    dof = len(names)
    tool = list(kin.tool_frames).index(spec.link)
    column_of = [names.index(n) for n in spec.joints]
    print(f"curobo dof={dof} tool_frames={list(kin.tool_frames)}")

    def js_full(q_full: np.ndarray) -> JointState:  # q in cuRobo's own joint order
        return JointState.from_position(torch.tensor(q_full, dtype=torch.float32, device="cuda"),
                                        joint_names=names)

    def position(q_full: np.ndarray) -> np.ndarray:
        pose = kin.compute_kinematics(js_full(q_full)).tool_poses.get_link_pose(spec.link)
        return pose.position.detach().cpu().numpy().reshape(-1, 3)

    def jac_rows(q_full: np.ndarray) -> np.ndarray:  # (B, 6, dof), as cuRobo lays it out
        J = kin.compute_kinematics(js_full(q_full)).tool_jacobians
        B, n_tools = q_full.shape[0], len(kin.tool_frames)
        if J.numel() != B * n_tools * 6 * dof:
            raise RuntimeError(f"unexpected cuRobo Jacobian shape {tuple(J.shape)} for B={B}, "
                               f"{n_tools} tools, dof={dof}")
        return J.reshape(B, n_tools, 6, dof)[:, tool].detach().cpu().numpy()

    # Validate the layout against finite differences of cuRobo's own FK.
    q_chk = common.embed(spec, common.sample_configs(spec, 8, seed=7), names)
    fd = np.zeros((8, 3, dof))
    for j in range(dof):
        hi, lo = q_chk.copy(), q_chk.copy()
        hi[:, j] += FD_EPS
        lo[:, j] -= FD_EPS
        fd[:, :, j] = (position(hi) - position(lo)) / (2 * FD_EPS)
    J_chk = jac_rows(q_chk)
    errs = {rows: float(np.max(np.abs(J_chk[:, rows:rows + 3, :] - fd))) for rows in (0, 3)}
    first = min(errs, key=errs.get)
    if errs[first] > FD_TOL:
        raise RuntimeError(f"cuRobo Jacobian matches neither row block against finite differences: {errs}")
    print(f"cuRobo linear rows start at {first} (FD errors {errs})")

    per_config = {}
    wall_batch1 = None
    for b in common.BATCH_SIZES:
        js = js_full(common.embed(spec, common.sample_configs(spec, b, seed=b), names))
        per_config[b] = kernel_time_us(lambda: kin.compute_kinematics(js)) / b
        if b == 1:
            def call():
                kin.compute_kinematics(js)
                torch.cuda.synchronize()
            wall_batch1 = common.median_us(call, n_calls=1000)

    q_acc = common.embed(spec, common.accuracy_configs(spec), names)
    lin = jac_rows(q_acc)[:, first:first + 3, :]                    # (N, 3, dof)
    rows = common.jac_to_spec_order(lin, column_of)

    common.save(spec, "curobo", "gpu", "float32", wall_batch1, per_config, rows.tolist(),
                note=f"GPU time of everything one compute_kinematics(compute_jacobian=True) call "
                     f"launches (torch profiler; excludes Python overhead); computes {len(frames)} "
                     f"tool frame(s): {frames}; linear rows validated against finite differences",
                build_seconds=None, task="jac")


if __name__ == "__main__":
    main()
