"""pyroffi FK of panda_hand, JAX and CUDA backends.
Run in the pyroffi env: python baseline_comparisons/fk_pyroffi.py"""

import os

import jax
import jax.numpy as jnp
import numpy as np
import pyroffi as pk
import yourdfpy

import common


def bench_backend(name: str, robot, use_cuda: bool, arm_idx: list[int], n_act: int, device: str) -> None:
    fk = jax.jit(lambda cfg: robot.forward_kinematics(cfg, use_cuda=use_cuda))
    link_idx = robot.links.names.index(common.LINK)

    def pad(q7: np.ndarray) -> jnp.ndarray:
        full = np.zeros((q7.shape[0], n_act), dtype=np.float32)  # fingers stay at 0
        full[:, arm_idx] = q7
        return jnp.asarray(full)

    per_config = {}
    for b in common.BATCH_SIZES:
        cfg = pad(common.sample_configs(b, seed=b))
        call = lambda: jax.block_until_ready(fk(cfg))
        per_config[b] = common.median_us(call, n_calls=200 if b <= 4096 else 50) / b
    latency = per_config[1]

    out = np.asarray(fk(pad(common.accuracy_configs())))[:, link_idx]  # (N, 7) wxyz_xyz
    poses = [common.to_pose_row(o[4:7], o[0:4]) for o in out]

    common.save(name, device, "float32", latency, per_config, poses,
                note=f"jax.jit + block_until_ready per call, use_cuda={use_cuda}")


def main() -> None:
    on_gpu = jax.default_backend() == "gpu"
    if not on_gpu and os.environ.get("JAX_PLATFORMS") != "cpu":
        raise SystemExit(f"JAX is on '{jax.default_backend()}', not GPU, and CPU wasn't requested; refusing to "
                         "record misleading numbers. Run with JAX_PLATFORMS=cuda to see the real plugin error, "
                         "or JAX_PLATFORMS=cpu for a deliberate CPU baseline.")
    device = "gpu" if on_gpu else "cpu"
    urdf =yourdfpy.URDF.load(str(common.REPO / "external/pyroffi/resources/panda/panda_spherized.urdf"))
    robot = pk.Robot.from_urdf(urdf)
    names = list(robot.joints.actuated_names)
    arm_idx = [names.index(f"panda_joint{i}") for i in range(1, 8)]
    n_act = robot.joints.num_actuated_joints

    bench_backend(f"pyroffi_jax_{device}", robot, False, arm_idx, n_act, device)
    if on_gpu:  # the CUDA FFI kernel needs a GPU
        try:
            bench_backend("pyroffi_cuda", robot, True, arm_idx, n_act, device)
        except Exception as e:  # GRiM/nvcc not available
            print(f"[pyroffi_cuda] skipped: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
