"""pyroffi FK, JAX and CUDA backends.
Run in the pyroffi env: python baseline_comparisons/fk_pyroffi.py [--robot panda|rby1]
GPU:  python ...    CPU baseline (JAX on CPU):  JAX_PLATFORMS=cpu python ...
Per-config times use the shared amortized protocol; batch-1 wall-clock is recorded separately."""

import os

import jax
import jax.numpy as jnp
import numpy as np
import pyroffi as pk
import yourdfpy

import common


def bench_backend(spec, name: str, robot, use_cuda: bool, device: str) -> None:
    fk = jax.jit(lambda cfg: robot.forward_kinematics(cfg, use_cuda=use_cuda))
    link_idx = robot.links.names.index(spec.link)
    names = list(robot.joints.actuated_names)  # fingers etc. not in spec.joints stay 0
    on_gpu = device == "gpu"

    def pad(q: np.ndarray) -> jnp.ndarray:
        return jnp.asarray(common.embed(spec, q, names), dtype=jnp.float32)

    batches = [b for b in common.BATCH_SIZES if on_gpu or b <= common.CPU_MAX_BATCH]
    n_inner, n_runs = (50, 9) if on_gpu else (3, 3)  # CPU at larger batches is slow

    per_config = {}
    wall_batch1 = None
    for b in batches:
        cfg = pad(common.sample_configs(spec, b, seed=b))
        per_config[b] = common.amortized_us(
            lambda: fk(cfg), jax.block_until_ready, n_inner=n_inner, n_runs=n_runs) / b
        if b == 1:
            wall_batch1 = common.median_us(lambda: jax.block_until_ready(fk(cfg)), n_calls=1000)

    out = np.asarray(fk(pad(common.accuracy_configs(spec))))[:, link_idx]  # (N, 7) wxyz_xyz
    poses = [common.to_pose_row(o[4:7], o[0:4]) for o in out]

    common.save(spec, name, device, "float32", wall_batch1, per_config, poses,
                note=f"jax.jit, use_cuda={use_cuda}; amortized over back-to-back calls "
                     f"(n_inner={n_inner}); computes all link poses")


def main() -> None:
    spec = common.parse_robot()
    on_gpu = jax.default_backend() == "gpu"
    if not on_gpu and os.environ.get("JAX_PLATFORMS") != "cpu":
        raise SystemExit(f"JAX is on '{jax.default_backend()}', not GPU, and CPU wasn't requested; refusing to "
                         "record misleading numbers. Run with JAX_PLATFORMS=cuda to see the real plugin error, "
                         "or JAX_PLATFORMS=cpu for a deliberate CPU baseline.")
    device = "gpu" if on_gpu else "cpu"

    robot = pk.Robot.from_urdf(yourdfpy.URDF.load(str(spec.urdf_path)))
    print(f"pyroffi actuated={robot.joints.num_actuated_joints} floating_base={robot.floating_base}")

    bench_backend(spec, f"pyroffi_jax_{device}", robot, False, device)
    if on_gpu:  # the CUDA FFI kernel needs a GPU
        try:
            bench_backend(spec, "pyroffi_cuda", robot, True, device)
        except Exception as e:  # GRiM/nvcc not available
            print(f"[pyroffi_cuda] skipped: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
