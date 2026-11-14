# Baseline comparisons

Compare CRAB against cuRobo, pyroffi and cricket (cloned under `external/`, gitignored).

## Environment

```
docker build -f baseline_comparisons/Dockerfile -t crab-baselines .
docker run --rm -it --gpus all -v "$PWD":/workspace crab-baselines
bash baseline_comparisons/setup.sh
```

Requires the NVIDIA container toolkit on the host. Each library has its own env (see the
Dockerfile header); run each library's benchmark in its own env and write results to a
shared CSV format, then compare outside.

## What each baseline gives us

| | Regime | FK entry point | Robots (same Panda URDF where possible) |
|---|---|---|---|
| CRAB | CPU, single query | `build_simple_jit_function(robot, forward_kinematics_spheres)` | `robots/panda/panda_spherized.urdf` |
| cricket | CPU, single query, float32 | generated `eefk` / `spherefk` in `fk_template.hh` | `external/cricket/resources/panda/panda_spherized.urdf` |
| pyroffi | GPU, batched | `pk.Robot.from_urdf(...).forward_kinematics`; existing `external/pyroffi/tests/bench_fk.py` | `external/pyroffi/resources/{panda,fetch,baxter,g1}/` |
| cuRobo | GPU, batched | `curobo.kinematics.Kinematics.compute_kinematics`; `external/curobo/curobo/examples/getting_started/forward_kinematics.py` | its own `franka.yml`, `unitree_g1_29dof.yml` |

## Comparison caveats

- Regimes differ: CPU single-query latency (CRAB, cricket) vs GPU batched throughput (cuRobo,
  pyroffi). Report both batch=1 latency and throughput vs batch size; don't claim a winner
  across regimes.
- Python call overhead: CRAB's binder adds a few microseconds per call. Time the kernel in C++
  too, or the comparison with cricket is unfair.
- Precision: cricket is float32, CRAB is float64. Cross-check outputs numerically before timing.
- Same sphere set: cuRobo's `franka.yml` has its own spheres, so compare link/EE poses, not
  sphere FK, unless its spheres are replaced.
- Exclude compilation/warm-up (cuRobo graph capture, JAX jit, CRAB JIT) from latency.

## Open items

- cricket's generated header includes VAMP headers, which are not in `external/`. Either clone
  VAMP, or generate from a custom template without VAMP includes (its `cricket.jit.JitSession`
  could then compile it for in-Python timing). Not yet verified.
- High-DOF: pyroffi and cuRobo ship a G1 humanoid. CRAB and cricket would need a spherized G1
  URDF (pyroffi has one under `resources/g1_description/`).
- The `nvidia/cuda:13.0.0-devel-ubuntu24.04` tag and the `cu130` PyTorch index are unverified;
  override with `--build-arg CUDA_IMAGE=...` if the build fails.
