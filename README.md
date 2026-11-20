<p align="center"><img src="assets/shrimp.svg" alt="SHRIMP" width="140"></p>

# SHRIMP

A motion-planning library built for fast, repeated planning calls: many problems at once on
the GPU, or one at a time on a CPU core, so a task planner can call it in its inner loop.

- **`src/`**: the planners (RRT-Connect on CPU and CUDA).
- **`crab/`**: CRAB, the kernel library underneath. Robot descriptions in Python are
  JIT-compiled into fast kinematics and collision kernels that the planners use.
  SHRIMP depends on CRAB, never the other way round. See [`crab/README.md`](crab/README.md).
- **`ik_benchmark/`**: benchmark problems, runners and plots.
- **`external/`**: third-party checkouts for baselines (not tracked).

Status: work in progress.
