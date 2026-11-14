# CRAB

Python-authored, JIT-compiled robot kinematics and collision kernels, for embedding
in native planners. Write a SymForce function of `q`; get a `-O3` kernel at runtime.

Status: work in progress. See `JIT_ARCHITECTURE.md` for how the pieces fit together.

## Build the JIT library

```
cmake -S crusty_compilations -B crusty_compilations/build
cmake --build crusty_compilations/build --target crab_jit_capi
```

Requires LLVM/Clang 18 dev libraries, Eigen3, and `symforce`, `yourdfpy`, `numpy`.

## Try it

```
python -m examples.demo
python -m tests.test_kernel_correctness
python -m benchmarks.bench_kernels
```

## Layout

- `crusty_model/`, `crusty_io/`, `crusty_kinematics/`: robot description, URDF loading, symbolic FK.
- `crusty_codegen/`: SymForce to C++ source.
- `crusty_jit/`: JIT engine and Python binder.
- `crusty_compilations/`: Clang/ORC JIT (C++).
- `robots/panda/`: spherized Panda URDF, SRDF, meshes (for visualization).
