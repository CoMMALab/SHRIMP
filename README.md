# CRAB

Python-authored, JIT-compiled robot kinematics and collision kernels, for embedding
in native planners. Write a SymForce function of `q`; get a `-O3` kernel at runtime.

Status: work in progress. See `JIT_ARCHITECTURE.md` for how the pieces fit together.

## Build the JIT library

```
cmake -S crusty_compilations -B crusty_compilations/build
cmake --build crusty_compilations/build --target crusty_jit_capi
```

Requires LLVM/Clang 18 dev libraries, Eigen3, and `symforce`, `yourdfpy`, `numpy`.

## Docker

```
docker build -t crab .
docker run --rm -it -v "$PWD":/workspace crab
# inside the container, then the build and run steps above
```

## Try it

```
python -m examples.demo
python -m tests.test_kernel_correctness
python -m benchmarks.bench_kernels
```

## Layout

Front end (`crab_*`: robot description to C++ source):
- `crab_model/`, `crab_io/`, `crab_kinematics/`: robot description, URDF loading, symbolic FK.
- `crab_codegen/`: SymForce to C++ source (including the fused collision kernel) and to Caspar CUDA kernels.

Back end (`crusty_*`: generated source to callable machine code):
- `crusty_jit/`: CPU JIT engine and Python binder.
- `crusty_compilations/`: Clang/ORC JIT (C++).
- `crusty_cuda/`: GPU path: compiles and calls Caspar CUDA kernels (optional; needs nvcc, torch, SymForce with Caspar).

- `robots/panda/`, `robots/rby1/`: spherized URDFs/SRDFs (Panda also has meshes for visualization).

## GPU path (optional)

Uses SymForce's Caspar, which lives in the SymForce repo (`required_dirs/symforce`, build it
with `pip install --no-build-isolation -e required_dirs/symforce`). Set
`CRAB_PRECISION=float32` before importing `crab_model`. Then:

```
python -m tests.test_caspar_correctness
```
