# SHRIMP image: SHRIMP + CRAB + the planners it is compared against (prrax, VAMP).
#
#   docker build -t shrimp .
#   docker run --gpus all -it --rm -v "$PWD":/src shrimp
#   bash setup.sh                       # once per container: builds the CRAB JIT lib, installs prrax/pyroffi, VAMP
#
# Source is NOT copied in. The repo is mounted at /src and put on PYTHONPATH (SHRIMP's `src`, CRAB's `crab_*`),
# so SHRIMP and CRAB are never pip-installed: edit on the host, rerun, nothing to reinstall.
#
# Two environments, since SymForce/LLVM (CRAB) and JAX (prrax) stacks conflict:
#   /opt/venvs/shrimp   SHRIMP + CRAB (SymForce, system LLVM 18 + clang) + the VAMP Python module
#   /opt/venvs/prrax    prrax + pyroffi (JAX, CUDA 13)
ARG CUDA_IMAGE=nvidia/cuda:13.0.0-devel-ubuntu24.04
FROM ${CUDA_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive

# Build toolchain, LLVM/Clang 18 dev libs (crusty_compilations links LLVM and clang-cpp), Eigen (generated sources
# include <Eigen/Dense>), GMP (SymForce's symengine). nvcc comes from the base image.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake ninja-build git curl ca-certificates \
        clang-18 llvm-18-dev libclang-18-dev libclang-cpp18-dev libeigen3-dev \
        python3 python3-pip python3-venv python3-dev libgmp-dev \
    && rm -rf /var/lib/apt/lists/*

# CRAB's JIT looks up a binary named exactly `clang`.
RUN ln -s /usr/bin/clang-18 /usr/local/bin/clang

# --- SHRIMP + CRAB. nanobind / scikit-build-core are for building the VAMP module (setup.sh). ---
RUN python3 -m venv /opt/venvs/shrimp && \
    /opt/venvs/shrimp/bin/pip install --no-cache-dir numpy scipy matplotlib symforce yourdfpy pytest \
        pybind11 nanobind scikit-build-core fire

# --- prrax + pyroffi (the jax[cuda13] extra pulls the NVIDIA pip libs incl. cuDNN; prrax and pyroffi themselves
#     are installed from the mounted external/ checkouts by setup.sh) ---
RUN python3 -m venv /opt/venvs/prrax && \
    /opt/venvs/prrax/bin/pip install --no-cache-dir "jax[cuda13]==0.10.2" jaxlib==0.10.2 \
        numpy scipy matplotlib yourdfpy pynvml pytest

# The mounted repo is on the path: `from src.rrtc import ...` (SHRIMP), `import crab_codegen` (CRAB).
ENV PYTHONPATH=/src:/src/crab
ENV PATH=/opt/venvs/shrimp/bin:$PATH
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
RUN git config --global --add safe.directory '*'
WORKDIR /src
CMD ["/bin/bash"]
