# Baseline comparison image: CRAB vs cuRobo vs pyroffi vs crocket.
# Isolated environments, since their CUDA/LLVM/torch/jax stacks conflict:
#   /opt/venvs/crab     CRAB (system LLVM 18 + clang)
#   /opt/venvs/curobo   cuRobo (PyTorch, CUDA 13)
#   /opt/venvs/pyroffi  pyroffi (JAX, CUDA 13)
#   /opt/venvs/crocket  crocket (pure Python + nvcc)
#
# Build from the repo root:  docker build -f baseline_comparisons/Dockerfile -t crab-baselines .
# Source is NOT copied in; mount the repo wherever you like and run baseline_comparisons/setup.sh.
ARG CUDA_IMAGE=nvidia/cuda:13.0.0-devel-ubuntu24.04
FROM ${CUDA_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake ninja-build git curl ca-certificates \
        clang-18 llvm-18-dev libclang-18-dev libclang-cpp18-dev libeigen3-dev \
        python3 python3-pip python3-venv python3-dev libgmp-dev \
    && rm -rf /var/lib/apt/lists/*

# CRAB's JIT looks up a binary named exactly `clang`.
RUN ln -s /usr/bin/clang-18 /usr/local/bin/clang

# --- CRAB (CPU JIT + the GPU path, crusty_cuda). PyPI symforce is a fallback for the CPU path;
#     setup.sh replaces it with the clone in required_dirs/symforce, which includes Caspar.
#     torch + build deps are for the GPU path; nvcc comes from the base image. ---
RUN python3 -m venv /opt/venvs/crab && \
    /opt/venvs/crab/bin/pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu130 && \
    /opt/venvs/crab/bin/pip install --no-cache-dir numpy symforce yourdfpy pytest pybind11 \
        "setuptools>=64" wheel setuptools_scm "cython>=3,<3.1.0"

# --- cuRobo (torch from the CUDA 13 wheel index; curobo itself installed by setup.sh) ---
RUN python3 -m venv /opt/venvs/curobo && \
    /opt/venvs/curobo/bin/pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu130 && \
    /opt/venvs/curobo/bin/pip install --no-cache-dir numpy pytest

# --- pyroffi (the jax[cuda13] extra pulls the NVIDIA pip libs incl. cuDNN, which the bare
#     plugin packages in pyroffi's `cuda` extra do not; pyroffi itself installed by setup.sh) ---
RUN python3 -m venv /opt/venvs/pyroffi && \
    /opt/venvs/pyroffi/bin/pip install --no-cache-dir \
        "jax[cuda13]==0.10.2" jaxlib==0.10.2 pynvml pytest

# --- crocket (pure Python, imported from external/crocket; sympy + beautifulsoup4 are for
#     GRiD's URDFParser; it compiles with the base image's nvcc) ---
RUN python3 -m venv /opt/venvs/crocket && \
    /opt/venvs/crocket/bin/pip install --no-cache-dir numpy sympy beautifulsoup4 lxml pytest

ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
RUN git config --global --add safe.directory '*'
CMD ["/bin/bash"]
