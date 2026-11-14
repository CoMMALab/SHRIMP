FROM ubuntu:noble

ENV DEBIAN_FRONTEND=noninteractive

# Build toolchain + LLVM/Clang 18 dev libs (crusty_compilations links LLVM and clang-cpp)
# + Eigen (JIT'd sources include <Eigen/Dense>).
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake ninja-build git ca-certificates \
        clang-18 llvm-18-dev libclang-18-dev libclang-cpp18-dev \
        libeigen3-dev \
        python3 python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/*

# crusty::jit looks up a binary named exactly `clang` on PATH.
RUN ln -s /usr/bin/clang-18 /usr/local/bin/clang

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /workspace
CMD ["/bin/bash"]
