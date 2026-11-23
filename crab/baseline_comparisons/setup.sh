#!/usr/bin/env bash
# Run once inside the container, from anywhere:
#   bash <repo>/baseline_comparisons/setup.sh
# Installs/builds each project from the mounted source. Re-runnable.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="$(cd "$REPO/.." && pwd)"      # SHRIMP root: external/ lives here, CRAB is crab/
cd "$REPO"

echo "== CRAB: build the JIT library"
cmake -S crusty_compilations -B crusty_compilations/build
cmake --build crusty_compilations/build --target crusty_jit_capi

echo "== cuRobo"
/opt/venvs/curobo/bin/pip install -e "$ROOT/external/curobo[cu13]"

echo "== pyroffi (needs its GRiM submodule for CUDA FK)"
if [ -z "$(ls -A "$ROOT/external/pyroffi/external/GRiM" 2>/dev/null)" ]; then
  git -C "$ROOT/external/pyroffi" submodule update --init --recursive
fi
/opt/venvs/pyroffi/bin/pip install -e "$ROOT/external/pyroffi[cuda]"

echo "== Caspar: build SymForce (with caspar) from the clone in required_dirs/symforce"
# Caspar is part of CRAB (crusty_cuda), so it goes in the crab env, replacing the PyPI symforce
# with the clone (a superset that includes caspar). --no-build-isolation builds with what's in
# the venv, so the build deps must be installed first.
# (cython is pinned by SymForce's setup.py: newer releases break symenginepy's wrapper.)
/opt/venvs/crab/bin/pip install "setuptools>=64" wheel setuptools_scm "cython>=3,<3.1.0" numpy pybind11
/opt/venvs/crab/bin/pip uninstall -y symforce
/opt/venvs/crab/bin/pip install --no-build-isolation -e required_dirs/symforce

echo "== crocket (pure Python, imported from external/crocket; deps are for GRiD's URDFParser)"
/opt/venvs/crocket/bin/pip install numpy sympy beautifulsoup4 lxml

echo "Done. Activate an env with:"
echo "  source /opt/venvs/crab/bin/activate      # CRAB (CPU JIT + GPU/Caspar)"
echo "  source /opt/venvs/curobo/bin/activate    # cuRobo"
echo "  source /opt/venvs/pyroffi/bin/activate   # pyroffi"
echo "  source /opt/venvs/crocket/bin/activate   # crocket"
