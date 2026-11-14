#!/usr/bin/env bash
# Run once inside the container (repo mounted at /workspace):
#   bash baseline_comparisons/setup.sh
# Installs/builds each project from the mounted source. Re-runnable.
set -euo pipefail
cd /workspace

echo "== CRAB: build the JIT library"
cmake -S crusty_compilations -B crusty_compilations/build -G Ninja
cmake --build crusty_compilations/build --target crusty_jit_capi

echo "== cuRobo"
/opt/venvs/curobo/bin/pip install -e "external/curobo[cu13]"

echo "== pyroffi (needs its GRiM submodule for CUDA FK)"
if [ -z "$(ls -A external/pyroffi/external/GRiM 2>/dev/null)" ]; then
  git -C external/pyroffi submodule update --init --recursive
fi
/opt/venvs/pyroffi/bin/pip install -e "external/pyroffi[cuda]"

echo "== cricket (build + python bindings, inside its conda env)"
micromamba run -n cricket bash -c '
  set -e
  cd /workspace/external/cricket
  cmake -GNinja -Bbuild .
  cmake --build build
  pip install --no-build-isolation -e .
'

echo "Done. Activate an env with:"
echo "  source /opt/venvs/crab/bin/activate      # CRAB"
echo "  source /opt/venvs/curobo/bin/activate    # cuRobo"
echo "  source /opt/venvs/pyroffi/bin/activate   # pyroffi"
echo "  micromamba activate cricket              # cricket"
