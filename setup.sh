#!/usr/bin/env bash
# Run once inside the container (repo mounted at /src), from anywhere:
#   bash /src/setup.sh                        # CRAB JIT library + prrax/pyroffi
#   VAMP_SRC=/path/to/vamp bash /src/setup.sh # ... and the VAMP Python module (mount VAMP and point at it)
# SHRIMP and CRAB are not installed: PYTHONPATH (set in the Dockerfile) points at the mounted source.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

echo "== CRAB: build the JIT library"
cmake -S crab/crusty_compilations -B crab/crusty_compilations/build
cmake --build crab/crusty_compilations/build --target crusty_jit_capi

echo "== pyroffi + prrax (into /opt/venvs/prrax; the checkouts under external/ must be on prrax's expected revision)"
/opt/venvs/prrax/bin/pip install -e external/pyroffi
/opt/venvs/prrax/bin/pip install -e external/prrax

if [ -n "${VAMP_SRC:-}" ]; then
  echo "== VAMP Python module (into /opt/venvs/shrimp): ${VAMP_SRC}"
  /opt/venvs/shrimp/bin/pip install --no-build-isolation "${VAMP_SRC}" \
      -C cmake.define.VAMP_ROBOTS="panda;baxter"
else
  echo "== VAMP skipped (set VAMP_SRC to a VAMP checkout to build its Python module)"
fi

echo "Done. Default env is /opt/venvs/shrimp (SHRIMP, CRAB, VAMP); for prrax:"
echo "  /opt/venvs/prrax/bin/python ik_benchmark/solve_prrax.py --chain mbm_baxter"
