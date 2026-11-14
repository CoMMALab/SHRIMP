"""Path C entry point (GPU): the counterpart of crusty_jit.simple.build_simple_jit_function.

    os.environ["CRAB_PRECISION"] = "float32"   # before importing crab_model
    kernel = build_cuda_function(robot, q_to_ee_pose, name="q_to_ee")
    poses = kernel(q_batch)                    # (N, nq) -> (N, 7) float32 CUDA tensor

Needs: nvcc + cmake, torch with CUDA, and SymForce with caspar (see README).
Generated sources and the compiled library are kept under
$XDG_CACHE_HOME/crusty/cuda/<name>_<robot>_<nq>dof (delete it for a cold build).
"""

import os
from pathlib import Path
from typing import Callable, Optional

from crab_codegen.caspar_kernel import build_caspar_library
from crab_codegen.fixtures import load_panda_robot

from .function import CudaFunction


def default_cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "crusty" / "cuda"


def build_cuda_function(robot=None,
                        symbolic_fn: Callable = None,
                        name: str = None,
                        extra_inputs: Optional[list[tuple[str, int]]] = None,
                        cache_dir: Optional[Path] = None) -> CudaFunction:
    robot = robot or load_panda_robot()
    spec = build_caspar_library(robot, symbolic_fn, name, extra_inputs=extra_inputs)

    out_dir = Path(cache_dir or default_cache_dir()) / f"{name}_{robot.name}_{robot.nq}dof"
    caslib = spec["caslib"]
    caslib.generate(out_dir)
    caslib.compile(out_dir)  # incremental: cmake rebuilds only what changed
    lib = caslib.import_lib(out_dir)
    return CudaFunction(lib, spec)
