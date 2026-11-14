"""Path C (GPU): the CUDA counterpart of crab_codegen.codegen. Turns any
`(robot, q, *extra_inputs) -> sf.Matrix` symbolic function into a Caspar kernel library
(SymForce's CUDA kernel generator, symforce.caspar): one CUDA thread per configuration.

- q (nq values per thread) is read with ReadSequential, in chunks, because SymForce's
  vector types stop at V9 (a 23-DOF robot becomes chunks 9+9+5).
- extra_inputs (e.g. a goal pose) are ReadUnique: the same value for every thread.
  Each must be <= MAX_VEC long.
- The output vector is written with WriteSequential, also chunked.

q is in CRAB's q order (Joint.id), exactly as for the CPU JIT. float32: set
CRAB_PRECISION=float32 before importing crab_model (see crab_model/__init__.py).

This module only generates; compiling and calling live in crusty_cuda.
"""

import re
from typing import Callable, Optional

import symforce.symbolic as sf
from symforce import typing as T
from symforce.caspar import CasparLibrary
from symforce.caspar import memory as mem

from crab_model.robot import Robot

MAX_VEC = 9


def chunk_sizes(n: int, max_chunk: int = MAX_VEC) -> list[int]:
    sizes = []
    while n > 0:
        sizes.append(min(max_chunk, n))
        n -= sizes[-1]
    if len(sizes) > 1 and sizes[-1] == 1:  # avoid a size-1 tail (no V1 guarantee)
        sizes[-2] -= 1
        sizes[-1] = 2
    return sizes


def _vec(n: int):
    return getattr(sf, f"V{n}")


def build_caspar_library(robot: Robot, symbolic_fn: Callable, name: str,
                         extra_inputs: Optional[list[tuple[str, int]]] = None) -> dict:
    """Returns {"caslib", "name", "lib_name", "nq", "n_out", "q_chunks", "extra_inputs",
    "out_chunks"}; crusty_cuda.build_cuda_function compiles and binds it."""
    extra_inputs = extra_inputs or []
    for ename, size in extra_inputs:
        if size > MAX_VEC:
            raise ValueError(f"extra input '{ename}' has size {size} > {MAX_VEC}")

    nq = robot.nq
    probe = symbolic_fn(robot, sf.Matrix(nq, 1).symbolic("q"),
                        *[sf.Matrix(size, 1).symbolic(ename) for ename, size in extra_inputs])
    n_out = probe.shape[0]

    q_chunks = chunk_sizes(nq)
    out_chunks = chunk_sizes(n_out)
    lib_name = "caspar_" + re.sub(r"\W", "_", name)
    caslib = CasparLibrary(name=lib_name)

    def kernel(**args):
        q = sf.Matrix([x for i in range(len(q_chunks)) for x in args[f"q{i}"].to_flat_list()])
        extras = [args[f"extra_{ename}"] for ename, _ in extra_inputs]
        flat = symbolic_fn(robot, q, *extras).to_flat_list()
        outs, col = [], 0
        for c in out_chunks:
            outs.append(_vec(c)(*flat[col:col + c]))
            col += c
        return outs[0] if len(outs) == 1 else tuple(outs)

    annotations = {f"q{i}": T.Annotated[_vec(c), mem.ReadSequential] for i, c in enumerate(q_chunks)}
    for ename, size in extra_inputs:
        annotations[f"extra_{ename}"] = T.Annotated[_vec(size), mem.ReadUnique]
    out_types = [T.Annotated[_vec(c), mem.WriteSequential] for c in out_chunks]
    annotations["return"] = out_types[0] if len(out_types) == 1 else T.Tuple[tuple(out_types)]

    kernel.__name__ = name
    kernel.__annotations__ = annotations  # the number of chunks varies per robot
    caslib.add_kernel(kernel)

    return {"caslib": caslib, "name": name, "lib_name": lib_name, "nq": nq, "n_out": n_out,
            "q_chunks": q_chunks, "extra_inputs": extra_inputs, "out_chunks": out_chunks}
