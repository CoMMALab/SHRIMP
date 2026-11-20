"""Back end for Path C: calls a compiled Caspar kernel (see crab_codegen.caspar_kernel).
The GPU counterpart of crusty_jit.binder.JitFunction.

Caspar keeps data on the GPU in a structure-of-arrays "caspar layout" (shape
(caspar_size(dim), N)). A pipeline that stays on the GPU should convert once and use
run_caspar() repeatedly; __call__ is the convenient form that converts on every call.
"""

import numpy as np
import torch
from symforce.caspar import memory as mem


def _as_cuda(x, cols: int) -> torch.Tensor:
    t = torch.as_tensor(x, dtype=torch.float32, device="cuda").reshape(-1, cols)
    return t.contiguous()


class CudaFunction:
    def __init__(self, lib, spec: dict):
        self._lib = lib
        self.name = spec["name"]
        self._kernel = getattr(lib, spec["name"])
        self.nq = spec["nq"]
        self.n_out = spec["n_out"]
        self._q_chunks = spec["q_chunks"]
        self._extra = spec["extra_inputs"]  # [(name, size)]
        self._out_chunks = spec["out_chunks"]

    def _to_caspar(self, stacked: torch.Tensor, dim: int) -> torch.Tensor:
        out = torch.empty(mem.caspar_size(dim), stacked.shape[0], device="cuda")
        getattr(self._lib, f"matrix{dim}1_stacked_to_caspar")(stacked, out)
        return out

    def to_caspar(self, q) -> list:
        """q: (N, nq) array/tensor in CRAB q order -> list of caspar-layout chunk tensors."""
        q = _as_cuda(q, self.nq)
        chunks, col = [], 0
        for c in self._q_chunks:
            chunks.append(self._to_caspar(q[:, col:col + c].contiguous(), c))
            col += c
        return chunks

    def extras_to_caspar(self, extras) -> list:
        return [self._to_caspar(_as_cuda(e, size), size) for e, (_, size) in zip(extras, self._extra)]

    def run_caspar(self, q_caspar: list, extras_caspar: list, n: int, out: list = None) -> list:
        """Launch the kernel on caspar-layout tensors. Pass `out` (from a previous call) to reuse
        output buffers. Asynchronous: synchronize before timing or reading."""
        if out is None:
            out = [torch.zeros(mem.caspar_size(c), n, device="cuda") for c in self._out_chunks]
        self._kernel(*q_caspar, *extras_caspar, *out, n)
        return out

    def from_caspar(self, out: list, n: int) -> torch.Tensor:
        """caspar-layout output chunks -> (n, n_out) float32 CUDA tensor."""
        parts = []
        for cas, c in zip(out, self._out_chunks):
            stacked = torch.empty(n, c, device="cuda")
            getattr(self._lib, f"matrix{c}1_caspar_to_stacked")(cas, stacked)
            parts.append(stacked)
        return torch.cat(parts, dim=1)

    def __call__(self, q, *extras) -> torch.Tensor:
        n = _as_cuda(q, self.nq).shape[0]
        out = self.run_caspar(self.to_caspar(q), self.extras_to_caspar(extras), n)
        return self.from_caspar(out, n)
