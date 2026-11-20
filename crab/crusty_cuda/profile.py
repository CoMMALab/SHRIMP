"""Kernel-only GPU timing, via the PyTorch profiler (CUPTI). Needs only torch.

Calling a kernel from Python costs host time (pybind / torch dispatch) that dwarfs the kernel at
small and medium batch. The profiler reports GPU durations, so that overhead drops out and the
result is comparable to a C++ launch loop.
"""

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile


def _device_us(event) -> float:
    total = getattr(event, "device_time_total", None)
    return float(total if total is not None else event.cuda_time_total)


def kernel_time_us(fn, kernel_name: str = None, n_launches: int = 20, n_runs: int = 5) -> float:
    """Median over n_runs of the GPU time (us) per call of fn(), summed over the kernels it runs.
    With kernel_name, only kernels whose name contains it count (case and underscores ignored:
    Caspar CamelCases names, ee_jac_x -> caspar::EeJacXKernel); with None, every GPU activity the
    call causes counts (multi-kernel libraries like cuRobo)."""
    fn()
    torch.cuda.synchronize()
    wanted = kernel_name.replace("_", "").lower() if kernel_name else None
    samples = []
    for _ in range(n_runs):
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(n_launches):
                fn()
            torch.cuda.synchronize()
        events = prof.key_averages()
        if wanted is None:
            matched = [e for e in events if _device_us(e) > 0]
        else:
            matched = [e for e in events if wanted in e.key.replace("_", "").lower()]
        if not matched:
            raise RuntimeError(f"no profiled GPU activity matches {kernel_name!r}; saw: "
                               f"{[e.key for e in events][:10]}")
        samples.append(sum(_device_us(e) for e in matched) / n_launches)
    return float(np.median(samples))
