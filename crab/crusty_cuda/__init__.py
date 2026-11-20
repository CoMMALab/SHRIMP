"""GPU back end (Path C). Imports are lazy so that `crusty_cuda.profile` (torch only) works in
environments without SymForce/Caspar, e.g. the baseline envs that reuse the same timer."""

__all__ = ["CudaFunction", "build_cuda_function", "kernel_time_us"]


def __getattr__(name):
    if name == "CudaFunction":
        from .function import CudaFunction
        return CudaFunction
    if name == "build_cuda_function":
        from .simple import build_cuda_function
        return build_cuda_function
    if name == "kernel_time_us":
        from .profile import kernel_time_us
        return kernel_time_us
    raise AttributeError(f"module 'crusty_cuda' has no attribute {name!r}")
