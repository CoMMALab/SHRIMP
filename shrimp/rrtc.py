"""SHRIMP's GPU RRT-Connect: assembles one CUDA translation unit from CRAB's robot header
(crab_codegen/robot_header.py: N, LO, HI, World, fkcc) + the planner (cuda/rrtc.cuh) + the file-driven
benchmark host program (cuda/rrtc_main.cuh; argv documented there).

    from crab_codegen.robot_header import build_robot_header
    from shrimp.rrtc import build_rrtc_source
    src = build_rrtc_source(build_robot_header(robot, q_index, lo, hi, pair_i, pair_j))

Compile with: nvcc -O3 -std=c++17 --expt-relaxed-constexpr -I<eigen3> (ik_benchmark/cuda_host.compile_cu).
SHRIMP depends on CRAB, never the other way round.
"""

from pathlib import Path

CUDA = Path(__file__).parent / "cuda"


def build_rrtc_source(robot_header: str, with_main: bool = True) -> str:
    parts = [robot_header, (CUDA / "rrtc.cuh").read_text()]
    if with_main:
        parts.append((CUDA / "rrtc_main.cuh").read_text())
    return "\n".join(parts)
