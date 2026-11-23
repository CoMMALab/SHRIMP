"""SHRIMP's GPU RRT-Connect: assembles one CUDA translation unit from CRAB's robot header
(crab_codegen/robot_header.py: N, LO, HI, World, fkcc) + the planner (cuda/rrtc.cuh) + the file-driven
benchmark host program (cuda/rrtc_main.cuh; argv documented there).

    from crab_codegen.robot_header import build_robot_header
    from src.rrtc import build_rrtc_source
    src = build_rrtc_source(build_robot_header(robot, q_index, lo, hi, pair_i, pair_j))

Compile with: nvcc -O3 -std=c++17 --expt-relaxed-constexpr -I<eigen3> (ik_benchmark/cuda_host.compile_cu).
SHRIMP depends on CRAB, never the other way round.
"""

from pathlib import Path

CUDA = Path(__file__).parent / "cuda"
CPU = Path(__file__).parent / "cpu"


def build_cpu_rrtc_source(cpu_robot_header: str, with_main: bool = True) -> str:
    """CPU RRT-Connect (cpu/rrtc.hpp) on crab_codegen.robot_header.build_cpu_robot_header's output.
    Compile with: g++ -O3 -march=native -std=c++17 -pthread -I<eigen3> (ik_benchmark/cuda_host.compile_cpp)."""
    parts = ["#define SHRIMP_ENV_SORTED   // World carries sorted obstacle lists + min distances (rrtc_main.hpp)",
             cpu_robot_header, (CPU / "kdtree.hpp").read_text(), (CPU / "rrtc.hpp").read_text()]
    if with_main:
        parts += [(CPU / "host_util.hpp").read_text(), (CPU / "rrtc_main.hpp").read_text()]
    return "\n".join(parts)


def build_cpu_crrtc_source(cpu_robot_header: str, tsr_header: str, with_main: bool = True) -> str:
    """CPU constrained RRT-Connect (cpu/crrtc.hpp): the unconstrained planner's pieces (rrtc.hpp) plus CRAB's TSR
    constraint header (crab_codegen.constraint_header.build_tsr_header(..., gpu=False)), appended after the robot
    header. Same compile line as build_cpu_rrtc_source."""
    parts = ["#define SHRIMP_ENV_SORTED   // World carries sorted obstacle lists + min distances (host_util.hpp)",
             cpu_robot_header, tsr_header, (CPU / "kdtree.hpp").read_text(), (CPU / "rrtc.hpp").read_text(),
             (CPU / "crrtc.hpp").read_text()]
    if with_main:
        parts += [(CPU / "host_util.hpp").read_text(), (CPU / "crrtc_main.hpp").read_text()]
    return "\n".join(parts)


def build_crrtc_source(robot_header: str, tsr_header: str, with_main: bool = True) -> str:
    """GPU constrained RRT-Connect (cuda/crrtc.cuh) on CRAB's GPU robot header and TSR header
    (crab_codegen.constraint_header.build_tsr_header(..., gpu=True)), plus the unconstrained planner's pieces
    (rrtc.cuh). Same compile line as build_rrtc_source."""
    parts = ["#define SHRIMP_ENV_SORTED   // World carries sorted obstacle lists + min distances (host_util.hpp)",
             robot_header, tsr_header, (CUDA / "rrtc.cuh").read_text(), (CUDA / "crrtc.cuh").read_text()]
    if with_main:
        parts += [(CPU / "host_util.hpp").read_text(), (CUDA / "crrtc_main.cuh").read_text()]
    return "\n".join(parts)


def build_rrtc_source(robot_header: str, with_main: bool = True) -> str:
    parts = ["#define SHRIMP_ENV_SORTED   // World carries sorted obstacle lists + min distances (rrtc_main.cuh)",
             robot_header, (CUDA / "rrtc.cuh").read_text()]
    if with_main:
        parts.append((CUDA / "rrtc_main.cuh").read_text())
    return "\n".join(parts)
