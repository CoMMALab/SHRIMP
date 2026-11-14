"""Path A: the generic path for one-off `(q, *extra_inputs) -> flat vector` symbolic
functions. Codegen, extern "C" wrapping, JIT compiling, and binding in a single call,
instead of hand-writing a new template/builder/export for every new function (see
JIT_ARCHITECTURE.md).

Not used for the fused collision kernel (Path B, crusty_codegen.collision_kernel) --
that has a genuinely bespoke shape (hand-fused control flow across multiple generated
functions). Both paths bind through the same crusty_jit.binder.JitFunction, though, so
there's exactly one binding mechanism either way.

Usage (single input, q):

    def q_to_ee_pose(robot, q) -> sf.Matrix:
        ...  # the actual math -- the only part that should take real effort
        return sf.Matrix(...)

    kernel = build_simple_jit_function(robot, q_to_ee_pose, name="q_to_ee")
    kernel(q)  # -> np.ndarray, shape inferred from q_to_ee_pose's own return shape

Usage (extra runtime inputs beyond q):

    def task_space_distance(robot, q, goal_pose_flat) -> sf.Matrix:
        ...
        return sf.Matrix([...])

    kernel = build_simple_jit_function(robot, task_space_distance, name="task_space_distance",
                                        extra_inputs=[("goal_pose", 7)])
    kernel(q, goal_pose)  # positional, in extra_inputs' declared order after q
"""

from typing import Callable, Optional

from crusty_codegen.codegen import build_and_generate_function
from crusty_codegen.fixtures import load_panda_robot
from crusty_codegen.jit_wrapper import generate_value_wrapper

from .binder import JitFunction, bind_jit_function
from .engine import CrabJitEngine


def build_simple_jit_function(robot=None,
                               symbolic_fn: Callable = None,
                               name: str = None,
                               extra_inputs: Optional[list[tuple[str, int]]] = None,
                               engine: Optional[CrabJitEngine] = None,
                               module_id: Optional[str] = None) -> JitFunction:
    """symbolic_fn(robot, q, *extra_values) -> sf.Matrix (a flat column vector).
    extra_inputs declares any inputs beyond q as (name, size) pairs -- see
    crusty_codegen.codegen.build_and_generate_function. Everything else --
    Codegen.function, reading back SymForce's real emitted symbol name, the extern "C"
    wrapper, compiling at -O3, and the ctypes binding -- is automatic regardless of
    how many inputs there are."""
    robot = robot or load_panda_robot()
    generated = build_and_generate_function(robot, symbolic_fn, name, extra_inputs=extra_inputs)

    wrapper_name = f"jit_{name}"
    source = "\n".join([
        "#include <Eigen/Dense>",
        "",
        generated["source"],
        "",
        generate_value_wrapper(generated["func_name"], generated["input_sizes"], generated["n_out"], wrapper_name),
    ])

    engine = engine or CrabJitEngine()
    module_id = module_id or f"crab_jit_simple_{name}_{robot.name}_{generated['nq']}dof"
    engine.compile(source, module_id)

    return bind_jit_function(engine, wrapper_name,
                              input_sizes=generated["input_sizes"], output_sizes=[generated["n_out"]])
