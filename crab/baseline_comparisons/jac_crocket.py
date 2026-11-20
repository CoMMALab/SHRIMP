"""crocket frame_jacobian (CUDA trace compiler): thread and warp tiers.
Run in the crocket env: python baseline_comparisons/jac_crocket.py [--robot panda|rby1]

crocket's kernel returns the pose and the full 6 x n_q world-aligned Jacobian (v; w rows);
we take the linear rows (= d position / dq). It therefore computes strictly MORE than the
CRAB/Caspar position Jacobian (rotation rows + pose), which handicaps crocket slightly.
Timing is crocket's own harness (CUDA events over 50 back-to-back launches)."""

import os
import sys
import time

import numpy as np

import common

GRID = common.ROOT / "external/pyroffi/external/GRiM/external/GRiD"
os.environ.setdefault("CROCKET_GRID_PATH", str(GRID))
sys.path.insert(0, str(common.ROOT / "external/crocket/python"))

import crocket
from crocket import harness
from crocket.cuda import Variant

REPS = 50
RUNS = 5


def main() -> None:
    spec = common.parse_robot()

    t0 = time.perf_counter()
    robot = crocket.Robot(spec.urdf_path, name=spec.name)
    program = crocket.trace(robot, "frame_jacobian", frame=spec.link)
    tiers = {"crocket_thread": Variant("thread"), "crocket_warp": Variant("warp", 32)}
    built = harness.build([(program, v) for v in tiers.values()], dtype=np.float32)
    build_seconds = time.perf_counter() - t0
    names = list(robot.joint_names)
    column_of = [names.index(n) for n in spec.joints]
    print(f"crocket n_q={robot.n_q} build {build_seconds:.1f}s")

    def inputs(q: np.ndarray) -> dict:
        return {"q": common.embed(spec, q, names).astype(np.float32)}

    for i, library in enumerate(tiers):
        per_config = {}
        for b in common.BATCH_SIZES:
            x = inputs(common.sample_configs(spec, b, seed=b))
            ms = float(np.median([built.run(i, x, reps=REPS)[1] for _ in range(RUNS)]))
            per_config[b] = ms * 1000.0 / b

        out, _ = built.run(i, inputs(common.accuracy_configs(spec)))
        jac = out["jacobian"].reshape(-1, 6, robot.n_q)[:, :3, :]  # linear rows
        rows = common.jac_to_spec_order(jac, column_of)

        common.save(spec, library, "gpu", "float32", None, per_config, rows.tolist(),
                    note="crocket frame_jacobian (also computes rotation rows + pose); CUDA events "
                         "over 50 back-to-back launches",
                    build_seconds=build_seconds, task="jac")


if __name__ == "__main__":
    main()
