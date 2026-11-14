"""crocket (CUDA trace compiler) FK of one link. Thread and warp tiers.
Run in the crab env (needs nvcc, numpy, sympy, beautifulsoup4):
    python baseline_comparisons/fk_crocket.py [--robot panda|rby1]

crocket parses URDFs with GRiD's URDFParser; pyroffi's GRiM checkout contains one.
Timing is crocket's own harness protocol: CUDA events around back-to-back launches of
the generated kernel (inputs already on the GPU). That is GPU kernel time per launch, NOT
the synchronized wall-clock per call the other scripts report, so the two agree at large
batch but crocket looks better at batch 1 (no launch/sync overhead included).
"""

import os
import sys
import time

import numpy as np

import common

GRID = common.REPO / "external/pyroffi/external/GRiM/external/GRiD"
os.environ.setdefault("CROCKET_GRID_PATH", str(GRID))
sys.path.insert(0, str(common.REPO / "external/crocket/python"))

import crocket
from crocket import harness
from crocket.cuda import Variant

REPS = 50
RUNS = 5


def rot_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    t = np.trace(R)
    if t > 0:
        s = 2.0 * np.sqrt(t + 1.0)
        return np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = 2.0 * np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k])
    q = np.zeros(4)
    q[1 + i] = 0.25 * s
    q[0] = (R[k, j] - R[j, k]) / s
    q[1 + j] = (R[j, i] + R[i, j]) / s
    q[1 + k] = (R[k, i] + R[i, k]) / s
    return q


def main() -> None:
    spec = common.parse_robot()

    t0 = time.perf_counter()
    robot = crocket.Robot(spec.urdf_path, name=spec.name)
    program = crocket.trace(robot, "frame_poses", frames=(spec.link,))
    tiers = {"crocket_thread": Variant("thread"), "crocket_warp": Variant("warp", 32)}
    built = harness.build([(program, v) for v in tiers.values()], dtype=np.float32)
    build_seconds = time.perf_counter() - t0  # parse + trace + nvcc (cached on disk after the first run)
    names = list(robot.joint_names)
    print(f"crocket n_q={robot.n_q} build {build_seconds:.1f}s")

    def inputs(q: np.ndarray) -> dict:
        return {"q": common.embed(spec, q, names).astype(np.float32)}

    for i, library in enumerate(tiers):
        per_config = {}
        for b in common.BATCH_SIZES:
            x = inputs(common.sample_configs(spec, b, seed=b))
            ms = float(np.median([built.run(i, x, reps=REPS)[1] for _ in range(RUNS)]))
            per_config[b] = ms * 1000.0 / b
        latency = per_config[1]

        out, _ = built.run(i, inputs(common.accuracy_configs(spec)))
        res = out["poses"]  # (N, 12): translation, then the 3x3 rotation column-major
        poses = [common.to_pose_row(o[:3], rot_to_quat_wxyz(o[3:12].reshape(3, 3, order="F"))) for o in res]

        common.save(spec, library, "gpu", "float32", latency, per_config, poses,
                    note="crocket harness: GPU kernel time per launch (events, back-to-back launches, "
                         "no sync/launch overhead; compare at large batch)",
                    build_seconds=build_seconds)


if __name__ == "__main__":
    main()
