"""Shared helpers for the FK comparison scripts (stdlib + numpy only).

Every script times FK of the Panda's `panda_hand` link and writes
results/<library>.json; compare.py reads them all. Same joint configs everywhere.
"""

import json
import pathlib
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
RESULTS = HERE / "results"

LINK = "panda_hand"
PANDA_LO = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
PANDA_HI = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])

BATCH_SIZES = [1, 16, 256, 4096, 65536]
N_ACCURACY = 64


def sample_configs(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(PANDA_LO, PANDA_HI, size=(n, 7))


def accuracy_configs() -> np.ndarray:
    return sample_configs(N_ACCURACY, seed=123)


def to_pose_row(position, quat_wxyz) -> list[float]:
    """[x, y, z, qw, qx, qy, qz] with the quaternion sign canonicalized so qw >= 0."""
    quat = np.asarray(quat_wxyz, dtype=np.float64)
    if quat[0] < 0:
        quat = -quat
    return [float(v) for v in (*np.asarray(position, dtype=np.float64), *quat)]


def median_us(fn, n_calls: int, n_warmup: int = 20) -> float:
    """Median wall-clock microseconds per call. fn must block until its result is ready."""
    for _ in range(n_warmup):
        fn()
    samples = []
    for _ in range(n_calls):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return float(np.median(samples)) * 1e6


def save(library: str, device: str, precision: str, latency_us: float,
         per_config_us_by_batch: dict, accuracy_poses: list, note: str = "") -> None:
    RESULTS.mkdir(exist_ok=True)
    record = {
        "library": library,
        "device": device,
        "precision": precision,
        "link": LINK,
        "latency_us_batch1": latency_us,
        "per_config_us_by_batch": {str(k): v for k, v in per_config_us_by_batch.items()},
        "accuracy_poses": accuracy_poses,
        "note": note,
    }
    (RESULTS / f"{library}.json").write_text(json.dumps(record, indent=1))
    print(f"[{library}] batch-1 latency {latency_us:.2f} us; wrote {RESULTS / (library + '.json')}")
