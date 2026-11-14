"""Print a comparison table from results/<robot>/ (plain python+numpy; any env).
Usage: python baseline_comparisons/compare.py [--robot panda|rby1] [--task fk|jac]

fk  : files <library>.json; accuracy vs CRAB (float64) pose: position error and rotation angle.
jac : files jac_<library>.json; accuracy vs CRAB: max |J - J_crab| over the 3 x dof position Jacobian.

Columns b=N: amortized time per configuration in NANOSECONDS (GPU libraries: 50 back-to-back
launches, one sync; crocket's own protocol). 'wall b=1' is the synchronized wall-clock of a
single batch-1 call in microseconds (includes Python/launch overhead; blank for crocket)."""

import json

import numpy as np

import common


def rot_err_rad(q: np.ndarray, ref: np.ndarray) -> float:
    """Max rotation angle between quaternion rows (wxyz). Normalizes, aligns signs, and uses the
    chord (4*asin(d/2)), which stays accurate for float32; 2*arccos(dot) loses ~1e-3 rad there."""
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    ref = ref / np.linalg.norm(ref, axis=1, keepdims=True)
    d = np.minimum(np.linalg.norm(q - ref, axis=1), np.linalg.norm(q + ref, axis=1))
    return float(np.max(4.0 * np.arcsin(np.clip(d / 2.0, 0.0, 1.0))))


def load(folder, task: str) -> dict:
    results = {}
    for p in sorted(folder.glob("*.json")):
        is_jac = p.stem.startswith("jac_")
        if (task == "jac") != is_jac:
            continue
        results[p.stem[4:] if is_jac else p.stem] = json.loads(p.read_text())
    return results


def main() -> None:
    args = common.parse_args()
    spec, task = args.spec, args.task
    folder = common.RESULTS / spec.name
    results = load(folder, task)
    if not results:
        raise SystemExit(f"no {task} results in {folder}; run the {task}_*.py scripts with --robot {spec.name}")
    ref = np.array(results["crab"]["accuracy_poses"]) if "crab" in results else None

    batches = common.BATCH_SIZES
    what = f"FK of {spec.link}" if task == "fk" else f"position Jacobian (3 x {spec.dof}) of {spec.link}"
    print(f"{spec.name} ({spec.dof} DOF): {what}; ns per configuration by batch size\n")
    err_cols = f"{'pos err m':>11}{'rot err rad':>13}" if task == "fk" else f"{'max |dJ|':>11}"
    header = f"{'library':<16}{'dev':<5}{'prec':<8}{'wall b=1':>10}"
    header += "".join(f"{'b=' + str(b):>12}" for b in batches) + err_cols + f"{'build s':>9}"
    print(header)
    for name, r in results.items():
        cols = ""
        for b in batches:
            v = r["per_config_us_by_batch"].get(str(b))
            cols += f"{v * 1000.0:>12.3f}" if v is not None else f"{'-':>12}"
        wall = r.get("latency_us_batch1")
        wall_s = f"{wall:.1f}" if wall is not None else "-"
        errs = ""
        if ref is not None:
            p = np.array(r["accuracy_poses"])
            if task == "fk":
                pos_err = float(np.max(np.linalg.norm(p[:, :3] - ref[:, :3], axis=1)))
                errs = f"{pos_err:>11.1e}{rot_err_rad(p[:, 3:], ref[:, 3:]):>13.1e}"
            else:
                errs = f"{float(np.max(np.abs(p - ref))):>11.1e}"
        build = f"{r['build_seconds']:.1f}" if r.get("build_seconds") is not None else "-"
        print(f"{name:<16}{r['device']:<5}{r['precision']:<8}{wall_s:>10}{cols}{errs}{build:>9}")

    print("\nnotes:")
    for name, r in results.items():
        print(f"  {name}: {r['note']}")
    print("\nbuild s is codegen + compile, near zero when the disk cache is warm (delete the cache for a cold build).")


if __name__ == "__main__":
    main()
