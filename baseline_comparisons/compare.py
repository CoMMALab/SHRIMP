"""Print a comparison table from results/<robot>/*.json (plain python+numpy; any env).
Accuracy is measured against CRAB (float64) as the reference.
Usage: python baseline_comparisons/compare.py [--robot panda|rby1]"""

import json

import numpy as np

import common


def main() -> None:
    spec = common.parse_robot()
    folder = common.RESULTS / spec.name
    results = {p.stem: json.loads(p.read_text()) for p in sorted(folder.glob("*.json"))}
    if not results:
        raise SystemExit(f"no {folder}/*.json yet; run the fk_*.py scripts with --robot {spec.name}")
    ref = np.array(results["crab"]["accuracy_poses"]) if "crab" in results else None

    batches = common.BATCH_SIZES
    print(f"{spec.name} ({spec.dof} DOF): FK of {spec.link}; per-config time in us (batch size across)\n")
    header = f"{'library':<16}{'device':<6}{'prec':<9}" + "".join(f"{'b=' + str(b):>11}" for b in batches)
    header += f"{'pos err (m)':>13}{'rot err (rad)':>15}{'build (s)':>11}"
    print(header)
    for name, r in results.items():
        cols = ""
        for b in batches:
            v = r["per_config_us_by_batch"].get(str(b))
            cols += f"{v:>11.3f}" if v is not None else f"{'-':>11}"
        pos_err = rot_err = float("nan")
        if ref is not None:
            p = np.array(r["accuracy_poses"])
            pos_err = float(np.max(np.linalg.norm(p[:, :3] - ref[:, :3], axis=1)))
            dots = np.clip(np.abs(np.sum(p[:, 3:] * ref[:, 3:], axis=1)), 0.0, 1.0)
            rot_err = float(np.max(2.0 * np.arccos(dots)))
        build = f"{r['build_seconds']:.1f}" if r.get("build_seconds") is not None else "-"
        print(f"{name:<16}{r['device']:<6}{r['precision']:<9}{cols}{pos_err:>13.2e}{rot_err:>15.2e}{build:>11}")

    print("\nnotes:")
    for name, r in results.items():
        print(f"  {name}: {r['note']}")
    print("\nbuild (s) for CRAB is codegen + JIT compile, and is near zero when the disk cache is warm.")


if __name__ == "__main__":
    main()
