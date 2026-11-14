"""Print a comparison table from results/*.json (plain python+numpy; run in any env).
Accuracy is measured against CRAB (float64) as the reference."""

import json

import numpy as np

import common


def main() -> None:
    results = {p.stem: json.loads(p.read_text()) for p in sorted(common.RESULTS.glob("*.json"))}
    if not results:
        raise SystemExit("no results/*.json yet; run the fk_*.py scripts first")
    ref = np.array(results["crab"]["accuracy_poses"]) if "crab" in results else None

    batches = common.BATCH_SIZES
    print(f"FK of {common.LINK}; per-config time in us (batch size across)\n")
    header = f"{'library':<14}{'device':<6}{'prec':<9}" + "".join(f"{'b=' + str(b):>11}" for b in batches)
    header += f"{'pos err (m)':>13}{'rot err (rad)':>15}"
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
        print(f"{name:<14}{r['device']:<6}{r['precision']:<9}{cols}{pos_err:>13.2e}{rot_err:>15.2e}")

    print("\nnotes:")
    for name, r in results.items():
        print(f"  {name}: {r['note']}")


if __name__ == "__main__":
    main()
