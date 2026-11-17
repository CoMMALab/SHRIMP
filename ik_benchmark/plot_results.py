"""Plots of the MBM results, one figure per robot: ik_benchmark/results/plots/{panda,baxter}.png
    python ik_benchmark/plot_results.py
Reads results/<chain>_{crab_cpu,crab_rrtc,prrax}.npz (missing files are skipped). Panels:
  1 solved % per family       2 median planning ms per family (alone, log)
  3 iterations CDF (solved)   4 batched time per problem (us, log; CPU, GPU only)
Times are kernel/planning only (prrax: its kernel_ms; no dispatch or FFI).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RES = Path(__file__).resolve().parent / "results"
ROBOTS = {"panda": "mbm", "baxter": "mbm_baxter"}
VAMP_MEDIAN_ITERS = {"baxter": 3413}                       # VAMP, range 0.5 (pasted summary)
SRC = {"CPU": ("crab_cpu", "tab:blue"), "GPU": ("crab_rrtc", "tab:green"), "prrax": ("prrax", "tab:orange")}


def load(chain):
    out = {}
    for name, (suffix, _) in SRC.items():
        f = RES / f"{chain}_{suffix}.npz"
        if f.exists():
            out[name] = np.load(f, allow_pickle=True)
    return out


def alone_ms(name, d, P):
    """Per-problem alone time (ms) as (problem indices, values), or None."""
    if name == "CPU":
        return np.arange(P), d["plan_ms"].astype(float)
    key = "kernel_ms" if name == "prrax" else "lat_ms"      # prrax: its kernel time, not wall (no dispatch)
    n = len(d[key])
    return np.arange(n) * P // n, d[key].astype(float)


def main():
    out_dir = RES / "plots"
    out_dir.mkdir(exist_ok=True)
    for robot, chain in ROBOTS.items():
        data = load(chain)
        if not data:
            continue
        ref = next(iter(data.values()))
        names = [str(n) for n in ref["category_names"]]
        cat = ref["category"]
        P = len(cat)
        fig, ax = plt.subplots(2, 2, figsize=(14, 9))
        fig.suptitle(f"{robot}: MBM, {P} problems")
        w = 0.8 / len(data)
        x = np.arange(len(names))

        for k, (name, d) in enumerate(data.items()):                      # 1: solved %
            col = SRC[name][1]
            s = np.asarray(d["solved"]).astype(float)
            ax[0, 0].bar(x + k * w, [100 * s[cat == c].mean() if (cat == c).any() else 0 for c in range(len(names))],
                         w, label=name, color=col)
        ax[0, 0].set_ylabel("solved %")
        ax[0, 0].set_ylim(0, 105)

        for k, (name, d) in enumerate(data.items()):                      # 2: median alone ms per family
            idx, ms = alone_ms(name, d, P)
            med = []
            for c in range(len(names)):
                m = (cat[idx] == c) & np.isfinite(ms)
                if name == "CPU":
                    m &= np.asarray(d["solved"])[idx].astype(bool)
                med.append(np.median(ms[m]) if m.any() else np.nan)
            ax[0, 1].bar(x + k * w, med, w, label=name, color=SRC[name][1])
        ax[0, 1].set_yscale("log")
        ax[0, 1].set_ylabel("median planning ms (alone)")
        for a in (ax[0, 0], ax[0, 1]):
            a.set_xticks(x + 0.4 - w / 2)
            a.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
            a.legend()

        for name in ("CPU", "GPU"):                                       # 3: iterations CDF
            if name not in data:
                continue
            d = data[name]
            it = np.asarray(d["iters"])[np.asarray(d["solved"]).astype(bool)]
            if len(it):
                it = np.sort(it)
                ax[1, 0].plot(it, np.arange(1, len(it) + 1) / len(it), label=name, color=SRC[name][1])
        if robot in VAMP_MEDIAN_ITERS:
            ax[1, 0].axvline(VAMP_MEDIAN_ITERS[robot], color="k", ls="--", label="VAMP median")
        ax[1, 0].set_xscale("log")
        ax[1, 0].set_xlabel("iterations (solved)")
        ax[1, 0].set_ylabel("fraction")
        ax[1, 0].legend()

        us, lab, cols = [], [], []                                        # 4: batched us / problem
        for name, d in data.items():
            if name == "prrax":          # only wall time is batched (includes dispatch): left out
                continue
            if name == "CPU":
                us.append(float(d["wall_ms"]) / P * 1e3 if "wall_ms" in d and np.ndim(d["wall_ms"]) == 0
                          else float(np.sum(d["wall_ms"])) / P * 1e3)
                lab.append(f"CPU ({int(d['threads'])} thr)")
            else:
                us.append(float(d["batch_ms"]) / P * 1e3)
                lab.append(name)
            cols.append(SRC[name][1])
        ax[1, 1].bar(lab, us, color=cols)
        ax[1, 1].set_yscale("log")
        ax[1, 1].set_ylabel("batched us / problem")
        for i, v in enumerate(us):
            ax[1, 1].text(i, v, f"{v:.0f}", ha="center", va="bottom")

        fig.tight_layout()
        fig.savefig(out_dir / f"{robot}.png", dpi=130)
        plt.close(fig)
        print(f"wrote {out_dir / f'{robot}.png'}")


if __name__ == "__main__":
    main()
