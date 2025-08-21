#!/usr/bin/env python3
import json, csv, argparse, numpy as np
import matplotlib.pyplot as plt
import itertools
from collections import defaultdict

def plot_json(json_path: str, out_prefix="plots"):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    pt = data["phase_transition"]  # dict of regimes
    regimes = list(pt.keys())
    se = [pt[k]["superposition_entropy"] for k in regimes]
    mi = [pt[k]["entanglement_mutual_info"] for k in regimes]

    # Phase-transition bars
    plt.figure(figsize=(8,4))
    x = np.arange(len(regimes))
    plt.bar(x-0.18, se, width=0.36, label="Superposition entropy")
    plt.bar(x+0.18, mi, width=0.36, label="Mutual info (A|B)")
    plt.xticks(x, regimes)
    plt.ylabel("Bits")
    plt.title("Phase transition metrics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_phase.png", dpi=200)

    # Binding bars (single run)
    b = data["binding"]
    plt.figure(figsize=(5,4))
    plt.bar(["Subcritical","Critical"], [b["subcritical_binding"], b["critical_binding"]])
    plt.ylabel("Average pairwise MI (bits)")
    plt.title("Binding strength")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_binding.png", dpi=200)

    print(f"Saved {out_prefix}_phase.png and {out_prefix}_binding.png")

def _read_csv(csv_path: str):
    with open(csv_path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def plot_csv(csv_path: str, out_prefix="sweep"):
    """legacy line plots (no error bars) per neuron count."""
    rows = _read_csv(csv_path)
    records = [r for r in rows if r["metric"] == "phase" and r["mutual_info"]]
    if not records:
        print("No phase records found in CSV.")
        return

    for r in records:
        r["neurons"] = int(r["neurons"]); r["steps"] = int(r["steps"])
        r["criticality"] = float(r["criticality"]); r["mutual_info"] = float(r["mutual_info"])

    records.sort(key=lambda d: (d["neurons"], d["criticality"]))
    for N, group in itertools.groupby(records, key=lambda d: d["neurons"]):
        g = list(group)
        crit = [d["criticality"] for d in g]
        mi = [d["mutual_info"] for d in g]
        plt.figure(figsize=(6,4))
        plt.plot(crit, mi, marker="o")
        plt.xlabel("Criticality parameter")
        plt.ylabel("Mutual information (bits)")
        plt.title(f"Phase MI vs criticality (neurons={N})")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{out_prefix}_phase_mi_N{N}.png", dpi=200)
    print("Saved MI vs criticality plots per neuron count.")

def plot_csv_errorbars(csv_path: str, out_prefix="sweep"):
    """mean±std across seeds; overlay baseline; also binding error bars."""
    rows = _read_csv(csv_path)

    # ---- Phase: aggregate mean±std per (neurons, steps, criticality)
    phase = [r for r in rows if r["metric"] in ("phase","baseline") and r["mutual_info"]]
    if not phase:
        print("No phase/baseline records found.")
    else:
        for r in phase:
            r["neurons"] = int(r["neurons"]); r["steps"] = int(r["steps"])
            r["criticality"] = float(r["criticality"]); r["mutual_info"] = float(r["mutual_info"])

        # group by (N, steps, metric)
        keys = sorted(set((r["neurons"], r["steps"]) for r in phase))
        for (N, steps) in keys:
            series = {"phase": defaultdict(list), "baseline": defaultdict(list)}
            for r in phase:
                if r["neurons"]==N and r["steps"]==steps:
                    series[r["metric"]][r["criticality"]].append(r["mutual_info"])

            # compute means/stds
            def _stats(d):
                xs = sorted(d.keys())
                mu = [np.mean(d[c]) for c in xs]
                sd = [np.std(d[c], ddof=0) for c in xs]
                return xs, mu, sd

            if series["phase"]:
                x_p, mu_p, sd_p = _stats(series["phase"])
                plt.figure(figsize=(6.8,4.2))
                plt.errorbar(x_p, mu_p, yerr=sd_p, fmt='-o', capsize=3, label="phase")
                if series["baseline"]:
                    x_b, mu_b, sd_b = _stats(series["baseline"])
                    plt.errorbar(x_b, mu_b, yerr=sd_b, fmt='--s', capsize=3, label="baseline (shuffled)")
                plt.xlabel("Criticality parameter")
                plt.ylabel("Mutual information (bits)")
                plt.title(f"Phase MI (mean±std) vs criticality\n(neurons={N}, steps={steps})")
                plt.grid(alpha=0.3); plt.legend()
                plt.tight_layout()
                plt.savefig(f"{out_prefix}_phase_mi_err_N{N}_S{steps}.png", dpi=200)

    # ---- Binding: aggregate mean±std per steps (neurons fixed at 120 in sweep)
    bind = [r for r in rows if r["metric"]=="binding"]
    if bind:
        for r in bind:
            r["steps"] = int(r["steps"])
            r["mutual_info_sub"] = float(r["mutual_info_sub"])
            r["mutual_info_crit"] = float(r["mutual_info_crit"])
        by_steps = defaultdict(list)
        for r in bind:
            by_steps[r["steps"]].append((r["mutual_info_sub"], r["mutual_info_crit"]))

        for steps, vals in by_steps.items():
            subs = [v[0] for v in vals]; crits = [v[1] for v in vals]
            mu = [np.mean(subs), np.mean(crits)]
            sd = [np.std(subs, ddof=0), np.std(crits, ddof=0)]
            plt.figure(figsize=(5.5,4))
            x = np.arange(2)
            plt.bar(x, mu, yerr=sd, capsize=4)
            plt.xticks(x, ["Subcritical","Critical"])
            plt.ylabel("Average pairwise MI (bits)")
            plt.title(f"Binding strength (mean±std)\n(steps={steps}, neurons=120)")
            plt.tight_layout()
            plt.savefig(f"{out_prefix}_binding_err_S{steps}.png", dpi=200)

    print("Saved error-bar plots.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", help="JSON produced by: python aic.py --save-json <file.json>")
    ap.add_argument("--csv", help="CSV produced by: python benchmark_sweep.py --out sweep_results.csv")
    ap.add_argument("--out-prefix", default="plots")
    ap.add_argument("--errorbars", action="store_true", help="If set with --csv, also render mean±std and baseline.")
    args = ap.parse_args()
    if args.json:
        plot_json(args.json, args.out_prefix)
    if args.csv:
        plot_csv(args.csv, args.out_prefix)
        if args.errorbars:
            plot_csv_errorbars(args.csv, args.out_prefix)

if __name__ == "__main__":
    main()
