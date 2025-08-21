
import csv, argparse, importlib, numpy as np
m = importlib.import_module("aic")  # assumes aic.py in same folder

def phase_metrics(eq, criticality_param: float, steps: int, rng: np.random.Generator):
    """Return (superposition_entropy, MI, MI_baseline_shuffledB)."""
    eps = 1e-12
    activity = eq._run_simulation(criticality_param, steps=steps)
    final_activity = activity[-1]

    # superposition-like entropy
    hist, _ = np.histogram(final_activity, bins=10, range=(-1, 1))
    p = hist / max(hist.sum(), eps)
    sup_entropy = -np.sum(p[p > 0] * np.log2(p[p > 0]))

    # pairwise MI between halves
    half = eq.n_neurons // 2
    A, B = final_activity[:half], final_activity[half:]

    def _mi(x, y):
        joint, _, _ = np.histogram2d(x, y, bins=4, range=[[-1, 1], [-1, 1]])
        joint /= max(joint.sum(), eps)
        pX = joint.sum(axis=1); pY = joint.sum(axis=0)
        H_X = -np.sum(pX[pX > 0] * np.log2(pX[pX > 0]))
        H_Y = -np.sum(pY[pY > 0] * np.log2(pY[pY > 0]))
        H_XY = -np.sum(joint[joint > 0] * np.log2(joint[joint > 0]))
        return H_X + H_Y - H_XY

    mi = _mi(A, B)

    # Baseline: shuffle B to break structure, recompute MI
    B_shuf = B[rng.permutation(B.shape[0])]
    mi_baseline = _mi(A, B_shuf)
    return sup_entropy, mi, mi_baseline

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="sweep_results.csv")
    ap.add_argument("--seeds", type=int, nargs="*", default=[101, 202, 303])
    ap.add_argument("--neurons", type=int, nargs="*", default=[120, 200, 256])
    ap.add_argument("--steps", type=int, nargs="*", default=[50, 100])
    ap.add_argument("--params", type=float, nargs="*", default=[0.6,0.8,1.0,1.2,1.5,2.0,2.5])
    args = ap.parse_args()

    rows = []
    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        np.random.seed(seed)

        for N in args.neurons:
            eq = m.EmergentQuantumness(n_neurons=N, quiet=True)
            init = np.random.rand(eq.n_neurons)

            # Phase param sweep
            for steps in args.steps:
                for crit in args.params:
                    eq.population = init.copy()
                    sup, mi, mi_base = phase_metrics(eq, crit, steps, rng)
                    rows.append({
                        "seed": seed, "neurons": N, "steps": steps,
                        "criticality": crit, "metric": "phase",
                        "superposition_entropy": sup, "mutual_info": mi,
                        "mutual_info_sub": "", "mutual_info_crit": ""
                    })
                    rows.append({
                        "seed": seed, "neurons": N, "steps": steps,
                        "criticality": crit, "metric": "baseline",
                        "superposition_entropy": "", "mutual_info": mi_base,
                        "mutual_info_sub": "", "mutual_info_crit": ""
                    })

            # Binding comparison (subcritical vs critical) on 120-neuron setting
            cbm = m.ConsciousnessBindingMechanism(n_neurons=120, quiet=True)
            for steps in args.steps:
                metrics, _ = cbm.unified_experience_from_distributed_processing(steps=steps)
                rows.append({
                    "seed": seed, "neurons": 120, "steps": steps,
                    "criticality": "sub/crit", "metric": "binding",
                    "superposition_entropy": "", "mutual_info": "",
                    "mutual_info_sub": metrics["subcritical_binding"],
                    "mutual_info_crit": metrics["critical_binding"]
                })

    fieldnames = ["seed","neurons","steps","criticality","metric",
                  "superposition_entropy","mutual_info","mutual_info_sub","mutual_info_crit"]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader(); w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.out}")

if __name__ == "__main__":
    main()
