# quantum-inspired-cognitive-stack
Reproducible quantum-inspired cognitive modeling with criticality sweeps, binding MI, and Φq (Quantum Phi). Includes CLI, tests, JSON/CSV export, plotting, and CI.


# Quantum-Inspired Cognitive Stack

**Short description:** Reproducible quantum-inspired cognitive modeling with criticality sweeps, binding mutual information (MI), and Φq (Quantum Phi). Includes CLI, tests, JSON/CSV export, plotting, and CI.

## Highlights
- **Phase transition metrics** (superposition entropy + MI) vs. a tunable *criticality* parameter
- **Binding strength** via average pairwise MI; higher at criticality
- **Φq table** for canonical 4-qubit states (product, GHZ, W, Bell⊗ancilla)
- **Reproducible CLI**: JSON/CSV outputs, quiet mode, seeds
- **Unit tests** (6/6), **GitHub Actions** CI

## Quickstart
```powershell
# install deps
python -m pip install -U numpy matplotlib pytest

# run unit tests
python -m pytest -q

# single run: export JSON, CSV, and Φq table
python aic.py --quiet --seed 123 --neurons 200 --steps 120 `
  --save-json run_123.json --save-csv run_123.csv --phi-table phi4.csv

# plot the single run
python plot_results.py --json run_123.json --out-prefix run_123

# sweep across criticality/neurons/steps + plots (with error bars & baseline)
python benchmark_sweep.py --out sweep_results.csv
python plot_results.py --csv sweep_results.csv --out-prefix sweep --errorbars
