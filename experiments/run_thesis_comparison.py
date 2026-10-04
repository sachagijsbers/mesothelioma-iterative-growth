"""Optional comparison: thesis mechanics (re-implemented) vs the iterative model, same tumour and growth.

Runs the thesis re-implementation (results/thesis_reimplementation, remeshed with GMSH every
step, several hours) and compares it with results/ct_derived_stiffness from
run_stiffness_control. Writes results/metrics_thesis_comparison.csv and
figures/results/thesis_vs_iterative.png.

    python -m experiments.run_thesis_comparison [--skip-simulation]
"""

import argparse
from pathlib import Path

import numpy as np

from meso_growth import data, figures, thesis_reimpl
from meso_growth.metrics import check_same_start, load_run, metrics, write_csv

ROOT = Path(__file__).resolve().parents[1]
FOLDERS = {"ct": "ct_derived_stiffness", "thesis": "thesis_reimplementation"}
LABELS = {"ct": "Iterative model, CT-derived stiffness", "thesis": "Thesis mechanics (re-implemented)"}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", default=data.DEFAULT_CASE)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--figures-dir", type=Path, default=ROOT / "figures" / "results")
    parser.add_argument("--growth-factor", type=float, default=3.0)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--skip-simulation", action="store_true", help="reuse existing results")
    args = parser.parse_args()

    data.fetch(args.case, args.data_dir)
    case = data.setup_case(args.case, args.data_dir)
    if load_run(args.results_dir / FOLDERS["ct"]) is None:
        raise SystemExit("run `python -m experiments.run_stiffness_control` first")
    if not args.skip_simulation:
        thesis_reimpl.run(case, args.results_dir / FOLDERS["thesis"], args.growth_factor, args.steps)

    runs = {name: load_run(args.results_dir / folder) for name, folder in FOLDERS.items()}
    check_same_start(runs)
    results = {name: metrics(phis, case.lung, case.any_lung, case.normal) for name, phis in runs.items()}
    write_csv(results, args.results_dir / "metrics_thesis_comparison.csv")
    for name, row in results.items():
        print(f"{LABELS[name]:40s} lung share {100 * row['lung_share']:5.1f}%")

    args.figures_dir.mkdir(parents=True, exist_ok=True)
    figures.comparison(dict(np.load(args.results_dir / FOLDERS["ct"] / "context.npz")), runs, results,
                       {"ct": "Iterative model", "thesis": "Thesis mechanics (re-impl.)"},
                       {"ct": figures.COLOURS["ct"], "thesis": figures.COLOURS["thesis"]},
                       "Same start, same prescribed growth,\ndifferent mechanical assumption",
                       args.figures_dir / "thesis_vs_iterative.png")
    print(f"figure in {args.figures_dir}")


if __name__ == "__main__":
    main()
