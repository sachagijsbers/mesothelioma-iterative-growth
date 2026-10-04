"""Controlled experiment: same tumour, same prescribed growth, two mechanical environments.

    A  uniform material properties everywhere (E = 10 kPa, nu = 0.45)     results/homogeneous_control
    B  CT-derived properties E(HU), nu(HU)                                 results/ct_derived_stiffness

Downloads the CT if needed, runs both simulations, writes results/metrics_stiffness_control.csv
and the three portfolio figures in figures/portfolio/.

    python -m experiments.run_stiffness_control                    # full run, about 1.5 h
    python -m experiments.run_stiffness_control --skip-simulation  # metrics + figures from existing results
"""

import argparse
from pathlib import Path

import numpy as np

from meso_growth import data, figures, iterative
from meso_growth.metrics import check_same_start, load_run, metrics, write_csv

ROOT = Path(__file__).resolve().parents[1]
RUNS = {  # name -> (folder, homogeneous)
    "uniform": ("homogeneous_control", True),
    "ct": ("ct_derived_stiffness", False),
}
LABELS = {"uniform": "Uniform material (control)", "ct": "CT-derived stiffness"}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", default=data.DEFAULT_CASE)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--figures-dir", type=Path, default=ROOT / "figures" / "portfolio")
    parser.add_argument("--growth-factor", type=float, default=3.0, help="final volume / starting volume")
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--skip-simulation", action="store_true", help="reuse existing results")
    args = parser.parse_args()

    data.fetch(args.case, args.data_dir)
    case = data.setup_case(args.case, args.data_dir)
    if not args.skip_simulation:
        for name, (folder, homogeneous) in RUNS.items():
            print(f"== {LABELS[name]}", flush=True)
            iterative.run(case, args.results_dir / folder, homogeneous, args.growth_factor, args.steps)

    runs = {name: load_run(args.results_dir / folder) for name, (folder, _) in RUNS.items()}
    check_same_start(runs)
    results = {name: metrics(phis, case.lung, case.any_lung, case.normal) for name, phis in runs.items()}
    write_csv(results, args.results_dir / "metrics_stiffness_control.csv")
    for name, row in results.items():
        print(f"{LABELS[name]:28s} lung share {100 * row['lung_share']:5.1f}%   added {row['added_voxels'] / 1000:.2f} ml"
              f"   lost {row['lost_voxels']} voxels")

    args.figures_dir.mkdir(parents=True, exist_ok=True)
    ct_dir = args.results_dir / RUNS["ct"][0]
    figures.growth_overview(ct_dir, args.figures_dir / "growth_overview.png")
    figures.comparison(dict(np.load(ct_dir / "context.npz")), runs, results, LABELS,
                       {"uniform": figures.COLOURS["uniform"], "ct": figures.COLOURS["ct"]},
                       "Same start, same prescribed growth,\ndifferent mechanics",
                       args.figures_dir / "stiffness_control.png")
    figures.stiffness_map(ct_dir, case.any_lung, args.figures_dir / "stiffness_map.png")
    print(f"figures in {args.figures_dir}")


if __name__ == "__main__":
    main()
