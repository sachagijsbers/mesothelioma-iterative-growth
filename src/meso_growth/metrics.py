"""Directional-growth metric and sanity checks.

Lung share of the added volume (the headline number):

    A = T_end \\ T_start        voxels (1 mm^3) that are tumour at the end but not at the start
    L = filled right-lung segmentation of the ORIGINAL CT, on the same 1 mm grid
    lung share = |A ∩ L| / |A|

It is a 3D volume fraction of the newly added tumour volume only. The anatomy is not deformed
by the model, so "in lung" means "inside the region that was lung on the original CT".
"""

import csv
from pathlib import Path

import numpy as np

from .geometry import volume_of


def load_run(folder):
    """Level sets phi of every step of a run, or None if the run has no output."""
    paths = sorted(Path(folder).glob("step_*.npz"))
    if not paths:
        return None
    return [np.load(p)["phi"] for p in paths]


def lung_share(first, last, lung):
    added = last & ~first
    return (added & lung).sum() / added.sum()


def metrics(phis, lung, any_lung, normal):
    """Metric plus the sanity numbers reported with it, for one run."""
    first, last = phis[0] < 0, phis[-1] < 0
    added = last & ~first
    centroid = lambda m: np.argwhere(m).mean(axis=0)  # 1 mm grid -> mm
    return {
        "start_ml": volume_of(phis[0]) / 1000,
        "final_ml": volume_of(phis[-1]) / 1000,
        "added_voxels": int(added.sum()),
        "added_in_lung_voxels": int((added & lung).sum()),
        "lung_share": lung_share(first, last, lung),
        "outside_any_lung_share": (added & ~any_lung).sum() / added.sum(),
        "lost_voxels": int((first & ~last).sum()),  # starting tumour voxels no longer tumour
        # centroid shift into the lung (along -normal) in mm
        "centroid_shift_into_lung_mm": float((centroid(last) - centroid(first)) @ -normal),
        "steps": len(phis) - 1,
    }


def write_csv(results, path):
    """One row per run; results maps run name -> metrics() dict."""
    keys = list(next(iter(results.values())))
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["run"] + keys)
        for name, row in results.items():
            writer.writerow([name] + [f"{row[k]:.6g}" for k in keys])


def check_same_start(runs):
    """Raise if the runs do not start from the identical tumour mask."""
    names = list(runs)
    reference = runs[names[0]][0] < 0
    for name in names[1:]:
        if not np.array_equal(runs[name][0] < 0, reference):
            raise ValueError(f"{name}: different starting tumour than {names[0]}")
