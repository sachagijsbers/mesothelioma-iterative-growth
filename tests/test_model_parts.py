"""Material mapping and the directional-growth metric."""

import numpy as np

from meso_growth.iterative import E_POINTS_KPA, HU_POINTS, poisson_from_hu, stiffness_from_hu
from meso_growth.metrics import lung_share


def test_stiffness_mapping_hits_points_and_is_monotone():
    assert np.allclose(stiffness_from_hu(HU_POINTS), E_POINTS_KPA)
    hu = np.linspace(-1200, 1200, 1000)
    assert np.all(np.diff(stiffness_from_hu(hu)) >= 0)
    assert stiffness_from_hu(-1200) == E_POINTS_KPA[0]  # clamped outside the table


def test_poisson_ratio():
    assert poisson_from_hu(-800) == 0.3
    assert poisson_from_hu(40) == 0.45


def test_lung_share_counts_added_voxels_only():
    first = np.zeros((10, 10, 10), bool)
    first[2:5, 2:5, 2:5] = True  # starting tumour, partly in lung: must not count
    last = first.copy()
    last[5:8, 2:5, 2:5] = True  # 27 added voxels
    lung = np.zeros_like(first)
    lung[:, :, :3] = True  # 9 of the added voxels (z = 2) lie in lung
    lung[2:5, 2:5, 2:5] = True
    assert lung_share(first, last, lung) == 9 / 27
