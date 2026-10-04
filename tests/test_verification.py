"""Numerical verification against closed-form results."""

import numpy as np
import pytest

from meso_growth.geometry import GRID_MM, Roi, advance, signed_distance, volume_of
from meso_growth.iterative import BETA, HOMOGENEOUS_NU, host_mesh, solve_growth


def test_eshelby_sphere():
    """Spherical inclusion with dilatational eigenstrain eps* in a homogeneous medium.

    Outside the inclusion the exact infinite-medium displacement is radial,
        u_r = (1 + nu) / (3 (1 - nu)) * eps* * a^3 / r^2      (Eshelby 1957),
    with eps* = beta / 3 here. The ROI boundary (u = 0) is 5x the radius away, so the
    infinite-medium solution should hold to within a few percent near the inclusion.
    """
    a, half = 9.0, 45.0
    roi = Roi(np.zeros(3), half_mm=half)
    phi = np.linalg.norm(roi.points, axis=-1) - a
    mesh = host_mesh(roi)
    u, *_ = solve_growth(mesh, roi, np.zeros(roi.shape), phi, homogeneous=True, verbose=False)

    p = mesh.p.T
    r = np.linalg.norm(p, axis=1)
    for radius in (12.0, 15.0):
        shell = np.abs(r - radius) < 1.6
        radial = p[shell] / r[shell, None]
        u_r = np.einsum("ij,ij->i", u[shell], radial)
        exact = (1 + HOMOGENEOUS_NU) / (3 * (1 - HOMOGENEOUS_NU)) * (BETA / 3) * a**3 / r[shell] ** 2
        assert np.min(u_r / np.linalg.norm(u[shell], axis=1)) > 0.99  # direction: radial
        assert abs(np.mean(u_r / exact) - 1.0) < 0.05  # magnitude: within 5 %


def test_volume_of_sphere():
    """The smoothed signed distance pulls curved surfaces slightly inward, so volume_of
    underestimates small convex shapes: -3.9 % at R = 8 mm, -0.7 % at R = 20 mm, -0.1 to -0.3 %
    for the 44-132 ml tumour of the experiment. Checked here at the tumour's scale."""
    roi = Roi(np.zeros(3), half_mm=28.0)
    phi = signed_distance(np.linalg.norm(roi.points, axis=-1) < 20.0)
    assert volume_of(phi) == pytest.approx(4 / 3 * np.pi * 20.0**3, rel=0.01)


def test_advance_hits_target_volume():
    roi = Roi(np.zeros(3), half_mm=20.0)
    phi = signed_distance(np.linalg.norm(roi.points, axis=-1) < 8.0)
    speed = np.ones(roi.shape)
    target = 1.5 * volume_of(phi)
    new_phi, step = advance(phi, speed, target)
    # bisection stops on the first voxel layer that reaches the target
    assert target <= volume_of(new_phi) < target + 4 * np.pi * 12.0**2 * GRID_MM
    assert np.all((phi < 0) <= (new_phi < 0))  # growth only, nothing removed
