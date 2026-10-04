"""Iterative model: pleural tumour growth as eigenstrain in CT-derived host tissue.

Setting: public thorax CT, synthetic pleural tumour, prescribed volume increase per step.

  * the FEM domain is the tumour AND its surroundings (region of interest), so lung,
    soft tissue and ribs resist the growth with their own stiffness;
  * stiffness per element from CT:  E(HU) piecewise log-linear (lung soft, bone stiff);
  * growth is an isotropic eigenstrain inside the tumour:
        sigma = C : (eps(u) - eps_g),   eps_g = (beta / 3) * chi * I
    with chi the tumour volume fraction of the element;
  * u = 0 on the outer ROI boundary (far field), no rigid-body modes;
  * mechanical pressure p = -tr(sigma) / 3 (compression positive).

Where the tumour grows follows the outward normal displacement u.n of its surface;
how much it grows per step stays prescribed (V_{n+1} = V_n + dV).

The host mesh is a FIXED structured tetrahedral grid; it is not remeshed. The tumour lives on
the 1 mm level-set grid and enters the FEM only through the volume fraction chi per element.
"""

import csv
import time
from pathlib import Path

import numpy as np
import pyamg
from scipy import ndimage
from scipy.sparse.linalg import cg
from scipy.spatial import cKDTree
from skfem import Basis, BilinearForm, ElementTetP0, ElementTetP1, ElementVector, LinearForm, MeshTet, asm, condense
from skfem.helpers import ddot, sym_grad, trace
from skimage.measure import marching_cubes

from .geometry import GRID_MM, TUMOUR_HU, advance, hu_to_density, signed_distance, to_voxel, volume_of

FEM_MM = 3.0  # host mesh spacing
# Nominal growth strain per step. The problem is linear and the speed field is normalised,
# so beta only scales u: the growth pattern does not depend on it.
BETA = 0.05

# Stiffness assumption (kPa) versus HU: lung parenchyma ~1-3, soft tissue ~10-20, bone >> soft.
# Literature order of magnitude, not measured for this patient.
HU_POINTS = np.array([-1000.0, -700.0, -300.0, 0.0, 100.0, 300.0, 1000.0])
E_POINTS_KPA = np.array([1.0, 2.0, 5.0, 10.0, 20.0, 5.0e2, 1.0e3])

# Control run: one material everywhere.
HOMOGENEOUS_E_KPA = 10.0
HOMOGENEOUS_NU = 0.45


def stiffness_from_hu(hu):
    """Young's modulus in kPa, piecewise linear in log10(E) between the HU_POINTS."""
    return 10.0 ** np.interp(hu, HU_POINTS, np.log10(E_POINTS_KPA))


def poisson_from_hu(hu):
    return np.where(hu < -300.0, 0.3, 0.45)  # compressible lung, nearly incompressible soft tissue


def host_mesh(roi):
    n = int(round((roi.shape[0] - 1) * GRID_MM / FEM_MM)) + 1
    axes = [roi.origin[a] + FEM_MM * np.arange(n) for a in range(3)]
    return MeshTet.init_tensor(*axes)


def nodal_grid(mesh, roi, values):
    """Reshape nodal values of the tensor-product host mesh to an (nx, ny, nz, ...) array."""
    index = np.round((mesh.p.T - roi.origin) / FEM_MM).astype(int)
    grid = np.zeros(tuple(index.max(axis=0) + 1) + values.shape[1:])
    grid[tuple(index.T)] = values
    return grid


def sample_grid(grid, roi, points, order=1):
    return ndimage.map_coordinates(grid, roi.index(points).T, order=order, mode="nearest")


def rigid_body_modes(points, basis):
    """Near-nullspace of elasticity (3 translations, 3 rotations) for algebraic multigrid."""
    x = points - points.mean(axis=0)
    modes = np.zeros((basis.N, 6))
    for c in range(3):
        modes[basis.nodal_dofs[c], c] = 1.0
    for k, (i, j) in enumerate([(0, 1), (1, 2), (2, 0)]):
        modes[basis.nodal_dofs[i], 3 + k] = -x[:, j]
        modes[basis.nodal_dofs[j], 3 + k] = x[:, i]
    return modes


def solve_growth(mesh, roi, hu, phi, homogeneous=False, verbose=True):
    """Solve linear elasticity with eigenstrain in the tumour; returns u at the nodes and element fields."""
    # P1 strains and P0 material are constant per element, so one quadrature point is exact.
    basis = Basis(mesh, ElementVector(ElementTetP1()), intorder=1)
    element_basis = basis.with_element(ElementTetP0())
    centroids = mesh.p[:, mesh.t].mean(axis=1).T
    hu_element = sample_grid(ndimage.uniform_filter(hu, 3), roi, centroids)
    chi = np.clip(0.5 - sample_grid(phi, roi, centroids) / FEM_MM, 0.0, 1.0)
    young = stiffness_from_hu(hu_element)
    nu = poisson_from_hu(hu_element)
    if homogeneous:
        # With a uniform E the eigenstrain load scales with E too, so u does not depend on
        # its value; only nu matters (soft-tissue value).
        young = np.full_like(young, HOMOGENEOUS_E_KPA)
        nu = np.full_like(nu, HOMOGENEOUS_NU)
    mu = young / (2.0 * (1.0 + nu))
    lam = young * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    growth = BETA / 3.0 * chi

    fields = {name: element_basis.interpolate(value) for name, value in
              [("mu", mu), ("lam", lam), ("g", growth)]}

    @BilinearForm
    def stiffness(u, v, w):
        eu, ev = sym_grad(u), sym_grad(v)
        return 2.0 * w.mu * ddot(eu, ev) + w.lam * trace(eu) * trace(ev)

    @LinearForm
    def load(v, w):
        # C : eps_g : eps(v) with eps_g = g I  ->  (3 lam + 2 mu) g tr(eps(v))
        return (3.0 * w.lam + 2.0 * w.mu) * w.g * trace(sym_grad(v))

    started = time.perf_counter()
    A = asm(stiffness, basis, **fields)
    b = asm(load, basis, **fields)
    A_free, b_free, u, free = condense(A, b, D=basis.get_dofs().all())
    solver = pyamg.smoothed_aggregation_solver(
        A_free.tocsr(), B=rigid_body_modes(mesh.p.T, basis)[free], max_coarse=500)
    u[free], info = cg(A_free, b_free, M=solver.aspreconditioner(), rtol=1e-8, maxiter=500)
    if info != 0:
        raise RuntimeError(f"CG did not converge (info={info})")
    if verbose:
        print(f"  solve: {basis.N} dofs in {time.perf_counter() - started:.1f} s", flush=True)

    displacement = np.column_stack([u[basis.nodal_dofs[c]] for c in range(3)])
    # Element strain from P1 displacement: gradient is constant per tetrahedron.
    gradient = basis.interpolate(u).grad.mean(axis=-1)  # (3, 3, n_elements)
    volumetric = np.trace(gradient, axis1=0, axis2=1)
    pressure = -((3.0 * lam + 2.0 * mu) * (volumetric / 3.0 - growth))  # -tr(sigma)/3, kPa
    return displacement, pressure, centroids, chi, young


def body_context(hu, labels, affine, site, step_mm=3.0):
    """Whole-thorax context for the figures: lung and bone surfaces, full axial CT slice at the site."""
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    factor = np.maximum(np.round(step_mm / spacing).astype(int), 1)
    coarse = tuple(slice(None, None, f) for f in factor)
    coarse_affine = affine.copy()
    coarse_affine[:3, :3] = affine[:3, :3] * factor

    def surface(mask, sigma):
        vertices, faces, _, _ = marching_cubes(ndimage.gaussian_filter(mask.astype(np.float32), sigma), 0.5)
        return vertices @ coarse_affine[:3, :3].T + coarse_affine[:3, 3], faces

    lungs_v, lungs_f = surface(labels[coarse] > 0, 1.0)
    # Max-pool before subsampling, otherwise thin ribs fall between the samples.
    bone = ndimage.maximum_filter(hu, size=tuple(factor))[coarse] > 250.0
    # Only bone around the lungs (rib cage, spine, sternum), not the CT table or the arms.
    bone &= ndimage.distance_transform_edt(labels[coarse] == 0, sampling=spacing * factor) < 35.0
    labelled, count = ndimage.label(bone)
    sizes = ndimage.sum(bone, labelled, np.arange(1, count + 1))
    bone = np.isin(labelled, 1 + np.flatnonzero(sizes > 60))  # ribs, spine, sternum; no calcified specks
    bone_v, bone_f = surface(bone, 0.7)

    # Axial slice through the site on a 1 mm world grid covering the body.
    corners = np.array([[i, j, 0] for i in (0, hu.shape[0] - 1) for j in (0, hu.shape[1] - 1)]) @ affine[:3, :3].T + affine[:3, 3]
    lo, hi = corners[:, :2].min(axis=0), corners[:, :2].max(axis=0)
    xs, ys = np.arange(lo[0], hi[0], 1.0), np.arange(lo[1], hi[1], 1.0)
    grid = np.stack(np.meshgrid(xs, ys, [site[2]], indexing="ij"), axis=-1).reshape(-1, 3)
    axial = ndimage.map_coordinates(hu, to_voxel(grid, affine), order=1, cval=-1000.0).reshape(len(xs), len(ys))
    return dict(body_lungs_vertices=lungs_v, body_lungs_faces=lungs_f, body_bone_vertices=bone_v, body_bone_faces=bone_f,
                axial_hu=axial.astype(np.int16), axial_origin=lo)


def run(case, out, homogeneous=False, growth_factor=3.0, steps=6):
    """Grow the starting tumour of `case` to growth_factor x its volume in `steps` steps.

    Writes context.npz, step_XX.npz (phi and fields per step) and steps.csv to `out`.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    roi = case.roi
    phi = signed_distance(case.tumour)
    hu_current = case.hu_roi.copy()
    hu_current[case.tumour] = TUMOUR_HU

    lung_vertices, lung_faces, _, _ = marching_cubes(
        ndimage.gaussian_filter(case.lung.astype(np.float32), 1.0), 0.5, spacing=(GRID_MM,) * 3)
    np.savez_compressed(out / "context.npz", lung_vertices=lung_vertices + roi.origin, lung_faces=lung_faces,
                        origin=roi.origin, spacing=GRID_MM, site=case.site, normal=case.normal,
                        hu=case.hu_roi.astype(np.int16),
                        **body_context(case.hu, case.labels, case.affine, case.site))

    mesh = host_mesh(roi)
    print(f"host mesh: {mesh.p.shape[1]} nodes, {mesh.t.shape[1]} tets", flush=True)
    v0 = volume_of(phi)
    dv = (growth_factor - 1.0) * v0 / steps
    rows = []
    for step in range(steps + 1):
        displacement, pressure, centroids, chi, young = solve_growth(mesh, roi, hu_current, phi, homogeneous)

        # Nodal displacement lives on a regular grid, so map it to the 1 mm grid directly.
        band = np.abs(phi) < 12.0
        grid_points = roi.points[band]
        u_grid = nodal_grid(mesh, roi, displacement)
        u_band = np.column_stack([
            ndimage.map_coordinates(u_grid[..., c], ((grid_points - roi.origin) / FEM_MM).T, order=1, mode="nearest")
            for c in range(3)
        ])

        normals = np.stack(np.gradient(phi, GRID_MM), axis=-1)[band]
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
        speed_band = np.clip(np.einsum("ij,ij->i", u_band, normals), 0.0, None)
        speed = np.zeros(roi.shape)
        speed[band] = speed_band
        speed = ndimage.gaussian_filter(speed, 1.5)
        speed /= speed.max()

        # Tumour surface + pressure sampled from the element field for the renderer.
        verts, _, _, _ = marching_cubes(phi, 0.0, spacing=(GRID_MM,) * 3)
        surface_points = verts + roi.origin
        surface_elements = mesh.element_finder()(*surface_points.T)
        surface_displacement = u_band[cKDTree(grid_points).query(surface_points)[1]]

        shell = (phi > 1.0) & (phi < 10.0)
        shell_points = roi.points[shell][::7]
        shell_u = u_band[cKDTree(grid_points).query(shell_points)[1]]

        rho_grid = ndimage.uniform_filter(hu_to_density(hu_current), size=5)
        np.savez_compressed(
            out / f"step_{step:02d}.npz", phi=phi.astype(np.float32), density_grid=rho_grid.astype(np.float32),
            surface_points=surface_points, surface_pressure=pressure[surface_elements],
            surface_displacement=surface_displacement, arrow_points=shell_points, arrow_vectors=shell_u,
            young_grid=np.log10(stiffness_from_hu(ndimage.uniform_filter(hu_current, 3))).astype(np.float32),
        )

        tumour_elements = chi > 0.5
        row = {
            "step": step,
            "volume_ml": volume_of(phi) / 1000.0,
            "mean_tumour_pressure_kpa": float(pressure[tumour_elements].mean()),
            "max_tumour_pressure_kpa": float(pressure[tumour_elements].max()),
            "max_displacement_mm": float(np.linalg.norm(displacement, axis=1).max()),
        }
        if step < steps:
            new_phi, advance_mm = advance(phi, speed, v0 + (step + 1) * dv)
            added = (new_phi < 0) & ~(phi < 0)
            row["max_advance_mm"] = advance_mm
            row["added_into_lung_fraction"] = (added & case.lung).sum() / max(added.sum(), 1)
            row["added_into_wall_fraction"] = (added & ~case.any_lung).sum() / max(added.sum(), 1)
            phi = new_phi
            hu_current[phi < 0] = TUMOUR_HU
        rows.append(row)
        print("  ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)

    with open(out / "steps.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows
