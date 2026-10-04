"""Re-implementation of the MSc thesis mechanics, wrapped in the same iterative growth loop.

This is NOT the original thesis code (DolfinX, see the mesothelioma-growth-model repository).
It re-implements the thesis formulation in scikit-fem so it can run on the same public CT and
the same synthetic tumour as the iterative model:

    rho = max((HU + 1000) / 1000, 0)        density, local 5 mm mean
    p   = k * max(rho - rho0, 0)             pressure
    f   = p * d                              d: interior node -> nearest boundary node,
                                             boundary node -> nearest interior node
    -div sigma(u) = f,   sigma = lam tr(eps) I + 2 mu eps
    u = 0 at nodes with rho >= 2 and at the density-weighted centre of mass

Only the tumour is meshed; the surrounding tissue enters only through the density in the load.
Additions compared with the thesis: the prescribed-volume loop, the level-set update, a weak
elastic foundation and removal of rigid-body motion. On the synthetic tumour no node reaches
rho >= 2, so it is held at the single centre-of-mass node.

The outward normal displacement u.n sets the local growth speed. The tumour is advanced on the
1 mm level-set grid until the prescribed volume is reached, re-meshed with GMSH and burned into
the density map before the next solve.
"""

import csv
from pathlib import Path

import gmsh
import numpy as np
import trimesh
from scipy import ndimage
from scipy.spatial import cKDTree
from skfem import Basis, BilinearForm, ElementTetP1, ElementVector, LinearForm, MeshTet, asm, condense, solve
from skfem.helpers import ddot, dot, sym_grad, trace
from skimage.measure import marching_cubes

from .geometry import GRID_MM, TUMOUR_HU, advance, hu_to_density, signed_distance, volume_of

# Thesis parameters. Because the problem is linear and the growth amount is prescribed,
# E and k only scale u; the growth pattern depends on rho, nu, geometry and the BCs.
E = 1.0e5
NU = 0.3
K = 5000.0
RHO0 = 1.0
RHO_FIXED = 2.0
# Weak elastic foundation that removes the rigid rotations left by the point BC.
SPRING = 1.0e-4 * E / 30.0**2
MESH_SIZE_MM = 1.8


def tetrahedral_mesh(phi, roi, stl_path):
    """Marching cubes on phi -> smoothed watertight surface -> GMSH tetrahedral mesh."""
    vertices, faces, _, _ = marching_cubes(phi, level=0.0, spacing=(GRID_MM,) * 3)
    surface = trimesh.Trimesh(vertices + roi.origin, faces, process=True)
    trimesh.smoothing.filter_taubin(surface, iterations=10)
    surface.fix_normals()
    if not surface.is_watertight:
        raise ValueError("tumour surface is not watertight")
    surface.export(stl_path)

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.merge(str(stl_path))
        # The smoothed marching-cubes triangulation is used as-is as the boundary mesh;
        # reparametrising thin pleural shapes gives overlapping patches.
        loop = gmsh.model.geo.addSurfaceLoop([tag for _, tag in gmsh.model.getEntities(2)])
        gmsh.model.geo.addVolume([loop])
        gmsh.model.geo.synchronize()
        gmsh.option.setNumber("Mesh.MeshSizeMin", 0.6 * MESH_SIZE_MM)
        gmsh.option.setNumber("Mesh.MeshSizeMax", MESH_SIZE_MM)
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.model.mesh.generate(3)
        node_tags, coordinates, _ = gmsh.model.mesh.getNodes()
        _, _, element_nodes = gmsh.model.mesh.getElements(3)
    finally:
        gmsh.finalize()

    order = np.argsort(node_tags)
    points = coordinates.reshape(-1, 3)[order]
    tets = np.searchsorted(node_tags[order], element_nodes[0].reshape(-1, 4))
    used = np.unique(tets)
    points, tets = points[used], np.searchsorted(used, tets)
    signed = tet_volumes(points, tets)
    tets[signed < 0] = tets[signed < 0][:, [0, 2, 1, 3]]
    return points, tets, surface


def tet_volumes(points, tets):
    a, b, c, d = (points[tets[:, i]] for i in range(4))
    return np.einsum("ij,ij->i", np.cross(b - a, c - a), d - a) / 6.0


def thesis_directions(points, boundary):
    """Interior nodes point to the nearest boundary node; boundary nodes point inward."""
    interior = np.setdiff1d(np.arange(len(points)), boundary)
    directions = points[boundary][cKDTree(points[boundary]).query(points)[1]] - points
    directions[boundary] = points[interior][cKDTree(points[interior]).query(points[boundary])[1]] - points[boundary]
    norm = np.linalg.norm(directions, axis=1, keepdims=True)
    return np.divide(directions, norm, out=np.zeros_like(directions), where=norm > 0)


def solve_mechanics(points, tets, rho):
    mesh = MeshTet(np.ascontiguousarray(points.T), np.ascontiguousarray(tets.T))
    basis = Basis(mesh, ElementVector(ElementTetP1()))
    mu = E / (2.0 * (1.0 + NU))
    lam = E * NU / ((1.0 + NU) * (1.0 - 2.0 * NU))

    boundary = mesh.boundary_nodes()
    pressure = K * np.clip(rho - RHO0, 0.0, None)
    force = thesis_directions(points, boundary) * pressure[:, None]
    force_dofs = np.zeros(basis.N)
    for c in range(3):
        force_dofs[basis.nodal_dofs[c]] = force[:, c]

    @BilinearForm
    def stiffness(u, v, w):
        eu, ev = sym_grad(u), sym_grad(v)
        return 2.0 * mu * ddot(eu, ev) + lam * trace(eu) * trace(ev) + SPRING * dot(u, v)

    @LinearForm
    def load(v, w):
        return dot(w.f, v)

    centre = (points * rho[:, None]).sum(axis=0) / rho.sum()
    fixed = np.union1d(np.flatnonzero(rho >= RHO_FIXED), [np.argmin(np.linalg.norm(points - centre, axis=1))])
    solution = solve(*condense(
        asm(stiffness, basis), asm(load, basis, f=basis.interpolate(force_dofs)),
        D=basis.nodal_dofs[:, fixed].ravel(),
    ))
    displacement = np.column_stack([solution[basis.nodal_dofs[c]] for c in range(3)])
    return remove_rigid_motion(points, displacement), pressure, force, boundary, fixed


def remove_rigid_motion(points, displacement):
    """Subtract the best-fit translation + infinitesimal rotation.

    With only point constraints the solve still contains near-rigid rotation, which
    changes neither shape nor volume and must not steer the growth direction.
    """
    x = points - points.mean(axis=0)
    rows = np.zeros((len(x), 3, 6))
    rows[:, :, :3] = np.eye(3)
    # omega x x  written as a matrix acting on omega
    rows[:, 0, 4], rows[:, 0, 5] = x[:, 2], -x[:, 1]
    rows[:, 1, 3], rows[:, 1, 5] = -x[:, 2], x[:, 0]
    rows[:, 2, 3], rows[:, 2, 4] = x[:, 1], -x[:, 0]
    coefficients = np.linalg.lstsq(rows.reshape(-1, 6), displacement.ravel(), rcond=None)[0]
    return displacement - (rows.reshape(-1, 6) @ coefficients).reshape(-1, 3)


def growth_speed(phi, roi, points, boundary, displacement):
    """Outward normal displacement u.n, extended from the surface to nearby grid points."""
    band = phi < 20.0
    normal = np.stack(np.gradient(phi, GRID_MM), axis=-1)[band]
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
    nearest = cKDTree(points[boundary]).query(roi.points[band])[1]
    speed = np.zeros(roi.shape)
    speed[band] = np.clip(np.einsum("ij,ij->i", displacement[boundary][nearest], normal), 0.0, None)
    speed = ndimage.gaussian_filter(speed, 1.5)
    if speed.max() <= 0.0:
        raise RuntimeError("FEM gives no outward displacement anywhere")
    return speed / speed.max()


def run(case, out, growth_factor=3.0, steps=6):
    """Same loop as iterative.run, with the thesis mechanics and a new GMSH mesh every step."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    roi = case.roi
    phi = signed_distance(case.tumour)
    hu_current = case.hu_roi.copy()
    hu_current[case.tumour] = TUMOUR_HU
    np.savez_compressed(out / "context.npz", origin=roi.origin, spacing=GRID_MM, site=case.site,
                        normal=case.normal, hu=case.hu_roi.astype(np.int16))

    v0 = volume_of(phi)
    dv = (growth_factor - 1.0) * v0 / steps
    rows = []
    for step in range(steps + 1):
        points, tets, _ = tetrahedral_mesh(phi, roi, out / f"surface_{step:02d}.stl")
        volumes_tet = tet_volumes(points, tets)
        rho_grid = ndimage.uniform_filter(hu_to_density(hu_current), size=5)
        rho = ndimage.map_coordinates(rho_grid, roi.index(points).T, order=1, mode="nearest")
        displacement, pressure, force, boundary, fixed = solve_mechanics(points, tets, rho)
        np.savez_compressed(
            out / f"step_{step:02d}.npz", points=points, tets=tets, boundary=boundary, rho=rho,
            pressure=pressure, force=force, displacement=displacement, phi=phi.astype(np.float32),
        )

        row = {
            "step": step,
            "volume_ml": volume_of(phi) / 1000.0,
            "mesh_volume_ml": volumes_tet.sum() / 1000.0,
            "nodes": len(points),
            "tets": len(tets),
            "min_tet_volume_mm3": volumes_tet.min(),
            "fixed_nodes": len(fixed),
            "mean_boundary_pressure_pa": pressure[boundary].mean(),
            "max_displacement": np.linalg.norm(displacement, axis=1).max(),
        }
        if step < steps:
            speed = growth_speed(phi, roi, points, boundary, displacement)
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
