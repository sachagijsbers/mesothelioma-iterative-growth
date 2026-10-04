"""Portfolio figures. Plotting only: every number shown comes from metrics.py or the run outputs.

growth_overview   CT + 3D view of one run, start and end, with the growth-direction arrows
comparison        axial + coronal CT slices of two runs side by side (control, thesis comparison)
stiffness_map     CT-derived stiffness on the slice through the tumour + the assumed E(HU) curve
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes

from .iterative import HOMOGENEOUS_E_KPA, stiffness_from_hu

DISCLAIMER = ("Exploratory mechanics-driven simulation on a synthetic tumour using a public CT dataset. "
              "Not a clinical prediction.")
CREDIT = "CT: COVID-19-CT-Seg (Ma et al., Zenodo 3757476), CC BY-NC-SA."
INK, MUTED = "#0b0b0b", "#52514e"
GRID, AXIS = "#e5e4df", "#c9c8c2"
# Categorical slots 1-3 of a colour-blind-validated palette, fixed per run.
COLOURS = {"uniform": "#2a78d6", "ct": "#eb6834", "thesis": "#1baf7a"}
TUMOUR, START, BONE, LUNG = "#c0392b", "#f5b041", "#efe6d2", "#a9c8ec"
ARROW_CMAP = "viridis"
UP = np.array([0.0, 0.0, 1.0])
CROP_MM = 110.0
DPI = 270


def footer(figure, y=0.012):
    figure.text(0.5, y + 0.018, DISCLAIMER, ha="center", va="bottom", fontsize=8.5, color=MUTED)
    figure.text(0.5, y, CREDIT, ha="center", va="bottom", fontsize=7.5, color=MUTED)


# ---------------------------------------------------------------- 3D rendering


def polydata(vertices, faces):
    return pv.PolyData(vertices, np.column_stack([np.full(len(faces), 3), faces]).ravel())


def surface_from_phi(phi, origin, spacing):
    # extra smoothing for display only: removes the voxel terraces of the initial mask
    vertices, faces, _, _ = marching_cubes(gaussian_filter(phi, 1.2), 0.0, spacing=(spacing,) * 3)
    return polydata(vertices + origin, faces).smooth_taubin(n_iter=30, pass_band=0.08).compute_normals(auto_orient_normals=True)


def volume_ml(phi):
    return float(np.clip(0.5 - phi, 0, 1).sum()) / 1000.0


def growth_arrows(step, surface, spacing_mm=11.0, factor=22.0):
    """Arrows on the tumour surface along the outward normal, length and colour = growth speed u.n."""
    points = np.asarray(surface.points)
    displacement = step["surface_displacement"][cKDTree(step["surface_points"]).query(points)[1]]
    normals = np.asarray(surface.point_normals)
    speed = np.clip(np.einsum("ij,ij->i", displacement, normals), 0.0, None)
    speed /= speed.max()
    # one arrow per spacing_mm cell, the fastest point in each cell
    cells = np.floor(points / spacing_mm).astype(int)
    order = np.argsort(-speed)
    _, first = np.unique(cells[order], axis=0, return_index=True)
    keep = order[first]
    keep = keep[speed[keep] > 0.25]
    arrows = pv.PolyData(points[keep] + 0.5 * normals[keep])
    arrows.point_data["speed"] = speed[keep]
    arrows.point_data["vector"] = normals[keep] * speed[keep, None]
    return arrows.glyph(orient="vector", scale="speed", factor=factor,
                        geom=pv.Arrow(tip_length=0.3, tip_radius=0.12, shaft_radius=0.045))


def camera(plotter, centre, direction, distance):
    direction = direction / np.linalg.norm(direction)
    plotter.camera.position = centre + distance * direction
    plotter.camera.focal_point = centre
    plotter.camera.up = UP


def body_meshes(context):
    bone = polydata(context["body_bone_vertices"], context["body_bone_faces"]).smooth_taubin(n_iter=15)
    lungs = polydata(context["body_lungs_vertices"], context["body_lungs_faces"]).smooth_taubin(n_iter=15)
    return bone, lungs


def render_overview(final, context, size=1100):
    """Whole thorax from the front: where the tumour sits in the body."""
    bone, lungs = body_meshes(context)
    plotter = pv.Plotter(off_screen=True, window_size=(size, size))
    plotter.set_background("white")
    plotter.add_mesh(bone, color=BONE, opacity=0.35, smooth_shading=True)
    plotter.add_mesh(lungs, color=LUNG, opacity=0.25, smooth_shading=True)
    plotter.add_mesh(final, color=TUMOUR, smooth_shading=True, specular=0.3)
    centre = np.asarray(lungs.center)
    side = np.sign(final.center[0] - centre[0])
    camera(plotter, centre, np.array([0.35 * side, 1.0, 0.15]), 3.2 * max(lungs.bounds[1] - lungs.bounds[0], 1.0))
    image = plotter.screenshot(return_img=True)
    plotter.close()
    return image


def render_closeup(surface, context, ghost=None, arrows=None, size=1100):
    """Close-up from inside the lung, looking at the chest wall with the tumour on it."""
    normal = context["normal"]
    centre = np.asarray(surface.center)
    bone, _ = body_meshes(context)
    box = np.repeat(centre, 2) + np.tile([-85.0, 85.0], 3)
    ribs = bone.clip_box(box.tolist(), invert=False)
    plotter = pv.Plotter(off_screen=True, window_size=(size, size))
    plotter.set_background("white")
    plotter.add_mesh(ribs, color=BONE, opacity=0.9, smooth_shading=True)
    if ghost is not None:
        plotter.add_mesh(ghost, color=START, opacity=0.9, smooth_shading=True)
        plotter.add_mesh(surface, color=TUMOUR, opacity=0.35, smooth_shading=True)
    else:
        plotter.add_mesh(surface, color=TUMOUR, smooth_shading=True, specular=0.3)
    if arrows is not None:
        plotter.add_mesh(arrows, scalars="speed", cmap=ARROW_CMAP, clim=(0.0, 1.0), smooth_shading=True,
                         scalar_bar_args={"title": "growth speed (rel.)", "color": "black", "vertical": False,
                                          "position_x": 0.25, "position_y": 0.04, "width": 0.5, "height": 0.06,
                                          "title_font_size": 18, "label_font_size": 14, "n_labels": 3, "fmt": "%.1f"})
    side = np.cross(UP, normal)
    camera(plotter, centre, -normal + 0.35 * side / np.linalg.norm(side) + 0.3 * UP, 230.0)
    image = plotter.screenshot(return_img=True)
    plotter.close()
    return image


# ---------------------------------------------------------------- CT slices


def roi_extent(context):
    n = context["hu"].shape[0]
    x0, y0 = context["origin"][:2]
    return [x0, x0 + n, y0, y0 + n]


def overview_slice(axis, context, first, last):
    hu, (x0, y0) = context["axial_hu"], context["axial_origin"]
    axis.imshow(hu.T, cmap="gray", vmin=-1000, vmax=400, origin="lower",
                extent=[x0, x0 + hu.shape[0], y0, y0 + hu.shape[1]])
    k = first["phi"].shape[2] // 2
    extent = roi_extent(context)
    axis.contour(first["phi"][:, :, k].T, levels=[0.0], colors=START, linewidths=1.2, extent=extent, origin="lower")
    axis.contour(last["phi"][:, :, k].T, levels=[0.0], colors=TUMOUR, linewidths=1.6, extent=extent, origin="lower")
    axis.add_patch(Rectangle((extent[0], extent[2]), extent[1] - extent[0], extent[3] - extent[2],
                             fill=False, ec="#4aa3df", lw=1.2, ls="--"))
    for text, x, ha in [("R", 0.02, "left"), ("L", 0.98, "right")]:
        axis.text(x, 0.96, text, transform=axis.transAxes, color="white", ha=ha, va="top", fontsize=12, weight="bold")
    axis.set_axis_off()


def zoom_slice(axis, context, step, first=None):
    k = step["phi"].shape[2] // 2
    extent = roi_extent(context)
    axis.imshow(context["hu"][:, :, k].T, cmap="gray", vmin=-1000, vmax=400, origin="lower", extent=extent,
                interpolation="bicubic")
    if first is not None:
        axis.contour(first["phi"][:, :, k].T, levels=[0.0], colors=START, linewidths=1.4, linestyles="--",
                     extent=extent, origin="lower")
    axis.contourf(step["phi"][:, :, k].T, levels=[-1e9, 0.0], colors=[TUMOUR], alpha=0.35, extent=extent, origin="lower")
    axis.contour(step["phi"][:, :, k].T, levels=[0.0], colors=TUMOUR, linewidths=1.8, extent=extent, origin="lower")
    axis.set_axis_off()


def ct_slice(axis, context, phis, colour, plane="axial"):
    """Crop of the CT through the final tumour centre: start dashed white, end filled."""
    hu, origin = context["hu"], context["origin"]
    centre = np.argwhere(phis[-1] < 0).mean(axis=0).round().astype(int)
    if plane == "axial":
        cut = (slice(None), slice(None), centre[2])
        extent = [origin[0], origin[0] + hu.shape[0], origin[1], origin[1] + hu.shape[1]]
        middle = origin[:2] + centre[:2]
    else:
        cut = (slice(None), centre[1], slice(None))
        extent = [origin[0], origin[0] + hu.shape[0], origin[2], origin[2] + hu.shape[2]]
        middle = origin[[0, 2]] + centre[[0, 2]]
    axis.imshow(hu[cut].T, cmap="gray", vmin=-1000, vmax=300, origin="lower", extent=extent, interpolation="bicubic")
    axis.contourf(phis[-1][cut].T, levels=[-1e9, 0.0], colors=[colour], alpha=0.35, extent=extent, origin="lower")
    axis.contour(phis[-1][cut].T, levels=[0.0], colors=[colour], linewidths=2.2, extent=extent, origin="lower")
    axis.contour(phis[0][cut].T, levels=[0.0], colors="white", linewidths=1.8, linestyles="--",
                 extent=extent, origin="lower")
    half = CROP_MM / 2
    axis.set_xlim(middle[0] - half, middle[0] + half)
    axis.set_ylim(middle[1] - half, middle[1] + half)
    axis.set_axis_off()


def panel_grid(figure, top_in, rows, columns, size_in, gap_in=0.08, row_gap_in=0.3, left_in=0.25):
    """Axes of exactly size_in x size_in inches, laid out from top_in (inches from the top)."""
    width, height = figure.get_size_inches()
    axes = np.empty((rows, columns), dtype=object)
    for r in range(rows):
        y = height - top_in - (r + 1) * size_in - r * row_gap_in
        for c in range(columns):
            x = left_in + c * (size_in + gap_in)
            axes[r, c] = figure.add_axes([x / width, y / height, size_in / width, size_in / height])
    return axes


def label(figure, x_in, y_in_from_top, text, **kwargs):
    width, height = figure.get_size_inches()
    style = dict(fontsize=10, color=MUTED, va="bottom", ha="left") | kwargs
    figure.text(x_in / width, 1 - y_in_from_top / height, text, **style)


# ---------------------------------------------------------------- figures


def growth_overview(run_dir, path):
    """Six panels, 3 rows x 2 columns at 4:5: location, 3D, CT start/end, 3D start with arrows/end."""
    steps = [dict(np.load(p)) for p in sorted(run_dir.glob("step_*.npz"))]
    context = dict(np.load(run_dir / "context.npz"))
    origin, spacing = context["origin"], float(context["spacing"])
    first, last = steps[0], steps[-1]
    first_surface = surface_from_phi(first["phi"], origin, spacing)
    last_surface = surface_from_phi(last["phi"], origin, spacing)
    v0, v1 = volume_ml(first["phi"]), volume_ml(last["phi"])

    figure, axes = plt.subplots(3, 2, figsize=(8, 10))
    overview_slice(axes[0, 0], context, first, last)
    axes[0, 1].imshow(render_overview(last_surface, context))
    zoom_slice(axes[1, 0], context, first)
    zoom_slice(axes[1, 1], context, last, first=first)
    axes[2, 0].imshow(render_closeup(first_surface, context, arrows=growth_arrows(first, first_surface)))
    axes[2, 1].imshow(render_closeup(last_surface, context, ghost=first_surface))
    titles = [
        "a  Tumour on the right chest wall",
        "b  3D: rib cage, lungs, tumour",
        f"c  Start: {v0:.0f} ml",
        f"d  End: {v1:.0f} ml ({v1 / v0:.0f}x)",
        "e  Start; arrows = outward push",
        "f  End; yellow = starting tumour",
    ]
    for text, x, ha in [("L", 0.03, "left"), ("R", 0.97, "right")]:
        axes[0, 1].text(x, 0.95, text, transform=axes[0, 1].transAxes, ha=ha, va="top", fontsize=11, weight="bold")
    for axis, title in zip(axes.ravel(), titles):
        axis.set_axis_off()
        axis.set_title(title, loc="left", fontsize=11, va="bottom")
    figure.text(0.03, 0.985, "Where does prescribed tumour growth go?", fontsize=16, weight="bold", va="top")
    figure.text(0.03, 0.952, "Iterative FEM with CT-derived tissue stiffness", fontsize=11.5, color=MUTED, va="top")
    footer(figure, y=0.004)
    figure.subplots_adjust(left=0.01, right=0.99, top=0.905, bottom=0.045, wspace=0.03, hspace=0.13)
    figure.savefig(path, dpi=DPI)
    plt.close(figure)


def comparison(context, runs, results, labels, colours, title, path):
    """Two runs side by side; axial (top) and coronal (bottom) slice through the final tumour."""
    size = (8.0 - 0.5 - 0.08) / 2
    figure = plt.figure(figsize=(8, 1.75 + 2 * size + 0.3 + 0.9))
    label(figure, 0.25, 0.95, title, fontsize=17, color=INK, weight="bold")
    axes = panel_grid(figure, 1.75, 2, 2, size)
    for column, name in enumerate(runs):
        ct_slice(axes[0, column], context, runs[name], colours[name], plane="axial")
        ct_slice(axes[1, column], context, runs[name], colours[name], plane="coronal")
        share = results[name]["lung_share"]
        axes[0, column].set_title(f"{labels[name]}\n{100 * share:.1f}% of added volume in lung region",
                                  loc="left", fontsize=12, color=INK)
    for row, text in enumerate(["axial", "coronal"]):
        axes[row, 0].text(0.03, 0.97, text, transform=axes[row, 0].transAxes, color="white", fontsize=11,
                          va="top", weight="bold")
    first = next(iter(results.values()))
    label(figure, 0.25, 1.75 + 2 * size + 0.3 + 0.3,
          f"Dashed white = start ({first['start_ml']:.0f} ml), filled = end ({first['final_ml']:.0f} ml) in both runs. "
          "Lung region = lung on the original CT.")
    footer(figure, y=0.008)
    figure.savefig(path, dpi=DPI)
    plt.close(figure)


def shell_stiffness(hu, tumour, lung):
    """Median assigned E in a 6 mm shell around the starting tumour, per tissue class."""
    young = stiffness_from_hu(ndimage.uniform_filter(hu, 3))
    distance = ndimage.distance_transform_edt(~tumour)
    shell = (distance > 0) & (distance <= 6)
    classes = {"lung": shell & lung, "soft tissue": shell & ~lung & (hu < 200), "rib": shell & ~lung & (hu >= 200)}
    return {name: (float(np.median(hu[m])), float(np.median(young[m]))) for name, m in classes.items()}


def stiffness_map(run_dir, any_lung, path):
    """What the model sees: E(HU) on the slice through the tumour, plus the assumed curve."""
    context = np.load(run_dir / "context.npz")
    steps = sorted(run_dir.glob("step_*.npz"))
    first, last = dict(np.load(steps[0])), dict(np.load(steps[-1]))
    hu = context["hu"].astype(np.float32)
    origin = context["origin"]

    figure = plt.figure(figsize=(8, 10))
    figure.text(0.03, 0.975, "What the model sees", fontsize=17, color=INK, weight="bold", va="top")
    figure.text(0.03, 0.942, "Tissue stiffness derived from the CT, and how it pushes the tumour",
                fontsize=11.5, color=MUTED, va="top")

    axis = figure.add_axes([0.03, 0.39, 0.8, 0.5])
    centre = np.argwhere(last["phi"] < 0).mean(axis=0).round().astype(int)
    z = centre[2]
    extent = [origin[0], origin[0] + first["phi"].shape[0], origin[1], origin[1] + first["phi"].shape[1]]
    image = axis.imshow((10.0 ** first["young_grid"][:, :, z]).T, cmap="magma", norm=LogNorm(1.0, 1000.0),
                        origin="lower", extent=extent, interpolation="bicubic")
    axis.contour(last["phi"][:, :, z].T, levels=[0.0], colors="white", linewidths=2.2, extent=extent, origin="lower")
    axis.contour(first["phi"][:, :, z].T, levels=[0.0], colors="white", linewidths=1.8, linestyles="--",
                 extent=extent, origin="lower")
    # Displacement of the tissue around the tumour at step 0, in the plane of the slice.
    points, vectors = first["arrow_points"], first["arrow_vectors"]
    near = np.abs(points[:, 2] - (origin[2] + z)) < 1.0
    points, vectors = points[near], vectors[near]
    _, keep = np.unique(np.floor(points[:, :2] / 6.0).astype(int), axis=0, return_index=True)  # one per 6 mm cell
    points, vectors = points[keep], vectors[keep]
    length = np.linalg.norm(vectors[:, :2], axis=1)
    strong = length > 0.15 * length.max()
    scale = 9.0 / length.max()  # longest arrow 9 mm
    axis.quiver(points[strong, 0], points[strong, 1], vectors[strong, 0] * scale, vectors[strong, 1] * scale,
                color="#5ec8f2", angles="xy", scale_units="xy", scale=1, width=0.004, headwidth=4, headlength=4,
                zorder=4)
    middle = origin[:2] + centre[:2]
    axis.set_xlim(middle[0] - CROP_MM / 2, middle[0] + CROP_MM / 2)
    axis.set_ylim(middle[1] - CROP_MM / 2, middle[1] + CROP_MM / 2)
    axis.set_axis_off()
    colourbar = figure.colorbar(image, cax=figure.add_axes([0.86, 0.42, 0.025, 0.44]))
    colourbar.set_ticks([1, 10, 100, 1000], labels=["1", "10", "100", "1000"])
    colourbar.set_label("stiffness E (kPa)", color=MUTED, fontsize=10)
    colourbar.ax.tick_params(colors=MUTED, labelsize=10, length=0)
    colourbar.outline.set_visible(False)
    figure.text(0.03, 0.375, f"Axial slice. Dashed: start ({volume_ml(first['phi']):.0f} ml). "
                f"White: end ({volume_ml(last['phi']):.0f} ml). Blue arrows: tissue displacement at step 0.",
                fontsize=9.5, color=MUTED, va="top")

    axis = figure.add_axes([0.12, 0.1, 0.83, 0.21])
    hu_axis = np.linspace(-1000, 1000, 400)
    axis.plot(hu_axis, stiffness_from_hu(hu_axis), color=COLOURS["ct"], lw=2, label="CT-derived: E(HU), assumed")
    axis.axhline(HOMOGENEOUS_E_KPA, color=COLOURS["uniform"], lw=2, ls="--", label="Uniform control: 10 kPa")
    axis.set_yscale("log")
    axis.set_xlim(-1000, 1000)
    axis.set_ylim(0.7, 2000)
    for name, (h, e) in shell_stiffness(hu, first["phi"] < 0, any_lung).items():
        axis.plot(h, e, "o", ms=8, color=INK, mec="white", mew=2, zorder=5)
        axis.annotate(f"{name}\n{e:.1f} kPa" if e < 100 else f"{name}\n{e:.0f} kPa", (h, e), xytext=(10, 6),
                      textcoords="offset points", fontsize=10, color=INK, va="bottom")
    axis.set_xlabel("CT value (HU)", fontsize=10, color=MUTED)
    axis.set_ylabel("E (kPa)", fontsize=10, color=MUTED)
    axis.legend(loc="upper left", frameon=False, fontsize=10)
    for spine in ["top", "right"]:
        axis.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        axis.spines[spine].set_color(AXIS)
    axis.tick_params(colors=MUTED, labelsize=10, length=0)
    axis.grid(True, color=GRID, lw=0.8)
    axis.set_axisbelow(True)
    figure.text(0.03, 0.335, "Assumed stiffness curve. Dots: median around the starting tumour",
                fontsize=11.5, color=INK, va="top")
    footer(figure, y=0.008)
    figure.savefig(path, dpi=DPI)
    plt.close(figure)
