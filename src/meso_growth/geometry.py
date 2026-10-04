"""Geometry shared by both models: region of interest, synthetic tumour, level set, volume update.

The tumour is a signed distance function phi on a 1 mm grid (phi < 0 inside). Each growth step
moves the interface along its outward normal with a speed field from the mechanics, and the step
length is chosen by bisection so that the volume hits a prescribed target.
"""

import numpy as np
from scipy import ndimage

GRID_MM = 1.0  # level-set grid spacing
ROI_HALF_MM = 75.0  # half width of the cubic region of interest around the tumour site
TUMOUR_HU = 40.0  # CT value assigned to voxels once they become tumour (soft-tissue range)


def hu_to_density(hu):
    """Approximate mass density in g/cm^3 from Hounsfield units (thesis mapping)."""
    return np.clip((hu + 1000.0) / 1000.0, 0.0, None)


def largest_component(mask):
    labels, count = ndimage.label(mask, structure=np.ones((3, 3, 3)))
    if count == 0:
        raise ValueError("empty mask")
    sizes = ndimage.sum(mask, labels, index=np.arange(1, count + 1))
    return labels == (int(np.argmax(sizes)) + 1)


def to_voxel(points, affine):
    """World coordinates (mm) to voxel indices of an image with the given affine."""
    return (np.linalg.inv(affine) @ np.column_stack([points, np.ones(len(points))]).T)[:3]


def pleural_site(lung, affine, height_fraction):
    """Most lateral lung voxel at a given height, with the outward lung normal."""
    index = np.argwhere(lung)
    world = index @ affine[:3, :3].T + affine[:3, 3]
    z = world[:, 2].min() + height_fraction * np.ptp(world[:, 2])
    slab = np.abs(world[:, 2] - z) <= 1.0
    lateral = np.sign(np.median(world[:, 0]))
    site_index = index[slab][np.argmax(lateral * world[slab, 0])]
    lo = np.maximum(site_index - 6, 0)
    patch = ndimage.gaussian_filter(
        lung[lo[0]:site_index[0] + 7, lo[1]:site_index[1] + 7, lo[2]:site_index[2] + 7].astype(float), 2.0
    )
    gradient = np.array([np.gradient(patch, axis=a)[tuple(site_index - lo)] for a in range(3)])
    normal = -np.linalg.inv(affine[:3, :3]).T @ gradient
    return world[slab][np.argmax(lateral * world[slab, 0])], normal / np.linalg.norm(normal)


class Roi:
    """Axis-aligned 1 mm grid in world coordinates around the tumour site."""

    def __init__(self, centre, half_mm=ROI_HALF_MM):
        n = int(2 * half_mm / GRID_MM) + 1
        self.origin = np.asarray(centre, dtype=float) - half_mm
        self.shape = (n, n, n)
        axes = [self.origin[a] + GRID_MM * np.arange(n) for a in range(3)]
        self.points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)

    def sample(self, volume, affine, order, cval):
        voxels = to_voxel(self.points.reshape(-1, 3), affine)
        return ndimage.map_coordinates(volume, voxels, order=order, cval=cval).reshape(self.shape)

    def index(self, world):
        return (np.asarray(world) - self.origin) / GRID_MM


def initial_tumour(roi, lung, allowed, site, normal, radii_mm=(38.0, 30.0), depth_mm=24.0):
    """Dome-shaped pleural tumour: half an ellipsoid on the chest wall, bulging into the lung.

    The flat side lies on the parietal pleura (just outside the lung surface at the site),
    the dome pushes the lung inward. Chest wall outside the rib cage is excluded.
    """
    t1 = np.cross(normal, [0.0, 0.0, 1.0])
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(normal, t1)
    rel = roi.points - (site + 3.0 * normal)
    ellipsoid = (rel @ t2 / radii_mm[0]) ** 2 + (rel @ t1 / radii_mm[1]) ** 2 + (rel @ normal / depth_mm) ** 2 < 1.0
    mask = ellipsoid & (rel @ normal <= 0.0) & (lung | allowed)
    mask = ndimage.binary_opening(mask, structure=np.ones((3, 3, 3)), iterations=2)
    return ndimage.binary_fill_holes(largest_component(mask))


def pleural_space(hu_roi, any_lung):
    """Soft tissue that lies closer to the lung than to bone: between ribs and lung.

    A pleural tumour grows inside the rib cage, so chest wall tissue outside the ribs
    is excluded from the initial tumour.
    """
    bone = ndimage.binary_dilation(hu_roi > 200.0)
    to_lung = ndimage.distance_transform_edt(~any_lung)
    to_bone = ndimage.distance_transform_edt(~bone)
    return ~any_lung & ~bone & (hu_roi > -500.0) & (to_lung < to_bone)


def signed_distance(mask):
    """Negative inside, lightly smoothed so the zero level set is not voxel-stepped."""
    phi = ndimage.distance_transform_edt(~mask) - ndimage.distance_transform_edt(mask)
    return ndimage.gaussian_filter(phi * GRID_MM, 0.7)


def volume_of(phi):
    """Volume in mm^3 enclosed by the zero level set, with partial-voxel weighting."""
    return float(np.clip(0.5 - phi / GRID_MM, 0.0, 1.0).sum() * GRID_MM**3)


def advance(phi, speed, target_volume):
    """Move the interface by a * speed and choose a so the volume hits the target.

    speed is a non-negative field normalised to max 1; returns the new phi and the
    maximum advance a in mm.
    """
    def grown(a):
        return signed_distance(phi - a * speed < 0.0)

    low, high = 0.0, 1.0
    while volume_of(grown(high)) < target_volume:
        low, high = high, 2.0 * high
        if high > 64.0:
            raise RuntimeError("cannot reach target volume inside the region of interest")
    for _ in range(25):
        middle = 0.5 * (low + high)
        low, high = (middle, high) if volume_of(grown(middle)) < target_volume else (low, middle)
    return grown(high), high
