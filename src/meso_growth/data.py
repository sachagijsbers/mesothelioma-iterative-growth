"""Public CT data: download one case and build the simulation setup shared by every run.

Source: COVID-19 CT Lung and Infection Segmentation Dataset (Ma et al., Zenodo record 3757476,
https://zenodo.org/records/3757476). CT scans: Coronacases Initiative; lung masks: Ma et al.
See data/README.md for licence and citation. The CT archive is 1.1 GB, so a single case is
read from it with HTTP range requests instead of downloading the full zip.
"""

import argparse
import io
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

from .geometry import Roi, initial_tumour, largest_component, pleural_site, pleural_space

RECORD = "https://zenodo.org/api/records/3757476/files"
CT_ZIP = f"{RECORD}/COVID-19-CT-Seg_20cases.zip/content"
MASK_ZIP = f"{RECORD}/Lung_Mask.zip/content"
DEFAULT_CASE = "coronacases_001"
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"


class HttpRangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP range requests."""

    def __init__(self, url: str) -> None:
        response = urllib.request.urlopen(urllib.request.Request(url, method="HEAD"))
        self.url = response.url
        self.size = int(response.headers["Content-Length"])
        self.position = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.position, io.SEEK_END: self.size}[whence]
        self.position = base + offset
        return self.position

    def tell(self) -> int:
        return self.position

    def readinto(self, buffer) -> int:
        count = min(len(buffer), self.size - self.position)
        if count <= 0:
            return 0
        headers = {"Range": f"bytes={self.position}-{self.position + count - 1}"}
        data = urllib.request.urlopen(urllib.request.Request(self.url, headers=headers)).read()
        buffer[: len(data)] = data
        self.position += len(data)
        return len(data)


def extract_member(url: str, member: str, destination: Path) -> None:
    if destination.exists():
        return
    remote = io.BufferedReader(HttpRangeFile(url), buffer_size=8 << 20)
    with zipfile.ZipFile(remote) as archive:
        destination.write_bytes(archive.read(member))
    print(f"wrote: {destination}")


def fetch(case=DEFAULT_CASE, data_dir=DEFAULT_DATA_DIR):
    """Download the CT and lung mask of one case, if not already present."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    extract_member(MASK_ZIP, f"{case}.nii.gz", data_dir / f"{case}_lung_mask.nii.gz")
    extract_member(CT_ZIP, f"{case}.nii.gz", data_dir / f"{case}_ct.nii.gz")


@dataclass
class Case:
    """Everything a run needs: the full CT, the region of interest and the starting tumour."""

    hu: np.ndarray  # full CT in HU
    labels: np.ndarray  # lung segmentation labels of the full CT
    affine: np.ndarray
    site: np.ndarray  # tumour site on the chest wall (world mm)
    normal: np.ndarray  # outward lung normal at the site
    roi: Roi
    hu_roi: np.ndarray  # CT resampled on the 1 mm ROI grid
    lung: np.ndarray  # filled right-lung mask on the ROI grid (used by the metric)
    any_lung: np.ndarray  # any lung label on the ROI grid
    tumour: np.ndarray  # starting tumour mask on the ROI grid


def setup_case(case=DEFAULT_CASE, data_dir=DEFAULT_DATA_DIR, height=0.55):
    """Load the CT, pick the pleural site on the right lung and build the starting tumour."""
    data_dir = Path(data_dir)
    ct = nib.load(str(data_dir / f"{case}_ct.nii.gz"))
    hu = ct.get_fdata(dtype=np.float32)
    labels = np.asanyarray(nib.load(str(data_dir / f"{case}_lung_mask.nii.gz")).dataobj)
    affine = ct.affine

    # Right lung = negative x in the RAS world frame.
    centroid_x = {v: (np.argwhere(labels == v).mean(axis=0) @ affine[:3, :3].T + affine[:3, 3])[0]
                  for v in np.unique(labels) if v != 0}
    lung_full = ndimage.binary_fill_holes(largest_component(labels == min(centroid_x, key=centroid_x.get)))
    site, normal = pleural_site(lung_full, affine, height)

    roi = Roi(site)
    hu_roi = roi.sample(hu, affine, order=1, cval=-1000.0)
    lung = roi.sample(lung_full.astype(np.float32), affine, order=1, cval=0.0) > 0.5
    any_lung = roi.sample((labels > 0).astype(np.float32), affine, order=1, cval=0.0) > 0.5
    tumour = initial_tumour(roi, lung, pleural_space(hu_roi, any_lung), site, normal)
    return Case(hu, labels, affine, site, normal, roi, hu_roi, lung, any_lung, tumour)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default=DEFAULT_CASE)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args()
    fetch(args.case, args.data_dir)


if __name__ == "__main__":
    main()
