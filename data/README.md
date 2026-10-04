# Data

No imaging data is stored in this repository. The experiments download what they need.

## Dataset

**COVID-19 CT Lung and Infection Segmentation Dataset** (Ma et al., 2020)

- Source: Zenodo, https://zenodo.org/records/3757476 (DOI [10.5281/zenodo.3757476](https://doi.org/10.5281/zenodo.3757476))
- Contents: 20 thorax CT scans with left lung, right lung and infection labelled by two radiologists and verified by a third.
- CT scans: Coronacases Initiative and Radiopaedia, originally from https://academictorrents.com/details/136ffddd0959108becb2b3a86630bec049fcb0ff
- Case used: **`coronacases_001`** (CT from `COVID-19-CT-Seg_20cases.zip`, lung mask from `Lung_Mask.zip`)

This is a COVID-19 scan, not a mesothelioma patient. The tumour in the simulations is synthetic.

## Licence

The ReadMe inside the archive states: *"The 20 CT scans are from the Coronacases Initiative and Radiopaedia … with license CC BY-NC-SA. … The license of annotations is also CC BY-NC-SA."* The Zenodo record page lists CC BY 4.0. This repository follows the more restrictive terms, **CC BY-NC-SA**:

- the data is not redistributed here (download it from the source);
- figures derived from the CT (everything in `figures/`) are shared under CC BY-NC-SA 4.0, with attribution, for non-commercial use.

## Citation

> Jun Ma, Cheng Ge, Yixin Wang, Xingle An, Jiantao Gao, Ziqi Yu, Minqing Zhang, Xin Liu, Xueyuan Deng, Shucheng Cao, Hao Wei, Sen Mei, Xiaoyu Yang, Ziwei Nie, Chen Li, Lu Tian, Yuntao Zhu, Qiongjie Zhu, Guoqiang Dong, Jian He. *COVID-19 CT Lung and Infection Segmentation Dataset* (Version 1.0). Zenodo, 2020. https://doi.org/10.5281/zenodo.3757476

## Download

```bash
python -m meso_growth.data            # writes data/coronacases_001_ct.nii.gz and _lung_mask.nii.gz
```

The experiments call this automatically. The CT archive is 1.1 GB, so only the one case is read from it, with HTTP range requests (about 95 MB).

## Preprocessing (done in code, see `src/meso_growth/data.py`)

1. The right lung is the lung label whose centroid lies at negative x in world (RAS) coordinates. Its largest connected component is kept and its holes are filled, so vessels and airways inside the lung count as lung.
2. A 150 mm cube with a 1 mm grid is placed around the most lateral point of the right lung at 55% of its height. The CT and the lung masks are resampled onto it with the full image affine (linear interpolation, masks thresholded at 0.5).
3. The synthetic starting tumour is built on that grid (see [docs/MODEL.md](../docs/MODEL.md)).
