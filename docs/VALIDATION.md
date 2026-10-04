# Verification and validation

The current public experiment demonstrates **model behaviour**. It is not a clinical validation. This page separates three levels and states what has and has not been done at each.

## 1. Numerical verification: does the code solve the equations it claims to?

| Check | Result | Where |
|---|---|---|
| Spherical inclusion with uniform eigenstrain in a homogeneous medium, against the closed-form Eshelby solution $u_r = \frac{1+\nu}{3(1-\nu)}\,\varepsilon^*\,\frac{a^3}{r^2}$ | Displacement is radial (cos > 0.99) and within 1–2% of the exact magnitude at r = 12–15 mm (a = 9 mm, boundary at 45 mm) | `tests/test_verification.py` |
| Volume of the level set against an exact sphere | −0.7% at R = 20 mm. The smoothing of φ biases small convex shapes inward (−3.9% at R = 8 mm); for the experiment's tumour, the level-set volume and the voxel count differ by 0.1–0.3% | `tests/test_verification.py` |
| Volume update hits the prescribed target, without removing tumour | Holds | `tests/test_verification.py` |
| Stiffness mapping reproduces its table and is monotone | Holds | `tests/test_model_parts.py` |
| Metric counts only added voxels | Holds on a constructed example | `tests/test_model_parts.py` |
| Linear solver | CG with AMG reached the relative tolerance of 10⁻⁸ in all 14 solves (7 per run); the code raises an error otherwise | `iterative.solve_growth` |

**Reproducibility.** The control experiment was re-run from scratch with the repository code (`python -m experiments.run_stiffness_control`, 44 min on a laptop, Python 3.13, Windows). Both runs reproduced the earlier outputs bit for bit: the level set φ is identical at all 7 steps of both runs, and so are the metrics (56.9% and 97.9%; 87,837 and 87,844 added voxels). The 53.0% of the thesis re-implementation comes from the existing run, recomputed from its saved level sets with the repository's metric code; that simulation (about 4 h) was not repeated.

**Clean environment.** A fresh clone from GitHub in a new virtual environment installs with `pip install -e ".[test]"` and passes all tests. `python -m meso_growth.data` downloads CT files that are byte-identical to the ones used here, and builds the identical starting tumour. The full simulation was not repeated in that environment.

**Not done:** mesh convergence (only 3 mm was run), sensitivity to the ROI size and boundary condition, and convergence in the number of growth steps.

## 2. Controlled model behaviour: does the effect come from the mechanics?

The control experiment holds everything fixed except the material properties of the surroundings:

| | Uniform material (control) | CT-derived stiffness |
|---|---|---|
| Starting tumour | identical voxel mask (checked) | identical |
| Prescribed growth | 44 → 132 ml in 6 steps | same |
| Added volume | 87,837 voxels (87.8 ml) | 87,844 voxels (87.8 ml) |
| Tumour voxels lost | 0 | 0 |
| Tumour connected in every step | yes | yes |
| Closest distance to fixed boundary | 35 mm | 32 mm |
| **Added volume within lung region** | **56.9%** (49,981 voxels) | **97.9%** (85,987 voxels) |
| Added volume outside any lung | 43.1% | 2.1% |
| Centre-of-mass shift into the lung | −0.2 mm | +8.0 mm |

Per step, the share in the lung region stays between 97.6% and 98.1% with CT-derived stiffness. In the control, it falls from 60.4% to 54.1%.

**What this supports:** within this formulation and for the same prescribed volume, the spatial distribution of the added volume changes strongly when the surrounding tissue gets CT-derived properties. The difference is not caused by a different amount of growth.

**What it does not separate:** the CT run differs from the control in both E and ν (0.3 in lung). How much of the effect comes from the stiffness contrast and how much from the compressible lung has not been tested. Neither has the sensitivity to the E(HU) curve. Because the problem is linear, only the stiffness *ratios* matter, so a sensitivity study should vary those.

**Thesis mechanics on the same setup** (re-implementation, see [THESIS_VS_ITERATIVE.md](THESIS_VS_ITERATIVE.md)): 53.0% of the added volume (87,889 voxels) within the lung region. The mesh stays valid in all 7 remeshes (smallest tetrahedron > 4·10⁻⁴ mm³, mesh volume within 0.2 ml of the level-set volume).

## 3. Biological and clinical validation: does it describe real tumours?

**Not done.** No comparison with follow-up imaging of real mesothelioma, no real tumour geometry, no measured tissue stiffness, one CT only. The results are not predictions of where, how fast or how far a real tumour grows, and they must not be used for any clinical purpose.

What a first step towards validation would need: longitudinal CT of the same patients with tumour segmentations at two or more time points, the observed volume increase as the prescribed growth, and a comparison of predicted and observed spatial distribution with a metric like the one used here.
