# Assumptions and simplifications

Each item below is a modelling choice that shapes the result. Most are deliberate simplifications for an exploratory study, not claims about tumour biology.

## Growth

- **Growth is prescribed, not predicted.** The total volume (3× in 6 equal steps) is an input. The model says nothing about growth rate, timing or final size.
- **Growth is purely mechanical in direction.** The interface moves where the tissue lets it move outward most ($v_n = \max(\mathbf u\cdot\mathbf n, 0)$). There is no biology: no nutrient or oxygen supply, no cell proliferation or death, no invasion, no stress-dependent growth law.
- **Isotropic eigenstrain.** Growth acts as a uniform isotropic expansion of the whole tumour, scaled by its volume fraction per element.
- **Outward motion only.** The interface never retreats, so no tumour volume is lost (checked: 0 voxels lost in every run).

## Tumour and anatomy

- **Synthetic tumour.** The starting tumour is a half-ellipsoid placed on the chest wall, not a segmented real mesothelioma. Real pleural mesothelioma typically spreads as a rind along the pleura.
- **One CT, one site.** A single public CT (a COVID-19 case, not a mesothelioma patient), with a single tumour site and shape.
- **Anatomy is not deformed.** The lung mask and the CT values of the surroundings stay fixed. A grown tumour takes the place of the tissue it grows into instead of pushing that tissue aside. "Within the lung region" therefore means "inside the region that was lung on the original CT".
- **Tumour material.** Voxels that become tumour get HU = 40, so E ≈ 13 kPa and ν = 0.45.

## Mechanics

- **CT → stiffness mapping is assumed.** E(HU) is a piecewise log-linear curve with literature-order-of-magnitude values (lung ~1–3 kPa, soft tissue ~10–20 kPa, bone ~0.5–1 MPa). It is not measured or calibrated for this patient. Around the starting tumour, it assigns a median of 1.5 kPa to lung, 10.6 kPa to soft tissue and 531 kPa to rib, so lung is about **7× softer** than the tissue between the ribs.
- **Poisson's ratio is binary.** ν = 0.3 in lung (compressible) and 0.45 elsewhere (nearly incompressible). The control run uses 0.45 everywhere, so the two runs differ in E **and** ν.
- **Small-strain linear elasticity.** No hyperelasticity, viscoelasticity, plasticity or residual stress. Each step is solved from a stress-free state; stress does not accumulate between steps.
- **Boundary condition.** u = 0 on the faces of the 150 mm ROI cube. The final tumour stays 32–35 mm away from this boundary in every run. Sensitivity to the ROI size has not been tested.
- **No pleural sliding.** Lung and chest wall are bonded into one continuum. In reality, the visceral and parietal pleura can slide over each other, which would let a tumour spread along the chest wall.
- **No breathing, no lung pressure, no gravity.** The lung is a static elastic solid at the inflation state of the scan.
- **Fixed 3 mm host mesh.** The thinnest parts of the tumour are resolved only through the volume fraction χ. Ribs are partly smeared into neighbouring intercostal tissue by the 3 mm averaging: 14% of the soft-tissue voxels around the tumour get E > 50 kPa.

## Numerics

- **Level-set smoothing.** The signed distance is smoothed (σ = 0.7 mm), which slightly underestimates the volume of curved shapes (−0.1 to −0.3% for this tumour; see [VALIDATION.md](VALIDATION.md)).
- **Speed smoothing.** $v_n$ is smoothed with σ = 1.5 mm, which suppresses features smaller than a few millimetres.

## Thesis mechanics (comparison run only)

- Only the tumour is meshed, and the surroundings enter only through the density in the load $p = k\max(\rho-\rho_0, 0)$.
- On the synthetic tumour no node reaches ρ ≥ 2, so the tumour is held only at the centre-of-mass node, with a weak elastic foundation and removal of rigid-body motion.
- This is a re-implementation of the thesis formulation, not the original thesis code. See [THESIS_VS_ITERATIVE.md](THESIS_VS_ITERATIVE.md).
