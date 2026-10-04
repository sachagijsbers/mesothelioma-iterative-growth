# Model

The model answers one question: **given a prescribed amount of tumour growth, how does the mechanical environment influence where that growth goes?**

It does not predict how fast a tumour grows. The volume schedule is an input; the mechanics only decide the spatial distribution.

Code: [`src/meso_growth/iterative.py`](../src/meso_growth/iterative.py) (mechanics and loop), [`geometry.py`](../src/meso_growth/geometry.py) (level set and volume update), [`data.py`](../src/meso_growth/data.py) (CT and starting tumour).

## 1. Setting

- **CT:** one public thorax CT (`coronacases_001`, see [data/README.md](../data/README.md)) with its expert lung segmentation.
- **Region of interest (ROI):** a 150 mm cube around a site on the lateral right chest wall, at 55% of the lung height. All fields live on a 1 mm grid of 151³ voxels in world coordinates.
- **Starting tumour (synthetic):** half an ellipsoid (semi-axes 38 × 30 mm, depth 24 mm) with its flat side on the parietal pleura, bulging into the lung. It is restricted to lung and to soft tissue between lung and ribs. Its volume is 44 ml.

## 2. Tumour representation

The tumour is the zero level set of a signed distance function φ on the 1 mm grid (φ < 0 inside), lightly smoothed (Gaussian, σ = 0.7 mm) so the interface is not voxel-stepped. Its volume is

$$V(\varphi) = \sum_{\text{voxels}} \operatorname{clip}\left(\tfrac12 - \tfrac{\varphi}{h},\ 0,\ 1\right) h^3, \qquad h = 1\ \text{mm}.$$

## 3. Mechanics

**Domain.** The finite-element domain is the whole ROI: tumour **and** surrounding lung, pleura, intercostal muscle and ribs. It uses a fixed structured tetrahedral mesh with 3 mm spacing (132,651 nodes, 750,000 linear tetrahedra, 397,953 degrees of freedom). The host mesh is **not remeshed**. The tumour enters each element through its volume fraction

$$\chi_e = \operatorname{clip}\left(\tfrac12 - \tfrac{\varphi(\mathbf{x}_e)}{\Delta},\ 0,\ 1\right), \qquad \Delta = 3\ \text{mm}.$$

**Material.** Each element gets a Young's modulus from the CT value (HU, 3 mm local mean), piecewise linear in $\log_{10} E$:

| HU | −1000 | −700 | −300 | 0 | 100 | 300 | 1000 |
|---|---|---|---|---|---|---|---|
| E (kPa) | 1 | 2 | 5 | 10 | 20 | 500 | 1000 |

Poisson's ratio is ν = 0.3 for HU < −300 (compressible lung) and 0.45 otherwise (nearly incompressible soft tissue). Voxels that have become tumour are given HU = 40 (soft-tissue range). In the **control run**, every element has E = 10 kPa and ν = 0.45.

**Growth as eigenstrain.** Growth is an isotropic eigenstrain inside the tumour, and small-strain linear elasticity is solved:

$$\nabla\cdot\boldsymbol\sigma = \mathbf 0, \qquad \boldsymbol\sigma = \mathbb C(\mathrm{HU}) : \big(\boldsymbol\varepsilon(\mathbf u) - \boldsymbol\varepsilon_g\big), \qquad \boldsymbol\varepsilon_g = \frac{\beta}{3}\,\chi\,\mathbf I,$$

with $\mathbf u = \mathbf 0$ on the outer ROI boundary. In weak form, the load is $\int (3\lambda + 2\mu)\,\tfrac{\beta}{3}\chi\,\operatorname{tr}\boldsymbol\varepsilon(\mathbf v)\,\mathrm dV$.

**Discretisation.** P1 vector elements with P0 material and one quadrature point (exact for constant strain). The system is solved with conjugate gradients (relative tolerance 10⁻⁸), preconditioned by smoothed-aggregation algebraic multigrid with the six rigid-body modes as near-nullspace. The run stops if CG does not converge.

**Why β does not matter.** The problem is linear in β, and the speed in step 4 is normalised. β (0.05) and a uniform scaling of E therefore do not change the growth pattern. Only the *ratios* of the stiffnesses, the Poisson's ratios, the geometry and the boundary condition do.

## 4. Where to grow: speed from the mechanics

The outward normal displacement of the tumour surface sets the local speed of the interface:

$$v_n = \max(\mathbf u \cdot \mathbf n,\ 0), \qquad \mathbf n = \frac{\nabla\varphi}{|\nabla\varphi|},$$

evaluated in a 12 mm band around the interface. It is smoothed (Gaussian, σ = 1.5 mm) and normalised to a maximum of 1.

## 5. How much to grow: prescribed volume

Each step moves the interface by $a\,v_n$ and chooses the step length $a$ by bisection (25 iterations), so that the new volume reaches the prescribed target:

$$\varphi_{k+1} = \operatorname{SDF}\big(\{\varphi_k - a\,v_n < 0\}\big), \qquad V(\varphi_{k+1}) = V_0 + (k+1)\,\Delta V.$$

In the experiment: 6 steps of $\Delta V = 14.6$ ml, from 44 to 132 ml (3×). The interface only moves outward ($v_n \ge 0$), so no tumour volume is removed.

## 6. Iteration loop

```
phi_0 <- starting tumour;  HU_0 <- CT with tumour voxels set to 40 HU
for k = 0 .. 5:
    E, nu  <- E(HU_k), nu(HU_k)          per element of the fixed 3 mm host mesh
    chi    <- volume fraction of phi_k   per element
    u      <- solve linear elasticity with eigenstrain (beta/3) chi I
    v_n    <- max(u . n, 0), smoothed, normalised
    phi_k+1 <- advance phi_k by a v_n, a chosen so that V = V_0 + (k+1) dV
    HU_k+1 <- HU_k with the new tumour voxels set to 40 HU
```

The only material update is that new tumour voxels take the tumour CT value. The surrounding anatomy (lung mask, other CT values) is **not** deformed: the tumour takes the place of the tissue it grows into instead of pushing it aside. See [ASSUMPTIONS.md](ASSUMPTIONS.md).

## 7. Metric

The headline number is the share of the **newly added** tumour volume that lies inside the lung region of the original CT:

$$\text{lung share} = \frac{|A \cap L|}{|A|}, \qquad A = T_{\text{end}} \setminus T_{\text{start}}, \qquad T = \{\varphi < 0\},$$

where $L$ is the filled right-lung segmentation of the original CT on the same 1 mm grid, and $|\cdot|$ counts 1 mm³ voxels. It is a 3D volume fraction: not an area on a slice, and not the total tumour volume. Code: [`metrics.py`](../src/meso_growth/metrics.py).

## 8. Thesis mechanics (for comparison)

[`thesis_reimpl.py`](../src/meso_growth/thesis_reimpl.py) runs the same loop (sections 2, 4, 5, 6) with the MSc thesis mechanics: only the tumour is meshed (GMSH, remeshed every step), and a density-based pressure $p = k\max(\rho - \rho_0, 0)$ drives the load. See [THESIS_VS_ITERATIVE.md](THESIS_VS_ITERATIVE.md).
