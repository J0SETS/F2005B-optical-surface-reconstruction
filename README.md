# Reconstructing two interfaces of glass from ray data

This project implements Rubinstein & Wolansky, *Reconstruction of Optical Surfaces
from Ray Data*, Optical Review **8**, 281–283 (2001),
[DOI: 10.1007/s10043-001-0281-4](https://doi.org/10.1007/s10043-001-0281-4).
The supplied PDF is the source for the equations; the
[journal abstract](https://annex.jsap.or.jp/photonics/opticalreview/TOC-Lists/vol08/8d281tx.htm)
identifies the original work. Section 4, Eqs. (4)–(7), supplies the coupled PDEs.
Section 3 supplies the squared PDE-residual cost, extended to both surfaces as
suggested at the end of Section 4. This is a numerical implementation of that
formulation, not author-supplied software.

The laminated sample is approximated as one homogeneous, isotropic material:
**air (1.00) → glass (1.52) → air (1.00)**. We reconstruct its two outer interfaces.
Individual glass/polymer layers, dispersion, scattering, and absorption are not
modeled. All lengths are **millimetres**; all directions are unit vectors.

## Open the demonstration

Open `output/curved_reconstruction.html` in a browser. Drag to rotate freely,
scroll to zoom, and use the toolbar to pan or reset. The HTML includes Plotly and
works without an internet connection. `output/planes_reconstruction.html` is the
two-plane example; `output/curved_noisy_reconstruction.html` includes noise.

Open `geometria_optica.ipynb` for the numbered, executable explanation. Select
the project's `.venv/Scripts/python.exe` interpreter in VS Code and run all cells.
The notebook uses Plotly's MIME renderer supported by VS Code/JupyterLab; if a
notebook viewer does not render it, use the standalone HTML files.

```powershell
# From C:\Libros\Optica\Reto; no virtual-environment activation is required.
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe demo.py --case curved
.venv/Scripts/python.exe demo.py --case planes
.venv/Scripts/python.exe demo.py --spot-noise-mm 0.002 --direction-noise-rad 0.00002
.venv/Scripts/python.exe -m unittest discover -s tests -v
```

The notebook is already executed. To regenerate its clean source, run
`python tools/build_notebook.py`; this deliberately replaces notebook edits.
`python tools/execute_notebook.py` executes the existing notebook in place.
These commands use whichever interpreter you select; use `.venv/Scripts/python.exe`
in place of `python` for the project environment.

## Block 1 — What must be measured

For each matched ray, the two-surface method uses:

| Quantity | Meaning | Array shape |
|---|---|---|
| `x`, `y` | Rectangular input-ray coordinates on the plane `z=h1` | `(nx,)`, `(ny,)` |
| `w_in` | Incoming direction `(alpha,beta,gamma)` | `(ny,nx,3)` |
| `exit_xy` | Matched exit position `(xi,eta)` on `z=h2` | `(ny,nx,2)` |
| `w_out` | Outgoing direction `(a,b,c)` | `(ny,nx,3)` |
| `h1`, `h2` | Calibrated measurement plane heights | scalars |
| `Anchors` | Known front and back heights on one chosen matched ray | row, col, f, g |

One image on one downstream screen generally does **not** provide the exit
direction as well as position. A practical way to obtain that direction is to
track the same ray on two calibrated downstream planes and normalize the vector
joining the two spots. Incoming rays also need calibrated positions/directions.
The data must retain ray correspondence through all measurements.

The two reference heights are explicit geometric information, not recovered from
the synthetic truth by the inverse solver. In the demo they are read from the
known sample to mimic independent calibration. They are heights at the two
intersections of the *same ray*, generally at different physical XY coordinates.
If you have thickness at a common XY location instead, converting that measurement
to this anchor format requires an additional constraint implementation.

The experiment also supplies conservative disjoint height brackets: the demo uses
front in `(5,13)` and back in `(13.5,23)` mm. These keep optimization within a
forward, separated-sheet geometry. No true curvature, normal, or internal ray is
passed to `reconstruct()`.

## Block 2 — Coordinates and physical geometry

Lowercase `(x,y)` label a ray at the entrance measurement plane. They are not the
physical surface coordinates `(X,Y)`. The unknown scalar fields `f(x,y),g(x,y)`
are the heights at which that ray meets the two interfaces. Define

\[
s=(\alpha/\gamma,\beta/\gamma),\qquad t=(a/c,b/c),\qquad q=(\xi,\eta).
\]

The physical points and internal direction are

\[
P_1=(x+s_x(f-h_1),\ y+s_y(f-h_1),\ f),
\quad P_2=(\xi+t_x(g-h_2),\ \eta+t_y(g-h_2),\ g),
\quad w_2=\frac{P_2-P_1}{\|P_2-P_1\|}=(A,B,C).
\]

In this paragraph subscripts on `s` and `t` denote components. Below, derivative
notation such as `partial_x s` denotes a derivative of the entire vector field.
Plots use **P1 and P2**, never the input grid as if it were the physical surface.
In particular, `g-f` is separation along the ray's z coordinates; it is generally
not thickness at a common XY point, and not normal thickness.

## Block 3 — Snell slopes and Eqs. (4)–(7)

In physical coordinates, the two surface gradients are

\[
u=\frac{(n_2A-n_1\alpha,\ n_2B-n_1\beta)}{n_1\gamma-n_2C},
\qquad
v=\frac{(n_2A-n_3a,\ n_2B-n_3b)}{n_3c-n_2C}.
\]

Applying the chain rule to the ray parameterization gives the paper's system:

\[
e_{fx}=\frac{u_1+(f-h_1)u\cdot\partial_xs}{1-u\cdot s},\qquad
e_{fy}=\frac{u_2+(f-h_1)u\cdot\partial_ys}{1-u\cdot s},
\]
\[
e_{gx}=\frac{v\cdot\partial_xq+(g-h_2)v\cdot\partial_xt}{1-v\cdot t},\qquad
e_{gy}=\frac{v\cdot\partial_yq+(g-h_2)v\cdot\partial_yt}{1-v\cdot t}.
\]

The equations are `f_x=e_fx`, `f_y=e_fy`, `g_x=e_gx`, `g_y=e_gy`.
Each right-hand side depends on **both unknown surfaces** via `w2`.
The implementation uses the exact vector Snell law, without a paraxial approximation.

Two notation issues in the printed paper are handled explicitly: the entrance
intersection uses `(f-h1)` for arbitrary `h1`, consistent with its displacement
formula and Eqs. (4)–(5); the second exit slope is named `v2` (printed as `u2`
in its definition on page 283). The internal displacement is normalized by its
Euclidean length, including the square root.

## Block 4 — Loss function and numerical solution

For input-ray domain area `|Omega|`, minimize

\[
E(f,g)=\frac{1}{|\Omega|}\int_\Omega
[(f_x-e_{fx})^2+(f_y-e_{fy})^2+(g_x-e_{gx})^2+(g_y-e_{gy})^2]\,dx\,dy.
\]

The constant area normalization does not change the minimizer. Three-point,
second-order finite differences approximate derivatives, including one-sided
edge stencils. Tensor-product trapezoidal weights discretize the integral. This
also supports nonuniform rectangular input grids. The demo's 21×21 grid has
882 heights, 880 free parameters after fixing the two anchors, and 1764 PDE
residuals. Initialization is two constant fields at the supplied anchor heights.

`scipy.optimize.least_squares` minimizes the stacked, area-weighted residuals.
A sparse Jacobian pattern captures local derivative stencils and coupling between
the two surfaces. Anchors are removed from the unknown vector, so they hold
exactly. There is no fitted polynomial restriction or extra smoothing penalty.
The optimizer's `cost` is `E/2`; exported `energy` is `E`.

The paper's compatibility conditions need not hold exactly for noisy observations.
Least squares finds a compromise, but differentiation can amplify noise. The
current objective is the paper's uniform-weight energy, not a statistically
optimal measurement-likelihood model. It does not include uncertainty in anchors,
incoming calibration, or refractive index. Additional regularization, uncertainty
analysis, or smoothing of measured fields would be extensions for experimental use.

## Block 5 — Known demo geometry and independent measurements

For the curved example, with X,Y in mm:

```text
front: Z = 10 + 0.015 X - 0.010 Y + 0.002 X² + 0.0015 Y²
back:  Z = 16 - 0.010 X + 0.018 Y + 0.001 X² + 0.002 Y²
```

The input grid covers `[-10,10] × [-10,10]` mm at `h1=0`; the exit plane is
`h2=35` mm. Incoming direction is proportional to
`(0.08+0.002*x, -0.04+0.001*y, 1)`: an oblique, weakly divergent beam.
`intersect_quadratic()` uses analytic intersections and `refract()` uses vector
Snell refraction to generate external measurements. The inverse method receives
only those measurements, anchors, indices, and broad brackets. It does not use
the quadratic coefficients. The planar example uses two differently tilted planes.

The noise example adds independent Gaussian exit-spot noise with standard deviation
0.002 mm in each coordinate, plus direction-vector component noise of 0.00002
before renormalization. The seed is 7. That angular noise parameter is not exactly
the resulting total angular RMS. Anchors remain exact in this example.

## Block 6 — Validation and interpretation

`surface_errors()` evaluates each known physical graph at the reconstructed
physical XY coordinates and compares it with reconstructed Z. This avoids confusing
ray labels with surface coordinates. The independent Snell check computes normals
from mesh tangents and checks both refracted directions. Diagnostics also report
convergence, active bounds, minimum ray z-separation, and XY Jacobians to detect
local mesh folds. A positive local Jacobian is not a general proof of global
injectivity or nonintersection for arbitrary optics.

The default curved demo recovers heights with approximately **6.58 nm front RMS**
and **3.56 nm back RMS**. These small values describe noiseless simulated data and
discretization error. The supplied noise example gives errors of approximately
**2.63 µm** and **2.40 µm**, with exact anchors. See the generated JSON files for
the actual results of each run. Refining the grid reduces clean-data errors, while
real data can have a different optimum sampling/smoothing tradeoff.

The tests cover scalar Snell agreement, total internal reflection, analytic
derivatives, two-plane reconstruction, curved-grid refinement, a shifted coordinate
origin, a noisy reconstruction, data validation, and a deliberately ambiguous
normal-incidence slab. A parallel slab at normal incidence produces unchanged
external rays whatever its thickness or axial position: the anchors supply
information that the rays alone cannot determine. Anchors and convergence do not
prove uniqueness for every possible optical configuration.

## Block 7 — Using real ray data

```python
from reconstruction import RayData, Anchors, reconstruct

# Fill x, y, w_in, exit_xy, w_out with calibrated measurements.
data = RayData(x=x, y=y, w_in=w_in, exit_xy=exit_xy, w_out=w_out,
               h1=0.0, h2=35.0, n1=1.0, n2=1.52, n3=1.0)
# f_ref and g_ref are independently known heights on this matched ray.
anchors = Anchors(row=reference_row, col=reference_col, f=f_ref, g=g_ref)
result = reconstruct(data, anchors, f_bounds=(5,13), g_bounds=(13.5,23))
front_xyz, back_xyz = result.p1, result.p2  # physical surface meshes
data.save('my_measurements.npz')
```

Choose planes and bounds appropriate to your apparatus. This implementation assumes
smooth, single-valued, gently sloped surfaces in separate z brackets, forward
transmission, no total internal reflection, and a regular ray-to-surface mapping.
Unstructured samples must first be converted to calibrated rectangular ray-label
fields; arbitrary reshaping is not valid. Missing rays are not supported.
Near singularities or strongly deformed initial guesses can cause explicit failure;
the local optimizer is not a global solution method.

## Files

| File | Purpose |
|---|---|
| `geometria_optica.ipynb` | Numbered explanation, editable parameters, executed plots |
| `reconstruction.py` | Six documented blocks: data, derivatives, PDEs, solve, forward model, validation |
| `demo.py` | Four documented blocks: examples, interactive plots, HTML export, runner |
| `tests/test_reconstruction.py` | Physics and numerical regression tests |
| `output/*_measurements.npz` | External observations only; no hidden geometry |
| `output/*_reconstruction.npz` | Recovered fields and physical meshes |
| `output/*_metrics.json` | Diagnostics, truth parameters, anchors, bounds, noise settings |
| `output/*_reconstruction.html` | Portable, interactive 3D plots |
| `requirements.txt` | Runtime and notebook dependencies |

Reading the original PDF used `pypdf` and `pymupdf`; neither is required to run the
reconstruction. No browser service or external account is needed.
