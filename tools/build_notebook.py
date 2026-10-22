"""Generate the documented notebook from ordered Markdown/code blocks.

Run from the project root. This intentionally overwrites geometria_optica.ipynb;
edit the notebook directly for experiments and only rebuild when that is wanted.
"""
from pathlib import Path
from textwrap import dedent

import nbformat as nbf


def build():
    """Assemble a readable walkthrough without duplicating the reusable solver."""
    cells = []

    def markdown(text):
        """Append an explanatory block with indentation removed."""
        cells.append(nbf.v4.new_markdown_cell(dedent(text).strip()))

    def code(text):
        """Append an executable block with indentation removed."""
        cells.append(nbf.v4.new_code_cell(dedent(text).strip()))

    markdown(r'''
    # Two-interface reconstruction of glass from ray data

    **Air (n=1.00) → effective glass (n=1.52) → air (n=1.00)**

    This notebook implements the two-unknown-surface formulation of Rubinstein &
    Wolansky (2001), *Reconstruction of Optical Surfaces from Ray Data*,
    [DOI: 10.1007/s10043-001-0281-4](https://doi.org/10.1007/s10043-001-0281-4).
    We minimize the squared residuals of Eqs. (4)–(7), following the cost-function
    approach in Section 3 and its extension in Section 4.

    The laminate is treated as one homogeneous material. Both **outer interfaces**
    are unknown. All positions are in **mm**. No individual polymer/glass layers
    are reconstructed.

    Run blocks in order with the project's `.venv/Scripts/python.exe` kernel.
    The solver is documented in six `# %%` blocks in `reconstruction.py`, and
    plotting/export is in four blocks in `demo.py`. `README.md` contains the full
    equations, input requirements, and limitations. The notebook remains editable.
    ''')
    markdown('''
    ## Block 1 — Imports and editable experimental settings

    The default uses 441 matched rays. Set `CASE = "planes"` for two tilted planes,
    or keep `"curved"` for two gentle quadratic surfaces. `GRID_SIZE` must be odd
    so a central ray can supply both reference heights. The broad height brackets
    are independent geometric priors, not fitted truth parameters.
    ''')
    code('''
    from dataclasses import asdict
    from pathlib import Path
    import inspect
    import json
    import numpy as np
    import plotly.io as pio
    from IPython.display import Code, Markdown, display

    from reconstruction import (RayData, Anchors, PaperSystem, reconstruct,
                                simulate, surface_errors)
    from demo import demo_surfaces, make_figure, write_html, run_demo

    # VS Code and JupyterLab support this interactive Plotly MIME representation.
    pio.renderers.default = "plotly_mimetype"
    CASE = "curved"
    GRID_SIZE = 21
    F_BOUNDS_MM = (5.0, 13.0)
    G_BOUNDS_MM = (13.5, 23.0)
    OUTPUT = Path("output")
    OUTPUT.mkdir(exist_ok=True)
    ''')
    markdown(r'''
    ## Block 2 — Choose the known sample (used only for simulation and validation)

    For the curved example, the physical surface graphs are:

    $$F(X,Y)=10+0.015X-0.010Y+0.002X^2+0.0015Y^2,$$
    $$G(X,Y)=16-0.010X+0.018Y+0.001X^2+0.002Y^2.$$

    The capital coordinates are physical coordinates. These coefficients are
    **never passed to the inverse solver**. The solver has one free height per
    sampled ray per surface (apart from the two anchors), not a quadratic fit.
    ''')
    code('''
    front, back = demo_surfaces(CASE)
    print("Known front:", asdict(front))
    print("Known back: ", asdict(back))
    ''')
    markdown('''
    ## Block 3 — Generate external measurements by independent ray tracing

    Rays start on `z=0` over a 20×20 mm square; they are measured again on `z=35`.
    Incoming directions are proportional to `(0.08+0.002*x, -0.04+0.001*y, 1)`.
    Analytic ray/quadric intersections plus vector Snell refraction generate the
    observations. The default has no noise; Block 10 adds measurement noise.

    The two anchor heights are extracted here from the synthetic sample to mimic
    independent calibration. For real glass you must supply those values from
    measured geometry or implement other suitable constraints. They are **not**
    two heights at a common XY point: they belong to the same matched ray.
    ''')
    code('''
    data, anchors, truth = simulate(front, back, grid_size=GRID_SIZE)
    print(f"Indices: {data.n1:.2f} / {data.n2:.2f} / {data.n3:.2f}")
    print("Grid:", data.shape, "  Rays:", np.prod(data.shape))
    print("Known calibration:", asdict(anchors))
    # Save measurements separately: the file contains neither surfaces nor internal rays.
    data.save(OUTPUT / f"{CASE}_measurements.npz")
    ''')
    markdown(r'''
    ## Block 4 — Relate ray labels to physical interface points

    Let $(x,y)$ label each incoming ray; $s=(α/γ,β/γ)$,
    $t=(a/c,b/c)$, and $q=(ξ,η)$ be its measured exit point.
    The unknown fields $f(x,y),g(x,y)$ are **intersection heights**.

    $$P_1=((x,y)+s(f-h_1),f),\quad
    P_2=(q+t(g-h_2),g),\quad w_2=(P_2-P_1)/\|P_2-P_1\|.$$

    The code below makes the same initial guess used by the optimizer: two flat
    height fields at the anchor values. The true surfaces are not used here.
    ''')
    code('''
    system = PaperSystem(data)
    f_initial = np.full(data.shape, anchors.f)
    g_initial = np.full(data.shape, anchors.g)
    p1_initial, p2_initial, w2_initial = system.geometry(f_initial, g_initial)
    print("Initial physical front mesh shape:", p1_initial.shape)
    print("Initial PDE slope RMS:", np.sqrt(np.mean(
        system.residual_fields(f_initial, g_initial)**2)))
    ''')
    markdown(r'''
    ## Block 5 — Inspect the exact equations being solved

    Snell's law provides physical slopes $u=∇F$ and $v=∇G$.
    The chain rule converts them to derivatives in incoming-ray coordinates:

    $$e_{fx}=\frac{u_1+(f-h_1)u\cdot\partial_xs}{1-u\cdot s},\qquad
    e_{fy}=\frac{u_2+(f-h_1)u\cdot\partial_ys}{1-u\cdot s},$$
    $$e_{gx}=\frac{v\cdot\partial_xq+(g-h_2)v\cdot\partial_xt}{1-v\cdot t},\qquad
    e_{gy}=\frac{v\cdot\partial_yq+(g-h_2)v\cdot\partial_yt}{1-v\cdot t}.$$

    These are Eqs. (4)–(7). The implementation shown below is the actual function
    used by the solver. `normal1 = n1*w_in - n2*w2` and
    `normal2 = n2*w2 - n3*w_out` are parallel to the interface normals;
    negative XY components divided by the Z component give $u$ and $v$.
    ''')
    code('''
    # Display the live source so the formulas are visible without duplicating them.
    display(Code(inspect.getsource(PaperSystem.rhs), language="python"))
    ''')
    markdown(r'''
    ## Block 6 — Minimize the joint PDE cost

    $$E=\frac{1}{|\Omega|}\int_\Omega
    \left[(f_x-e_{fx})^2+(f_y-e_{fy})^2+
    (g_x-e_{gx})^2+(g_y-e_{gy})^2\right]\,dx\,dy.$$

    Second-order differences and trapezoidal area weights discretize this energy.
    Sparse nonlinear least squares fits both surfaces jointly. The two anchors
    are fixed exactly by removing them from the unknown vector. No additional
    boundary heights or curvature coefficients are supplied. A 21×21 grid has
    880 unknown values and 1764 residuals.
    ''')
    code('''
    result = reconstruct(data, anchors, f_bounds=F_BOUNDS_MM, g_bounds=G_BOUNDS_MM)
    print(json.dumps(result.diagnostics, indent=2))
    # p1 and p2 are the recovered PHYSICAL meshes for plotting or export.
    np.savez_compressed(OUTPUT / f"{CASE}_reconstruction.npz",
                        f=result.f, g=result.g, p1=result.p1, p2=result.p2)
    ''')
    markdown('''
    ## Block 7 — Compare the reconstruction with the known physical graphs

    Error is reconstructed Z minus known Z at each reconstructed physical XY
    coordinate. The Snell diagnostics separately compare refracted directions
    using normals obtained from the recovered meshes. Inspect convergence and
    whether any bounds are active before interpreting a solution.

    Clean synthetic errors reflect finite-difference discretization and optimizer
    tolerance. They are **not** a claim of nanometre accuracy on real laminated glass.
    ''')
    code('''
    errors = surface_errors(result, front, back)
    metrics = {**result.diagnostics, **errors, "ray_count": int(np.prod(data.shape)),
               "case": CASE, "grid_size": GRID_SIZE, "anchors": asdict(anchors),
               "front_truth": asdict(front), "back_truth": asdict(back),
               "f_bounds_mm": F_BOUNDS_MM, "g_bounds_mm": G_BOUNDS_MM,
               "spot_noise_mm": 0.0, "direction_noise_rad": 0.0, "seed": 7}
    for interface in ["front", "back"]:
        print(f"{interface.capitalize()} height RMSE: "
              f"{errors[interface + '_height_rmse_mm']*1000:.6f} micrometres")
    (OUTPUT / f"{CASE}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    ''')
    markdown('''
    ## Block 8 — Rotate the reconstructed glass freely

    **Drag** to orbit, **scroll** to zoom, and use the toolbar to pan or reset.
    The left view shows both reconstructed interfaces, known wireframes, and sample
    ray paths. The right view colors each physical surface by its height error.
    Both views use the same physical scale on X, Y, and Z; Z is not exaggerated.
    Toggle items by clicking the legend. The two view cameras operate independently.
    ''')
    code('''
    figure = make_figure(data, result, front, back,
                         title=f"{CASE.capitalize()} glass | n = 1.00 / 1.52 / 1.00")
    html_path = OUTPUT / f"{CASE}_reconstruction.html"
    write_html(figure, html_path, metrics, CASE)
    figure.show(config={"scrollZoom": True, "displaylogo": False})
    print("Standalone offline plot:", html_path.resolve())
    ''')
    markdown('''
    ## Block 9 — Check that finer sampling reduces numerical error

    This repeats the same **curved** experiment at three grid spacings. We expect
    roughly second-order convergence for clean smooth data. Grid refinement is a
    numerical accuracy check, not a substitute for uncertainty analysis.
    ''')
    code('''
    convergence_front, convergence_back = demo_surfaces("curved")
    convergence = []
    for count in [11, 21, 41]:
        sample, calibration, _ = simulate(convergence_front, convergence_back, grid_size=count)
        recovered = reconstruct(sample, calibration, f_bounds=F_BOUNDS_MM, g_bounds=G_BOUNDS_MM)
        error = surface_errors(recovered, convergence_front, convergence_back)
        row = {"grid": count, "spacing_mm": 20/(count-1), **error}
        convergence.append(row)
        print(f"{count} x {count}: front {error['front_height_rmse_mm']*1e6:.4f} nm, "
              f"back {error['back_height_rmse_mm']*1e6:.4f} nm")
    (OUTPUT / "grid_convergence.json").write_text(json.dumps(convergence, indent=2), encoding="utf-8")
    ''')
    markdown('''
    ## Block 10 — Repeat with measurement noise and with two planes

    We add 2 µm Gaussian noise to each exit-position component and 20 µrad-scale
    component perturbations to exit directions before normalization, with seed 7.
    Input calibration and the reference heights remain exact. Differentiation of
    noisy measurements affects the PDE residual; least squares finds a compromise.
    These results describe this particular noise model, not an instrument specification.

    The last call also exports the simple two-plane example, so both demos are
    available even when the main notebook uses curved surfaces.
    ''')
    code('''
    noisy_run = run_demo("curved", GRID_SIZE, OUTPUT,
                         spot_noise_mm=0.002, direction_noise_rad=0.00002)
    noisy_metrics = noisy_run[4]
    print("Noisy curved sample:")
    for interface in ["front", "back"]:
        print(f"  {interface}: {noisy_metrics[interface + '_height_rmse_mm']*1000:.3f} micrometres RMS")
    planes_run = run_demo("planes", GRID_SIZE, OUTPUT)
    print("Additional plots:")
    print((OUTPUT / "curved_noisy_reconstruction.html").resolve())
    print((OUTPUT / "planes_reconstruction.html").resolve())
    ''')
    markdown('''
    ## Block 11 — Load measured ray data for a future experiment

    `RayData` stores only external measurements and optical constants. The code
    below demonstrates the format by reloading our exported measurements. Replace
    the filename and supply independently calibrated anchors for an actual sample.
    Ray correspondence and the rectangular entrance-ray grid must be preserved.

    A single downstream screen generally cannot provide both outgoing position
    and direction; two calibrated downstream screens/planes can provide that
    direction by tracking the same ray. If your only extra datum is thickness at
    one common XY point, that is a different constraint from the two ray-height
    anchors used here and needs an extension of this implementation.
    ''')
    code('''
    reloaded = RayData.load(OUTPUT / f"{CASE}_measurements.npz")
    print("Loaded only these data fields:", list(reloaded.__dict__))
    # Real-data usage after filling measured arrays and calibration:
    # measured = RayData(x, y, w_in, exit_xy, w_out, h1, h2, 1.0, 1.52, 1.0)
    # reference = Anchors(row, col, f_measured, g_measured)
    # recovered = reconstruct(measured, reference, f_bounds=(...), g_bounds=(...))
    ''')
    markdown('''
    ## Block 12 — Scope and identifiability

    A parallel glass slab at normal incidence leaves external rays unchanged, so
    those rays cannot reveal its thickness or axial position. The anchors provide
    necessary geometric information in this example. They do not guarantee global
    uniqueness for every optical setup, and the optimizer is local.

    The implementation supports smooth graphs in disjoint z brackets, forward
    transmission, and nonsingular ray maps. It rejects total internal reflection
    and detected local folds. General overhangs, missing rays, unknown indices,
    uncertain anchors, multilayer refraction, and strong measurement noise need
    additional modeling. `g-f` compares heights on the same ray; it is generally
    not the material thickness at a common physical XY point.

    Run `python -m unittest discover -s tests -v` to check physics, planes, curved
    refinement, noisy data, and an intentionally ambiguous normal-incidence slab.
    ''')
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata['kernelspec'] = {'display_name': 'Python (project .venv)',
                                      'language': 'python', 'name': 'python3'}
    notebook.metadata['language_info'] = {'name': 'python', 'version': '3.13'}
    root = Path(__file__).resolve().parents[1]
    nbf.write(notebook, root/'geometria_optica.ipynb')
    print(f"Wrote {len(cells)} documented notebook cells.")


if __name__ == '__main__':
    build()
