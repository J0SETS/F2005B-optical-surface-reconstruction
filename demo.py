"""Run the two-interface demonstration and export a self-contained interactive plot.

Usage: python demo.py --case curved --grid 21
       python demo.py --case planes --grid 21
       python demo.py --spot-noise-mm 0.002 --direction-noise-rad 0.00002
"""

# %% Block 1 - Imports and the known demonstration geometries
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from reconstruction import QuadraticSurface, reconstruct, simulate, surface_errors


def demo_surfaces(case='curved'):
    """Two simple, distinct interfaces around z=10 and z=16 mm."""
    if case == 'curved':
        return (QuadraticSurface(10, .015, -.010, .002, 0, .0015),
                QuadraticSurface(16, -.010, .018, .001, 0, .002))
    if case == 'planes':
        return QuadraticSurface(10, .025, -.015), QuadraticSurface(16, -.012, .020)
    raise ValueError("case must be 'curved' or 'planes'.")


# %% Block 2 - Interactive 3-D view and error maps in physical coordinates
def make_figure(data, result, front, back, title='Two-interface reconstruction'):
    """Plot both physical meshes, truth wireframes, sample rays, and height errors.

    Plotly preserves metric aspect ratio: one millimetre has the same length on
    all three axes. Truth is evaluated at reconstructed physical XY coordinates.
    """
    fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'scene'}, {'type': 'scene'}]],
                        column_widths=[.62, .38], horizontal_spacing=.025,
                        subplot_titles=['Recovered geometry + known truth', 'Height error on each interface'])
    colors = ['#16878a', '#e49b36']
    errors = [points[..., 2]-surface.height(points[..., 0], points[..., 1])
              for points, surface in [(result.p1, front), (result.p2, back)]]
    # Display nanometres for the clean demo and micrometres when noise dominates.
    max_mm = max(float(np.max(abs(e))) for e in errors)
    scale, unit_label = (1e6, 'nm') if max_mm < .001 else (1e3, 'µm')
    color_limit = max(max_mm*scale, 1e-6)
    for k, (name, points, surface, color, error) in enumerate(zip(
        ['Front', 'Back'], [result.p1, result.p2], [front, back], colors, errors)):
        X, Y, Z = np.moveaxis(points, -1, 0)
        fig.add_trace(go.Surface(x=X, y=Y, z=Z, name=f'{name}: reconstructed',
                                 colorscale=[[0, color], [1, color]], showscale=False,
                                 opacity=.78, showlegend=True,
                                 hovertemplate=f'{name}<br>X=%{{x:.3f}} mm<br>Y=%{{y:.3f}} mm<br>Z=%{{z:.6f}} mm<extra></extra>'), 1, 1)
        wire_x, wire_y, wire_z = [], [], []
        step = max(1, len(data.x)//6)
        truth_z = surface.height(X, Y)
        for axis in (0, 1):
            for i in range(0, points.shape[axis], step):
                section = (i, slice(None)) if axis == 0 else (slice(None), i)
                wire_x.extend(X[section].tolist()+[None])
                wire_y.extend(Y[section].tolist()+[None])
                wire_z.extend(truth_z[section].tolist()+[None])
        fig.add_trace(go.Scatter3d(x=wire_x, y=wire_y, z=wire_z, mode='lines',
                                   name=f'{name}: known truth',
                                   line={'color': '#213547', 'width': 2}, hoverinfo='skip'), 1, 1)
        fig.add_trace(go.Surface(x=X, y=Y, z=Z, surfacecolor=error*scale,
                                 colorscale='RdBu_r', cmin=-color_limit, cmax=color_limit,
                                 showscale=(k == 1), name=f'{name} error',
                                 colorbar={'title': f'Error ({unit_label})', 'x': 1.01, 'len': .65},
                                 customdata=error*scale,
                                 hovertemplate=f'{name}<br>X=%{{x:.3f}} mm<br>Y=%{{y:.3f}} mm<br>Error=%{{customdata:.3f}} {unit_label}<extra></extra>'), 1, 2)
    # Show short incoming/outgoing segments so the glass occupies most of the view.
    ray_x, ray_y, ray_z = [], [], []
    for row in np.linspace(1, len(data.y)-2, 3, dtype=int):
        for col in np.linspace(1, len(data.x)-2, 3, dtype=int):
            a, b = result.p1[row, col], result.p2[row, col]
            ray = np.array([a-3*data.w_in[row, col], a, b, b+3*data.w_out[row, col]])
            ray_x.extend(ray[:, 0].tolist()+[None])
            ray_y.extend(ray[:, 1].tolist()+[None])
            ray_z.extend(ray[:, 2].tolist()+[None])
    fig.add_trace(go.Scatter3d(x=ray_x, y=ray_y, z=ray_z, mode='lines', name='Matched rays',
                               line={'color': '#a04d9a', 'width': 3}, hoverinfo='skip'), 1, 1)
    scene = dict(xaxis_title='X (mm)', yaxis_title='Y (mm)', zaxis_title='Z (mm)',
                 aspectmode='data', dragmode='orbit',
                 camera={'eye': {'x': 1.45, 'y': -1.75, 'z': .95}})
    # Pull the camera back in the narrower error panel to keep all axes visible.
    error_scene = {**scene, 'camera': {'eye': {'x': 2.2, 'y': -2.65, 'z': 1.45}}}
    fig.update_layout(title={'text': title, 'x': .03}, template='plotly_white',
                      scene=scene, scene2=error_scene, height=720,
                      margin={'l': 0, 'r': 60, 't': 100, 'b': 75},
                      legend={'orientation': 'h', 'y': -.04, 'x': .02},
                      font={'family': 'Arial, sans-serif', 'size': 12})
    return fig


# %% Block 3 - Portable HTML export (Plotly JavaScript is embedded; no internet)
def write_html(fig, path, metrics, case):
    """Write one file that opens locally and supports rotation, pan, zoom, export."""
    rmse = ' / '.join(f"{metrics[f'{s}_height_rmse_mm']*1000:.5f}" for s in ('front', 'back'))
    plot = fig.to_html(full_html=False, include_plotlyjs=True,
                       config={'scrollZoom': True, 'displaylogo': False, 'responsive': True})
    page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Glass reconstruction · {case}</title>
<style>body{{margin:0;background:#f3f6f8;color:#213547;font:15px Arial,sans-serif}}
header,footer{{max-width:1400px;margin:auto;padding:22px 32px}}h1{{margin:6px 0 12px;font-size:27px}}
.eyebrow{{color:#16878a;font-size:12px;letter-spacing:2px;font-weight:bold}}
.metrics{{display:flex;flex-wrap:wrap;gap:12px;margin-top:18px}}
.card{{background:white;border:1px solid #dde5e9;border-radius:8px;padding:12px 18px}}
.plot{{background:white;margin:0 18px;border-radius:12px}}p{{line-height:1.55;margin:8px 0}}
footer{{font-size:13px;color:#536778}}code{{background:#e9eff2;padding:2px 5px}}</style></head>
<body><header><div class="eyebrow">OPTICAL SURFACE RECONSTRUCTION</div>
<h1>Two interfaces, one effective glass index</h1>
<p>Air → glass → air · n = 1.00 / 1.52 / 1.00 · {case} synthetic sample</p>
<p><b>Drag</b> either 3D plot to rotate. <b>Scroll</b> to zoom. Use the toolbar to pan or reset.
Click legend entries to hide or show geometry.</p>
<div class="metrics"><div class="card"><b>Height RMSE, front / back</b><br>{rmse} µm</div>
<div class="card"><b>Matched rays</b><br>{metrics['ray_count']}</div>
<div class="card"><b>PDE slope RMS</b><br>{metrics['pde_slope_rms']:.3e}</div>
<div class="card"><b>Known calibration</b><br>One height on each interface</div></div></header>
<main class="plot">{plot}</main><footer><p>Colored surfaces: reconstructed geometry.
Dark wireframes: known synthetic geometry. The error view colors the actual geometry;
its vertical axis is physical Z, not error. The clean-demo error reflects numerical
discretization, not achievable measurement accuracy.</p>
<p>Method: Rubinstein &amp; Wolansky (2001), Eqs. (4)–(7), solved jointly by least squares.
The two reference heights and broad height brackets are supplied calibration inputs.
See <code>geometria_optica.ipynb</code> for numbered steps and <code>README.md</code> for equations.</p>
</footer></body></html>'''
    Path(path).write_text(page, encoding='utf-8')


# %% Block 4 - Reproducible run and saved measurements/results
def run_demo(case='curved', grid=21, output='output', spot_noise_mm=0.0, direction_noise_rad=0.0):
    """Generate data, reconstruct, validate, and export all reproducible artifacts."""
    folder = Path(output)
    folder.mkdir(parents=True, exist_ok=True)
    front, back = demo_surfaces(case)
    data, anchors, truth = simulate(front, back, grid_size=grid,
                                    spot_noise_mm=spot_noise_mm, direction_noise_rad=direction_noise_rad)
    result = reconstruct(data, anchors, f_bounds=(5, 13), g_bounds=(13.5, 23))
    metrics = {**result.diagnostics, **surface_errors(result, front, back),
               'ray_count': grid*grid, 'grid_size': grid, 'case': case,
               'spot_noise_mm': spot_noise_mm, 'direction_noise_rad': direction_noise_rad,
               'seed': 7, 'anchors': asdict(anchors),
               'f_bounds_mm': [5, 13], 'g_bounds_mm': [13.5, 23],
               'front_truth': asdict(front), 'back_truth': asdict(back)}
    suffix = '_noisy' if spot_noise_mm or direction_noise_rad else ''
    stem = f'{case}{suffix}'
    data.save(folder/f'{stem}_measurements.npz')
    np.savez_compressed(folder/f'{stem}_reconstruction.npz', f=result.f, g=result.g,
                        p1=result.p1, p2=result.p2)
    (folder/f'{stem}_metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    fig = make_figure(data, result, front, back, f'{case.capitalize()} glass | 1.00 → 1.52 → 1.00')
    write_html(fig, folder/f'{stem}_reconstruction.html', metrics, case)
    return data, anchors, truth, result, metrics, fig


def main():
    """Command-line entry point; no browser is launched automatically."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=['curved', 'planes'], default='curved')
    parser.add_argument('--grid', type=int, default=21)
    parser.add_argument('--output', default='output')
    parser.add_argument('--spot-noise-mm', type=float, default=0.0)
    parser.add_argument('--direction-noise-rad', type=float, default=0.0)
    args = parser.parse_args()
    *_, metrics, _ = run_demo(args.case, args.grid, args.output,
                             args.spot_noise_mm, args.direction_noise_rad)
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
