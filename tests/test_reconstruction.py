"""Physics and numerical regression tests; run with python -m unittest discover -s tests -v."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from demo import demo_surfaces
from reconstruction import (Anchors, PaperSystem, RayData, derivative_1d, reconstruct,
                            refract, simulate, surface_errors)


class ReconstructionTests(unittest.TestCase):
    """Check known physical outcomes, discretization convergence, and data contracts."""

    def solve(self, data, anchors):
        """Use the same conservative height brackets for all synthetic samples."""
        return reconstruct(data, anchors, f_bounds=(5, 13), g_bounds=(13.5, 23))

    def test_nonuniform_quadratic_derivative(self):
        """A three-point derivative must differentiate quadratics exactly, including edges."""
        x = np.array([-3., -1.1, 0., .7, 2.5])
        np.testing.assert_allclose(derivative_1d(x) @ (2*x*x-3*x+4), 4*x-3, atol=1e-13)

    def test_snell_angle_and_total_internal_reflection(self):
        """Compare vector refraction to scalar Snell's law at 30 degrees."""
        theta = np.deg2rad(30)
        direction = np.array([np.sin(theta), 0., np.cos(theta)])
        outgoing = refract(direction, [0., 0., 1.], 1., 1.52)
        self.assertAlmostEqual(outgoing[0], np.sin(theta)/1.52, places=14)
        np.testing.assert_allclose(refract(outgoing, [0, 0, 1], 1.52, 1), direction, atol=1e-14)
        with self.assertRaisesRegex(ValueError, 'Total internal reflection'):
            refract([np.sin(np.deg2rad(60)), 0, .5], [0, 0, 1], 1.52, 1)

    def test_two_tilted_planes(self):
        """Both interface tilts and positions must be recovered without fitting plane coefficients."""
        front, back = demo_surfaces('planes')
        data, anchors, _ = simulate(front, back, grid_size=13)
        result = self.solve(data, anchors)
        metrics = surface_errors(result, front, back)
        self.assertLess(max(metrics.values()), 2e-7)
        self.assertEqual(result.f[anchors.row, anchors.col], anchors.f)
        self.assertEqual(result.g[anchors.row, anchors.col], anchors.g)
        self.assertEqual(result.diagnostics['active_bounds'], 0)

    def test_curved_grid_refinement(self):
        """Halving spacing should reduce second-order discretization error by roughly four."""
        front, back = demo_surfaces('curved')
        errors = []
        for grid in [11, 21]:
            data, anchors, _ = simulate(front, back, grid_size=grid)
            result = self.solve(data, anchors)
            metrics = surface_errors(result, front, back)
            errors.append(metrics['front_height_rmse_mm']+metrics['back_height_rmse_mm'])
        self.assertLess(errors[1], .4*errors[0])
        self.assertLess(errors[1], 2e-5)

    def test_normal_incidence_cannot_identify_slab_heights(self):
        """Identical measured rays admit multiple parallel slabs: anchors carry real information."""
        x = y = np.linspace(-3, 3, 5)
        xy = np.stack(np.meshgrid(x, y), axis=-1)
        w = np.broadcast_to([0., 0., 1.], (5, 5, 3)).copy()
        data = RayData(x, y, w, xy, w)
        for f, g in [(9., 16.), (11., 19.)]:
            result = self.solve(data, Anchors(2, 2, f, g))
            np.testing.assert_allclose(result.f, f, atol=1e-12)
            np.testing.assert_allclose(result.g, g, atol=1e-12)
            self.assertLess(result.diagnostics['energy'], 1e-25)

    def test_nonzero_entrance_plane(self):
        """Translating the coordinate origin must preserve the paper's RHS (f-h1 matters)."""
        front, back = demo_surfaces()
        data, anchors, truth = simulate(front, back, grid_size=9)
        f, g = truth['p1'][..., 2], truth['p2'][..., 2]
        original = PaperSystem(data).rhs(f, g)
        shifted = RayData(data.x, data.y, data.w_in, data.exit_xy, data.w_out, h1=4, h2=39)
        translated = PaperSystem(shifted).rhs(f+4, g+4)
        np.testing.assert_allclose(original, translated, atol=3e-15)

    def test_measurement_noise(self):
        """A seeded 2-micron spot-noise example remains usable but cannot fit data exactly."""
        front, back = demo_surfaces()
        data, anchors, _ = simulate(front, back, grid_size=13, spot_noise_mm=.002,
                                     direction_noise_rad=2e-5)
        result = self.solve(data, anchors)
        metrics = surface_errors(result, front, back)
        self.assertGreater(result.diagnostics['energy'], 1e-9)
        self.assertLess(metrics['front_height_rmse_mm'], .02)
        self.assertLess(metrics['back_height_rmse_mm'], .02)

    def test_measurement_roundtrip_and_bad_input(self):
        """Export/import must preserve external data; malformed measurements fail explicitly."""
        data, anchors, _ = simulate(*demo_surfaces(), grid_size=7)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'data.npz'
            data.save(path)
            loaded = RayData.load(path)
            np.testing.assert_array_equal(loaded.exit_xy, data.exit_xy)
            self.assertEqual(loaded.n2, 1.52)
        with self.assertRaisesRegex(ValueError, 'disjoint'):
            reconstruct(data, anchors, f_bounds=(5, 18), g_bounds=(13, 23))
        with self.assertRaisesRegex(ValueError, 'increasing'):
            RayData(data.x[::-1], data.y, data.w_in, data.exit_xy, data.w_out)
        with self.assertRaisesRegex(ValueError, 'positive z'):
            RayData(data.x, data.y, -data.w_in, data.exit_xy, data.w_out)


if __name__ == '__main__':
    unittest.main()
