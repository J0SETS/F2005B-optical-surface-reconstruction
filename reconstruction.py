"""Two-interface reconstruction from Rubinstein & Wolansky (2001), Eqs. (4)-(7).

All lengths use one consistent unit (millimetres in the demo). Arrays have shape
(ny, nx); x labels columns, y labels rows. The unknowns f and g are HEIGHTS ALONG
MATCHED RAYS, not height maps evaluated at the same physical (X, Y) coordinates.
See README.md and geometria_optica.ipynb for the derivation and measurement model.
"""

# %% Block 1 - Data containers and validation
from dataclasses import dataclass
import numpy as np
from scipy import sparse
from scipy.optimize import least_squares


def unit(v):
    """Normalize vectors along their last axis; reject zero/nonfinite vectors."""
    v = np.asarray(v, dtype=float)
    length = np.linalg.norm(v, axis=-1, keepdims=True)
    if not np.all(np.isfinite(v)) or np.any(length <= 1e-14):
        raise ValueError("Directions must be finite and nonzero.")
    return v / length


@dataclass
class RayData:
    """External observations: rectangular entrance grid and matched exit rays.

    w_in/w_out: (ny,nx,3) forward unit directions (normalized on construction).
    exit_xy: (ny,nx,2) spot coordinates on z=h2, indexed by ENTRANCE ray labels.
    x/y: increasing 1-D coordinates of the rays on z=h1, not surface coordinates.
    """

    x: np.ndarray
    y: np.ndarray
    w_in: np.ndarray
    exit_xy: np.ndarray
    w_out: np.ndarray
    h1: float = 0.0
    h2: float = 35.0
    n1: float = 1.0
    n2: float = 1.52
    n3: float = 1.0

    def __post_init__(self):
        self.x, self.y = np.asarray(self.x, float), np.asarray(self.y, float)
        for coordinates in (self.x, self.y):
            if (coordinates.ndim != 1 or len(coordinates) < 3
                    or not np.all(np.isfinite(coordinates))
                    or np.any(np.diff(coordinates) <= 0)):
                raise ValueError("x and y must be increasing finite 1-D grids of length >=3.")
        self.w_in, self.w_out = unit(self.w_in), unit(self.w_out)
        self.exit_xy = np.asarray(self.exit_xy, float)
        shape = (len(self.y), len(self.x))
        if self.w_in.shape != shape + (3,) or self.w_out.shape != shape + (3,):
            raise ValueError("Direction arrays must have shape (ny,nx,3).")
        if self.exit_xy.shape != shape + (2,) or not np.all(np.isfinite(self.exit_xy)):
            raise ValueError("exit_xy must be a finite array with shape (ny,nx,2).")
        if (not np.all(np.isfinite([self.h1, self.h2, self.n1, self.n2, self.n3]))
                or self.h2 <= self.h1 or min(self.n1, self.n2, self.n3) <= 0):
            raise ValueError("Require h1<h2 and positive finite refractive indices.")
        if min(np.min(self.w_in[..., 2]), np.min(self.w_out[..., 2])) <= 1e-6:
            raise ValueError("This implementation requires forward rays with positive z direction.")
        if abs(self.n1-self.n2) < 1e-8 or abs(self.n2-self.n3) < 1e-8:
            raise ValueError("Each interface needs a refractive-index contrast.")

    @property
    def shape(self):
        return (len(self.y), len(self.x))

    def save(self, path):
        """Save only externally measurable data, with no hidden surface truth."""
        np.savez_compressed(path, **self.__dict__)

    @classmethod
    def load(cls, path):
        """Load an NPZ measurement file produced by save()."""
        with np.load(path, allow_pickle=False) as data:
            values = {key: data[key] for key in data.files}
        for key in ('h1', 'h2', 'n1', 'n2', 'n3'):
            values[key] = float(values[key])
        return cls(**values)


@dataclass(frozen=True)
class Anchors:
    """Known f and g heights on one matched ray (row, col).

    These are two explicit independent calibration inputs. g is the exit-surface
    height on that ray, generally at a different physical XY point from f.
    """

    row: int
    col: int
    f: float
    g: float


# %% Block 2 - Second-order finite differences and area weights
def derivative_1d(coordinates):
    """Three-point derivative matrix, including one-sided boundary stencils.

    Solving the local polynomial moment equations supports nonuniform grids.
    """
    coordinates = np.asarray(coordinates, float)
    rows, cols, values = [], [], []
    n = len(coordinates)
    for i in range(n):
        indices = np.arange(max(0, min(i-1, n-3)), max(0, min(i-1, n-3)) + 3)
        offsets = coordinates[indices] - coordinates[i]
        weights = np.linalg.solve(np.vstack([offsets**k for k in range(3)]), [0, 1, 0])
        rows.extend([i]*3)
        cols.extend(indices)
        values.extend(weights)
    return sparse.csr_matrix((values, (rows, cols)), shape=(n, n))


def trapezoid_weights(coordinates):
    """Positive nodal quadrature weights for the integral of the squared PDE error."""
    d = np.diff(coordinates)
    return np.r_[d[0]/2, (d[:-1]+d[1:])/2, d[-1]/2]


# %% Block 3 - Ray geometry and the paper's four PDE right-hand sides
class PaperSystem:
    """Discretization of Eqs. (4)-(7), including measured-field derivatives."""

    def __init__(self, data: RayData):
        self.data = data
        ny, nx = data.shape
        self.size = nx*ny
        self.dx = sparse.kron(sparse.eye(ny), derivative_1d(data.x), format='csr')
        self.dy = sparse.kron(derivative_1d(data.y), sparse.eye(nx), format='csr')
        self.xy = np.stack(np.meshgrid(data.x, data.y), axis=-1).reshape(-1, 2)
        self.win = data.w_in.reshape(-1, 3)
        self.wout = data.w_out.reshape(-1, 3)
        self.spots = data.exit_xy.reshape(-1, 2)
        self.s = self.win[:, :2] / self.win[:, 2, None]
        self.t = self.wout[:, :2] / self.wout[:, 2, None]
        self.sx, self.sy = self.dx @ self.s, self.dy @ self.s
        self.tx, self.ty = self.dx @ self.t, self.dy @ self.t
        self.qx, self.qy = self.dx @ self.spots, self.dy @ self.spots
        area = np.outer(trapezoid_weights(data.y), trapezoid_weights(data.x)).ravel()
        self.weights = np.sqrt(area / area.sum())

    def geometry(self, f, g):
        """Recover physical points P1, P2 and the unknown internal unit ray."""
        f, g = np.asarray(f).ravel(), np.asarray(g).ravel()
        p1 = np.column_stack([self.xy + self.s*(f-self.data.h1)[:, None], f])
        p2 = np.column_stack([self.spots + self.t*(g-self.data.h2)[:, None], g])
        if np.any(g <= f):
            raise ValueError("Exit height must exceed entrance height on every ray.")
        return p1, p2, unit(p2-p1)

    def rhs(self, f, g):
        """Return e_fx,e_fy,e_gx,e_gy; each has one entry per input ray.

        u and v are slopes with respect to PHYSICAL coordinates. Chain rules
        convert these slopes into derivatives with respect to input ray labels.
        No paraxial approximation is used.
        """
        f, g = np.asarray(f).ravel(), np.asarray(g).ravel()
        _, _, internal = self.geometry(f, g)
        d = self.data
        normal1 = d.n1*self.win - d.n2*internal
        normal2 = d.n2*internal - d.n3*self.wout
        if min(np.min(abs(normal1[:, 2])), np.min(abs(normal2[:, 2]))) < 1e-9:
            raise ValueError("Snell slope denominator is singular; choose a different graph axis.")
        u = -normal1[:, :2]/normal1[:, 2, None]
        v = -normal2[:, :2]/normal2[:, 2, None]
        den_f = 1 - np.sum(u*self.s, axis=1)
        den_g = 1 - np.sum(v*self.t, axis=1)
        if min(np.min(abs(den_f)), np.min(abs(den_g))) < 1e-8:
            raise ValueError("Ray-to-surface parameterization is singular (grazing/folded rays).")
        dot = lambda a, b: np.sum(a*b, axis=1)
        ex_f = (u[:, 0] + (f-d.h1)*dot(u, self.sx))/den_f
        ey_f = (u[:, 1] + (f-d.h1)*dot(u, self.sy))/den_f
        ex_g = (dot(v, self.qx) + (g-d.h2)*dot(v, self.tx))/den_g
        ey_g = (dot(v, self.qy) + (g-d.h2)*dot(v, self.ty))/den_g
        return ex_f, ey_f, ex_g, ey_g

    def residual_fields(self, f, g):
        """The four unweighted PDE residuals (dimensionless slopes)."""
        f, g = np.asarray(f).ravel(), np.asarray(g).ravel()
        target = self.rhs(f, g)
        measured = (self.dx @ f, self.dy @ f, self.dx @ g, self.dy @ g)
        return np.stack([a-b for a, b in zip(measured, target)])


# %% Block 4 - Joint nonlinear least squares with exact height anchors
@dataclass
class Reconstruction:
    """Recovered ray-indexed heights, physical meshes, and numerical diagnostics."""

    f: np.ndarray
    g: np.ndarray
    p1: np.ndarray
    p2: np.ndarray
    optimizer: object
    diagnostics: dict


def reconstruct(data: RayData, anchors: Anchors, *, f_bounds, g_bounds,
                max_nfev=400, verbose=0):
    """Minimize the paper's PDE energy for BOTH unknown surfaces together.

    f_bounds and g_bounds are conservative, disjoint height brackets supplied by
    the experimenter: h1 < f_low < f_high < g_low < g_high < h2. This implementation
    is intended for gently sloped separated sheets, not arbitrary folded optics.
    Fixed anchor entries are removed from the optimizer, not softly penalized.
    The initial guess is two constant height fields, using only those anchors.
    """
    ny, nx = data.shape
    if not (0 <= anchors.row < ny and 0 <= anchors.col < nx):
        raise ValueError("Anchor index is outside the input grid.")
    fl, fu = map(float, f_bounds)
    gl, gu = map(float, g_bounds)
    if not data.h1 < fl < anchors.f < fu < gl < anchors.g < gu < data.h2:
        raise ValueError("Require disjoint surface brackets inside measurement planes, with interior anchors.")
    system = PaperSystem(data)
    m = system.size
    index = anchors.row*nx + anchors.col
    fixed = np.array([index, m+index])
    free = np.setdiff1d(np.arange(2*m), fixed)
    initial = np.r_[np.full(m, anchors.f), np.full(m, anchors.g)]

    def unpack(parameters):
        fields = initial.copy()
        fields[free] = parameters
        return fields[:m], fields[m:]

    def residual(parameters):
        f, g = unpack(parameters)
        return (system.residual_fields(f, g)*system.weights).ravel()

    # Each PDE residual depends on a three-point derivative stencil on its own
    # surface and on BOTH local heights through the internal ray direction.
    identity = sparse.eye(m, format='csr')
    sx = abs(system.dx) + identity
    sy = abs(system.dy) + identity
    pattern = sparse.bmat([[sx, identity], [sy, identity],
                           [identity, sx], [identity, sy]], format='csr')[:, free]
    lower = np.r_[np.full(m, fl), np.full(m, gl)][free]
    upper = np.r_[np.full(m, fu), np.full(m, gu)][free]
    fit = least_squares(residual, initial[free], bounds=(lower, upper),
                        jac_sparsity=pattern, method='trf', tr_solver='lsmr',
                        tr_options={'atol': 1e-10, 'btol': 1e-10},
                        x_scale='jac', ftol=1e-11, xtol=1e-11, gtol=1e-11,
                        max_nfev=max_nfev, verbose=verbose)
    f, g = unpack(fit.x)
    p1, p2, internal = system.geometry(f, g)
    diagnostics = {
        'success': bool(fit.success), 'message': fit.message,
        'function_evaluations': int(fit.nfev),
        'energy': float(2*fit.cost),  # mean-area integral of all four residuals
        'pde_slope_rms': float(np.sqrt(np.mean(system.residual_fields(f, g)**2))),
        'optimality': float(fit.optimality),
        'active_bounds': int(np.count_nonzero(fit.active_mask)),
        'minimum_ray_z_separation_mm': float(np.min(g-f)),
    }
    # Independent vector form of Snell's law, using normals from mesh tangents.
    # This also catches a wrong physical-coordinate mapping in the plotting code.
    for name, points, incoming, outgoing, n_before, n_after in (
        ('front', p1, system.win, internal, data.n1, data.n2),
        ('back', p2, internal, system.wout, data.n2, data.n3),
    ):
        normal = unit(np.cross(system.dx @ points, system.dy @ points))
        predicted = refract(incoming, normal, n_before, n_after)
        angle = 2*np.arcsin(np.clip(np.linalg.norm(predicted-outgoing, axis=1)/2, 0, 1))
        diagnostics[f'{name}_snell_angle_rms_mrad'] = float(1000*np.sqrt(np.mean(angle**2)))
        jac_xy = ((system.dx @ points[:, 0])*(system.dy @ points[:, 1])
                  - (system.dy @ points[:, 0])*(system.dx @ points[:, 1]))
        diagnostics[f'{name}_minimum_xy_jacobian'] = float(jac_xy.min())
        if np.any(jac_xy <= 0):
            raise RuntimeError("Recovered mesh folds in XY; graph reconstruction is invalid.")
    if not fit.success:
        raise RuntimeError(f"Reconstruction failed to converge: {fit.message}")
    return Reconstruction(f.reshape(data.shape), g.reshape(data.shape),
                          p1.reshape(data.shape+(3,)), p2.reshape(data.shape+(3,)),
                          fit, diagnostics)


# %% Block 5 - Independent forward ray tracer for synthetic measurements
@dataclass(frozen=True)
class QuadraticSurface:
    """Physical graph z=z0+sx*X+sy*Y+qxx*X^2+qxy*X*Y+qyy*Y^2.

    z0 is in mm, slopes are dimensionless, and quadratic coefficients are 1/mm.
    This parameterization is used ONLY to generate and evaluate the demo truth.
    The inverse solver has a free height at every grid point on each surface.
    """

    z0: float
    sx: float = 0.0
    sy: float = 0.0
    qxx: float = 0.0
    qxy: float = 0.0
    qyy: float = 0.0

    def height(self, x, y):
        return self.z0+self.sx*x+self.sy*y+self.qxx*x*x+self.qxy*x*y+self.qyy*y*y

    def normal(self, x, y):
        """Unit normal pointing toward positive z."""
        fx = self.sx+2*self.qxx*x+self.qxy*y
        fy = self.sy+self.qxy*x+2*self.qyy*y
        return unit(np.stack([-fx, -fy, np.ones_like(fx)], axis=-1))


def refract(direction, forward_normal, n_before, n_after):
    """Exact vector Snell refraction; reject backward incidence or total reflection.

    The normal points into the transmitted medium (along forward propagation).
    Its tangential component scales by n_before/n_after.
    """
    direction, normal = unit(direction), unit(forward_normal)
    cosine = np.sum(direction*normal, axis=-1, keepdims=True)
    if np.any(cosine <= 0):
        raise ValueError("Ray must meet the forward-facing side of the surface.")
    tangent = (n_before/n_after)*(direction-cosine*normal)
    square = 1-np.sum(tangent*tangent, axis=-1, keepdims=True)
    if np.any(square < -1e-12):
        raise ValueError("Total internal reflection: no transmitted ray exists.")
    return unit(tangent + np.sqrt(np.maximum(square, 0))*normal)


def intersect_quadratic(origin, direction, surface):
    """Nearest strictly forward analytic ray/quadratic intersection.

    The analytic intersection avoids using inverse-solver equations to manufacture
    synthetic measurements. The selected root must precede any other forward root.
    """
    o, w = np.asarray(origin), unit(direction)
    x, y, z = np.moveaxis(o, -1, 0)
    a, b, c = np.moveaxis(w, -1, 0)
    s = surface
    aa = s.qxx*a*a+s.qxy*a*b+s.qyy*b*b
    bb = s.sx*a+s.sy*b+2*s.qxx*x*a+s.qxy*(x*b+y*a)+2*s.qyy*y*b-c
    cc = s.height(x, y)-z
    linear = abs(aa) < 1e-14
    if np.any(linear & (abs(bb) < 1e-14)):
        raise ValueError("Ray is parallel to or lies in the surface.")
    discriminant = bb*bb-4*aa*cc
    if np.any((~linear) & (discriminant < 0)):
        raise ValueError("Ray misses surface.")
    # Stable quadratic roots: q/a and c/q, avoiding subtractive cancellation.
    q = -0.5*(bb+np.copysign(np.sqrt(np.maximum(discriminant, 0)), bb))
    with np.errstate(divide='ignore', invalid='ignore'):
        r1 = np.where(linear, -cc/bb, q/aa)
        r2 = np.where(linear, np.inf, cc/q)
    r1 = np.where(r1 > 1e-9, r1, np.inf)
    r2 = np.where(r2 > 1e-9, r2, np.inf)
    distance = np.minimum(r1, r2)
    if not np.all(np.isfinite(distance)):
        raise ValueError("No strictly forward ray/surface intersection.")
    return o+distance[..., None]*w


def simulate(front, back, *, grid_size=21, half_width=10.0,
             spot_noise_mm=0.0, direction_noise_rad=0.0, seed=7):
    """Generate matched air/glass/air rays, plus separately stored ground truth.

    A weakly divergent, oblique incident beam provides angular diversity.
    Noise is applied ONLY to exit measurements; entrance calibration and anchors
    are treated as exact. Direction noise is per Cartesian perturbation component
    before normalization, not a prescribed polar-angle RMS.
    """
    if (not isinstance(grid_size, (int, np.integer)) or grid_size < 3
            or grid_size % 2 != 1 or not np.isfinite(half_width) or half_width <= 0):
        raise ValueError("Use an odd grid_size >=3 and a positive half_width.")
    if min(spot_noise_mm, direction_noise_rad) < 0:
        raise ValueError("Noise standard deviations cannot be negative.")
    x = y = np.linspace(-half_width, half_width, grid_size)
    X, Y = np.meshgrid(x, y)
    origin = np.stack([X, Y, np.zeros_like(X)], axis=-1)
    incoming = unit(np.stack([0.08+0.002*X, -0.04+0.001*Y, np.ones_like(X)], axis=-1))
    p1 = intersect_quadratic(origin, incoming, front)
    internal = refract(incoming, front.normal(p1[..., 0], p1[..., 1]), 1.0, 1.52)
    p2 = intersect_quadratic(p1, internal, back)
    outgoing = refract(internal, back.normal(p2[..., 0], p2[..., 1]), 1.52, 1.0)
    h2 = 35.0
    screen = p2 + ((h2-p2[..., 2])/outgoing[..., 2])[..., None]*outgoing
    if np.any(p1[..., 2] <= 0) or np.any(p2[..., 2] >= h2):
        raise ValueError("Demo surfaces must lie between z=0 and z=35 mm.")
    if np.any(p2[..., 2] <= p1[..., 2]):
        raise ValueError("Demo surfaces are not ordered along the beam.")
    rng = np.random.default_rng(seed)
    noisy_spots = screen[..., :2]+rng.normal(0, spot_noise_mm, X.shape+(2,))
    noisy_direction = unit(outgoing+rng.normal(0, direction_noise_rad, outgoing.shape))
    observations = RayData(x, y, incoming, noisy_spots, noisy_direction)
    center = grid_size//2
    anchors = Anchors(center, center, float(p1[center, center, 2]), float(p2[center, center, 2]))
    truth = {'p1': p1, 'p2': p2, 'internal': internal, 'screen': screen, 'w_out': outgoing}
    return observations, anchors, truth


# %% Block 6 - Validation against physical ground truth (never used in inversion)
def surface_errors(result, front, back):
    """Compare reconstructed z to exact z at the reconstructed physical XY points."""
    metrics = {}
    for name, points, surface in [('front', result.p1, front), ('back', result.p2, back)]:
        error = points[..., 2]-surface.height(points[..., 0], points[..., 1])
        metrics[f'{name}_height_rmse_mm'] = float(np.sqrt(np.mean(error**2)))
        metrics[f'{name}_height_max_abs_mm'] = float(np.max(abs(error)))
    return metrics
