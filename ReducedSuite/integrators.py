"""
integrators.py
--------------
Numerical integration routines for the NFL self-energy suite.

Implementations:
  1. romberg_1d       : Romberg integration (deterministic, high-order)
  2. simpson_adaptive : Adaptive Simpson with error control
  3. trapezoid_2d     : Tensorized 2D trapezoidal (for grid-based eval)
  4. monte_carlo      : Plain MC + importance sampling
  5. vegas_lite       : Lightweight stratified MC for >2D integrals

All routines accept a callable f(x, *args) where x is a backend tensor
of shape (n,) for 1D or (n, d) for d-dimensional.

Author: NFL suite
"""

from __future__ import annotations
import math
from typing import Callable, Tuple
import numpy as np


# =============================================================================
# 1. Romberg integration (1D)
# =============================================================================

def romberg_1d(f: Callable, a: float, b: float,
               max_levels: int = 12, tol: float = 1e-10,
               args: tuple = ()) -> Tuple[float, dict]:
    """
    Romberg integration on [a, b] with Richardson extrapolation.

    Returns (integral, info_dict) where info_dict contains:
        - 'levels': number of refinement levels used
        - 'error': estimated absolute error
        - 'converged': bool
        - 'T': the Romberg table (lower triangular)

    The routine refines a trapezoid grid until either max_levels is reached
    or the last two diagonal entries differ by less than tol*|integral|.

    Cost: O(2^max_levels) function evaluations in worst case.
    """
    T = np.zeros((max_levels, max_levels))

    # Level 0: single trapezoid
    h = (b - a)
    T[0, 0] = 0.5 * h * (f(a, *args) + f(b, *args))

    err = np.inf
    converged = False
    n_used = 1

    for k in range(1, max_levels):
        # Refine: add midpoints
        n = 2 ** k
        h = (b - a) / n
        # Sum of new (odd-indexed) points
        x_new = a + h * np.arange(1, n, 2)
        s_new = sum(f(xi, *args) for xi in x_new)
        T[k, 0] = 0.5 * T[k - 1, 0] + h * s_new

        # Richardson extrapolation
        for j in range(1, k + 1):
            factor = 4 ** j
            T[k, j] = (factor * T[k, j - 1] - T[k - 1, j - 1]) / (factor - 1)

        err = abs(T[k, k] - T[k - 1, k - 1])
        n_used = k + 1
        if err < tol * (1 + abs(T[k, k])):
            converged = True
            break

    info = {
        "levels":    n_used,
        "error":     err,
        "converged": converged,
        "T":         T[:n_used, :n_used],
    }
    return T[n_used - 1, n_used - 1], info


# =============================================================================
# 2. Adaptive Simpson (1D)
# =============================================================================

def simpson_adaptive(f: Callable, a: float, b: float,
                     tol: float = 1e-9, max_depth: int = 25,
                     args: tuple = ()) -> Tuple[float, dict]:
    """
    Adaptive Simpson's rule with recursive bisection.

    Bisects intervals where the Simpson estimate disagrees with the
    composite-Simpson estimate over the two halves by more than 15*tol.
    """
    n_eval = [0]

    def _simpson(x0, x2, f0, f2):
        n_eval[0] += 1
        xm = 0.5 * (x0 + x2)
        fm = f(xm, *args)
        return (x2 - x0) / 6.0 * (f0 + 4.0 * fm + f2), xm, fm

    def _recur(x0, x2, f0, f2, S, tol_local, depth):
        x1 = 0.5 * (x0 + x2)
        f1 = f(x1, *args)
        n_eval[0] += 1
        h = x2 - x0
        Sleft  = (h / 12.0) * (f0 + 4.0 * f(0.5*(x0+x1), *args) + f1)
        Sright = (h / 12.0) * (f1 + 4.0 * f(0.5*(x1+x2), *args) + f2)
        n_eval[0] += 2

        diff = abs(Sleft + Sright - S)
        if depth >= max_depth or diff < 15.0 * tol_local:
            return Sleft + Sright + (Sleft + Sright - S) / 15.0
        return (_recur(x0, x1, f0, f1, Sleft,  tol_local/2, depth+1)
                + _recur(x1, x2, f1, f2, Sright, tol_local/2, depth+1))

    f0 = f(a, *args)
    f2 = f(b, *args)
    n_eval[0] += 2
    S0, _, _ = _simpson(a, b, f0, f2)
    result = _recur(a, b, f0, f2, S0, tol, 0)
    return result, {"n_eval": n_eval[0]}


# =============================================================================
# 3. Tensorized trapezoidal 2D (backend-agnostic)
# =============================================================================

def trapezoid_2d(F, x, y, xp):
    """
    Trapezoidal integration of F(x_i, y_j) over a tensor product grid.

    Args:
        F : (Nx, Ny) tensor of integrand values.
        x : (Nx,) tensor of grid points in first axis.
        y : (Ny,) tensor of grid points in second axis.
        xp: backend.

    Returns: scalar (integral).

    Cost: O(Nx * Ny), GPU-friendly.
    """
    # First integrate along y axis for each x
    inner = xp.trapezoid(F, x=y, axis=-1)
    # Then integrate the result along x axis
    return xp.trapezoid(inner, x=x, axis=-1)


# =============================================================================
# 4. Monte Carlo (plain + importance sampling)
# =============================================================================

def monte_carlo(f: Callable, bounds, n_samples: int,
                xp, seed: int = None, args: tuple = (),
                importance_pdf: Callable = None,
                importance_sampler: Callable = None) -> Tuple[float, float, dict]:
    """
    Monte Carlo integration over a hyperrectangle.

    Args:
        f       : callable f(x, *args), x of shape (n, d).
        bounds  : list of (lo, hi) tuples, one per dimension.
        n_samples: number of MC samples.
        xp      : backend.
        seed    : RNG seed.
        args    : extra args to f.
        importance_pdf : callable p(x) returning PDF of importance distribution
                         (must integrate to 1 over the same domain).
        importance_sampler : callable returning samples ~ importance_pdf.

    Returns: (estimate, std_error, info_dict).
    """
    bounds = np.asarray(bounds, dtype=np.float64)
    d = len(bounds)
    volume = np.prod(bounds[:, 1] - bounds[:, 0])

    if importance_pdf is None:
        # Uniform sampling
        samples_np = np.empty((n_samples, d))
        rng = np.random.default_rng(seed)
        for k in range(d):
            samples_np[:, k] = rng.uniform(bounds[k, 0], bounds[k, 1], n_samples)
        samples = xp.asarray(samples_np)
        vals = f(samples, *args)
        vals_np = xp.to_numpy(vals)
        est = volume * vals_np.mean()
        sem = volume * vals_np.std(ddof=1) / math.sqrt(n_samples)
    else:
        # Importance sampling
        samples = importance_sampler(n_samples, seed)
        vals = f(samples, *args)
        weights = 1.0 / importance_pdf(samples)
        vals_np = xp.to_numpy(vals)
        wts_np = xp.to_numpy(weights) if hasattr(weights, "shape") else weights
        wval = vals_np * wts_np
        est = wval.mean()
        sem = wval.std(ddof=1) / math.sqrt(n_samples)

    return est, sem, {"n_samples": n_samples, "dim": d, "volume": float(volume)}


# =============================================================================
# 5. Lightweight Vegas-like stratified Monte Carlo
# =============================================================================

class StratifiedMC:
    """
    Stratified Monte Carlo with adaptive grid refinement.

    A simplified Vegas-like algorithm: domain is divided into nstrat
    sub-intervals per dimension, samples are distributed proportional to
    sigma_i (per-stratum standard deviation), and the grid is refined
    iteratively to reduce variance.

    Use case: 2D integrals like the (p, nu) self-energy where the
    integrand has sharp features (peaks near resonance, IR singularities).
    """

    def __init__(self, bounds, n_strata: int = 16, seed: int = 0):
        self.bounds = np.asarray(bounds, dtype=np.float64)
        self.d = len(self.bounds)
        self.n_strata = n_strata
        self.rng = np.random.default_rng(seed)
        # Importance weights per stratum (uniform initially)
        self.weights = np.ones([n_strata] * self.d) / (n_strata ** self.d)

    def integrate(self, f: Callable, n_per_iter: int = 5000,
                  n_iters: int = 5, args: tuple = ()):
        """
        Run n_iters of stratified MC, adapting weights each iteration.
        Returns (mean estimate over iterations, standard error, history).
        """
        history = []
        for it in range(n_iters):
            est, sem = self._one_iter(f, n_per_iter, args)
            history.append((est, sem))
        ests = np.array([h[0] for h in history])
        sems = np.array([h[1] for h in history])
        # Weighted average by inverse variance
        w = 1.0 / (sems ** 2 + 1e-30)
        mean = (ests * w).sum() / w.sum()
        err = math.sqrt(1.0 / w.sum())
        return mean, err, history

    def _one_iter(self, f, n_total, args):
        # Cumulative weights -> sample stratum indices
        flat_w = self.weights.flatten()
        cum_w = np.cumsum(flat_w)
        u = self.rng.uniform(size=n_total)
        strat_flat = np.searchsorted(cum_w, u)
        strat_flat = np.clip(strat_flat, 0, len(flat_w) - 1)

        # Unflatten to per-dim indices
        strat_idx = np.array(np.unravel_index(strat_flat,
                                              [self.n_strata] * self.d)).T

        # Sample within each stratum
        # Box bounds for each dimension stratum
        lo = self.bounds[:, 0]
        hi = self.bounds[:, 1]
        box_size = (hi - lo) / self.n_strata

        samples = np.empty((n_total, self.d))
        for k in range(self.d):
            u_k = self.rng.uniform(size=n_total)
            samples[:, k] = lo[k] + (strat_idx[:, k] + u_k) * box_size[k]

        # Evaluate (assume f works on numpy arrays here)
        vals = f(samples, *args)
        if hasattr(vals, "detach"):
            vals = vals.detach().cpu().numpy()
        elif hasattr(vals, "get"):
            vals = vals.get()

        # Reweight: each sample has weight (1/pdf) where pdf = flat_w / volume_per_box
        volume_per_box = np.prod(box_size)
        total_volume = np.prod(hi - lo)
        # pdf at each sample point:
        pdf = flat_w[strat_flat] / volume_per_box

        vals_w = vals / pdf
        est = vals_w.mean()
        sem = vals_w.std(ddof=1) / math.sqrt(n_total)

        # Update stratum weights based on per-stratum variance contribution
        # (Vegas-like). Group samples by stratum:
        new_w = np.zeros_like(flat_w)
        for s_idx in np.unique(strat_flat):
            mask = (strat_flat == s_idx)
            if mask.sum() > 1:
                new_w[s_idx] = np.std(vals[mask]) + 1e-12
            else:
                new_w[s_idx] = 1e-6
        new_w = new_w / new_w.sum()
        # Smooth update: blend with previous
        flat_w_new = 0.5 * flat_w + 0.5 * new_w
        flat_w_new = flat_w_new / flat_w_new.sum()
        self.weights = flat_w_new.reshape([self.n_strata] * self.d)

        return est, sem


# =============================================================================
# Self-test
# =============================================================================

if __name__ == "__main__":
    print("Self-tests for integrators")
    print("=" * 60)

    # Test 1: Romberg on a smooth function with known integral
    # int_0^1 sin(x) dx = 1 - cos(1) ~ 0.45969769
    truth = 1.0 - math.cos(1.0)
    val, info = romberg_1d(lambda x: math.sin(x), 0.0, 1.0, tol=1e-12)
    print(f"Romberg  sin on [0,1]: {val:.12f}  truth {truth:.12f}  "
          f"err {abs(val-truth):.2e}  levels {info['levels']}")

    # Test 2: Adaptive Simpson on a peaked function
    # int_{-5}^{5} 1/(1+x^2) dx = 2*atan(5) ~ 2.7468015
    truth2 = 2.0 * math.atan(5.0)
    val2, info2 = simpson_adaptive(lambda x: 1.0/(1.0+x**2), -5.0, 5.0, tol=1e-10)
    print(f"Simpson Lorentz on [-5,5]: {val2:.10f}  truth {truth2:.10f}  "
          f"err {abs(val2-truth2):.2e}  n_eval {info2['n_eval']}")

    # Test 3: 2D trapezoidal
    # int_0^1 int_0^1 x*y dx dy = 0.25
    from backends import get_backend
    xp = get_backend("numpy")
    x = xp.linspace(0, 1, 200)
    y = xp.linspace(0, 1, 200)
    X, Y = xp.meshgrid(x, y, indexing="ij")
    F = X * Y
    val3 = trapezoid_2d(F, x, y, xp)
    print(f"Trapezoid 2D xy on [0,1]^2: {float(val3):.6f}  truth 0.250000")

    # Test 4: Monte Carlo
    # int_0^1 int_0^1 (x + y)^2 dx dy = 7/6
    truth4 = 7.0/6.0
    def integrand4(pts, *args):
        return (pts[:, 0] + pts[:, 1]) ** 2
    val4, sem4, _ = monte_carlo(integrand4, [(0, 1), (0, 1)],
                                 n_samples=200_000, xp=xp, seed=0)
    print(f"MC (x+y)^2 on [0,1]^2: {val4:.6f} +/- {sem4:.6f}  truth {truth4:.6f}")

    # Test 5: Stratified MC on the same
    smc = StratifiedMC([(0, 1), (0, 1)], n_strata=8, seed=0)
    val5, err5, hist = smc.integrate(integrand4, n_per_iter=20_000, n_iters=5)
    print(f"Stratified MC: {val5:.6f} +/- {err5:.6f}  truth {truth4:.6f}")
