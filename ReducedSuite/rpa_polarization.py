"""
rpa_polarization.py
-------------------
Analytical RPA polarization (Lindhard function) for 3D free electron gas.

We implement the full complex Lindhard function Pi(q, omega) including:
  - Static limit Pi(q, 0) with the logarithmic Kohn singularity at q = 2 k_F
  - Dynamic real part Re Pi(q, omega)
  - Imaginary part Im Pi(q, omega) supported within the particle-hole continuum

These are the standard 3D results (see e.g. Mahan, Many-Particle Physics §5.5,
or Giuliani-Vignale §4.4). All quantities normalized by the density of states
N_0 = m k_F / (2 pi^2 hbar^2).

Conventions:
  - All energies in units of E_F
  - All momenta in units of k_F
  - hbar = 1
  - m_e = 1 (the bare fermion mass m chosen such that E_F = k_F^2 / 2m)
  - Convention: Pi^R retarded, with Im Pi <= 0 for omega > 0

Key result we use elsewhere: the static Lindhard function
    Re Pi(q, 0) / N_0 = -F_static(q / 2k_F)
where
    F_static(x) = 1/2 + (1 - x^2)/(4x) * ln|(1+x)/(1-x)|
has the logarithmic Kohn anomaly at x = 1 (i.e. q = 2 k_F).
"""

from __future__ import annotations
import numpy as np
from typing import Union

ArrayLike = Union[float, np.ndarray]


# =============================================================================
# Static Lindhard function (zero frequency, full q-dependence)
# =============================================================================

def F_static(x: ArrayLike, eps: float = 1e-10) -> np.ndarray:
    """
    Static Lindhard form factor:
        F(x) = 1/2 + (1 - x^2)/(4x) * ln|(1+x)/(1-x)|
    such that Re Pi(q, 0) / N_0 = -F(q / 2k_F).

    Properties:
        F(0) = 1
        F(1) = 1/2 (with logarithmic singularity in derivative -- Kohn anomaly)
        F(x -> infty) ~ 1/(3 x^2) -> 0
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    out = np.empty_like(x)

    # Limit x -> 0:  F -> 1 - x^2/3 + O(x^4)
    small = np.abs(x) < eps
    out[small] = 1.0 - x[small] ** 2 / 3.0

    # Logarithmic singularity at x = 1:
    near_one = (np.abs(x - 1.0) < eps) & ~small
    out[near_one] = 0.5

    regular = ~small & ~near_one
    xr = x[regular]
    # Argument of log: (1 + x)/(1 - x), abs value
    arg = np.abs((1.0 + xr) / (1.0 - xr))
    # Avoid log of zero
    arg = np.maximum(arg, 1e-300)
    out[regular] = 0.5 + (1.0 - xr ** 2) / (4.0 * xr) * np.log(arg)

    return out


def re_Pi_static(q: ArrayLike, kF: float = 1.0, N0: float = None,
                 m: float = 1.0) -> np.ndarray:
    """
    Real part of polarization at omega = 0, in physical units.

    Returns Re Pi(q, 0) = -N_0 * F_static(q / 2k_F).

    If N_0 is None, computed from m, kF: N_0 = m * kF / (2 pi^2).
    """
    if N0 is None:
        N0 = m * kF / (2.0 * np.pi ** 2)
    x = np.asarray(q, dtype=float) / (2.0 * kF)
    return -N0 * F_static(x)


# =============================================================================
# Dynamic Lindhard function (full complex Pi(q, omega))
# =============================================================================

def lindhard_3D(omega: ArrayLike, q: ArrayLike,
                kF: float = 1.0, m: float = 1.0,
                eps: float = 1e-10) -> np.ndarray:
    """
    Full complex 3D Lindhard function Pi^R(q, omega), retarded.

    Standard form (Mahan eq. 5.5.27, AGD §10.2):
        Re Pi / N_0 = -1/2 - (1/(8x)) * [(1 - (x - nu)^2) ln|(x - nu + 1)/(x - nu - 1)|
                                       + (1 - (x + nu)^2) ln|(x + nu + 1)/(x + nu - 1)|]

    Variables:
        x  = q / (2 kF)
        nu = omega / (v_F q)

    Imaginary part inside particle-hole continuum:
        Region IR  (|nu| + x < 1):     Im Pi / N_0 = -(pi/2) * nu
        Region edge (|1-x| < |nu| < 1+x): Im Pi / N_0 = sign(nu) * (-pi/(8x))*(1 - (|nu|-x)^2)
        Otherwise zero.

    Returns complex array. Im Pi <= 0 for omega > 0 (retarded convention).
    """
    omega = np.asarray(omega, dtype=float)
    q = np.asarray(q, dtype=float)
    vF = kF / m

    q_safe = np.where(np.abs(q) > eps, q, eps)
    x = q_safe / (2.0 * kF)
    nu = omega / (vF * q_safe)

    # -------- Real part --------
    def log_safe(a, b):
        """Compute ln|a/b| robustly."""
        return np.log(np.abs(a) + 1e-300) - np.log(np.abs(b) + 1e-300)

    a1 = x - nu
    a2 = x + nu

    # (1 - a1^2) * ln|(a1+1)/(a1-1)|, with care near a1 = ±1
    def term(a):
        result = np.zeros_like(a)
        regular = np.abs(np.abs(a) - 1.0) > eps
        result[regular] = (1.0 - a[regular] ** 2) * log_safe(
            a[regular] + 1.0, a[regular] - 1.0
        )
        # At a = ±1, the prefactor (1 - a^2) vanishes faster than log diverges
        result[~regular] = 0.0
        return result

    re_Pi_norm = -0.5 - (term(a1) + term(a2)) / (8.0 * x)

    # -------- Imaginary part --------
    abs_nu = np.abs(nu)
    im_Pi_norm = np.zeros_like(omega)

    # Region 1: IR — |nu| + x < 1
    region_IR = (abs_nu + x < 1.0) & (x > eps)
    im_Pi_norm = np.where(region_IR,
                          -(np.pi / 2.0) * nu, im_Pi_norm)

    # Region 2: edge — |1 - x| < |nu| < 1 + x  (and not in IR)
    region_edge = (abs_nu < 1.0 + x) & (abs_nu > np.abs(1.0 - x)) & ~region_IR
    edge_val = -(np.pi / (8.0 * x)) * (1.0 - (abs_nu - x) ** 2)
    im_Pi_norm = np.where(region_edge,
                          np.sign(nu) * edge_val, im_Pi_norm)

    # Multiply by N_0 to give physical Pi
    N0 = m * kF / (2.0 * np.pi ** 2)
    return N0 * (re_Pi_norm + 1j * im_Pi_norm)


# =============================================================================
# Continuum boundaries (for plotting and diagnostics)
# =============================================================================

def particle_hole_continuum(q: np.ndarray, kF: float = 1.0,
                             m: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Boundaries of the particle-hole continuum in (q, omega):
        omega_+(q) = v_F q + q^2 / (2m)
        omega_-(q) = | v_F q - q^2 / (2m) |
    Inside [omega_-, omega_+], Im Pi != 0.
    """
    vF = kF / m
    omega_plus = vF * q + q ** 2 / (2.0 * m)
    omega_minus = np.abs(vF * q - q ** 2 / (2.0 * m))
    return omega_minus, omega_plus


# =============================================================================
# Kohn anomaly diagnostic
# =============================================================================

def kohn_singularity_strength(q_array: np.ndarray, kF: float = 1.0,
                               m: float = 1.0) -> dict:
    """
    Quantifies the logarithmic singularity at q = 2 k_F in Re Pi(q, 0).

    Returns a dictionary with:
        q_array : input grid
        re_Pi   : Re Pi(q, 0)
        d_re_Pi : numerical derivative d Re Pi / dq
        q_2kF   : closest q value to 2 k_F in the grid

    The Kohn anomaly is the logarithmic divergence of d Re Pi / dq at q = 2 k_F.
    """
    re_pi = re_Pi_static(q_array, kF=kF, m=m)
    d_re_pi = np.gradient(re_pi, q_array)
    idx_2kF = int(np.argmin(np.abs(q_array - 2.0 * kF)))
    return {
        "q_array":  q_array,
        "re_Pi":    re_pi,
        "d_re_Pi":  d_re_pi,
        "q_2kF":    float(q_array[idx_2kF]),
        "idx_2kF":  idx_2kF,
    }


# =============================================================================
# Self-test
# =============================================================================

if __name__ == "__main__":
    print("RPA polarization self-test")
    print("=" * 60)

    # Static Lindhard at canonical points
    print("\nF_static at canonical x values:")
    print(f"  F(0)    = {F_static(np.array([0.0]))[0]:.6f}  (expected 1.0)")
    print(f"  F(0.5)  = {F_static(np.array([0.5]))[0]:.6f}")
    print(f"  F(1.0)  = {F_static(np.array([1.0]))[0]:.6f}  (expected 0.5)")
    print(f"  F(2.0)  = {F_static(np.array([2.0]))[0]:.6f}")
    print(f"  F(5.0)  = {F_static(np.array([5.0]))[0]:.6f}  (~ 1/(3*25)={1/75:.4f})")

    # Test dynamic Lindhard at omega -> 0
    q_test = np.array([0.1, 0.5, 1.0, 1.5, 2.0, 2.5])
    omega_small = 1e-6
    Pi_small_omega = lindhard_3D(np.full_like(q_test, omega_small),
                                  q_test, kF=1.0, m=1.0)
    Pi_static = re_Pi_static(q_test, kF=1.0, m=1.0)
    print("\nConsistency check: Pi(q, omega->0+) vs Re Pi(q, 0):")
    for q, p_dyn, p_stat in zip(q_test, Pi_small_omega, Pi_static):
        print(f"  q={q:.2f}:  Re Pi(dynamic) = {p_dyn.real:+.6f},  "
              f"Re Pi(static) = {p_stat:+.6f},  Im Pi = {p_dyn.imag:+.4e}")

    # Inside particle-hole continuum: Im Pi should be nonzero
    print("\nInside particle-hole continuum (q=1, omega=1):")
    Pi_inside = lindhard_3D(np.array([1.0]), np.array([1.0]),
                             kF=1.0, m=1.0)[0]
    print(f"  Pi = {Pi_inside:+.4f}   (Im Pi should be < 0)")

    # Kohn anomaly diagnostic
    q_grid = np.linspace(0.01, 3.0, 1000)
    diag = kohn_singularity_strength(q_grid)
    idx = diag["idx_2kF"]
    print(f"\nKohn anomaly:")
    print(f"  q_2kF (closest grid) = {diag['q_2kF']:.4f}")
    print(f"  d Re Pi / dq at 2 k_F = {diag['d_re_Pi'][idx]:+.4f}  "
          f"(should be large/divergent in IR limit)")
