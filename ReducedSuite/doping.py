"""
doping.py
---------
Effective doping of the electron gas induced by the bath.

Physical idea:
  Re Sigma_e(omega = 0) shifts the effective chemical potential of the gas:
      mu_eff = mu - Re Sigma_e(0)
  This is mathematically equivalent to a doping (carrier density shift) of
  the electron gas, induced by the bosonic bath.

Definition of fractional doping:
      delta n / n_0 = -N_0 * Re Sigma_e(0) / n_0
where:
  - N_0 = m k_F / (2 pi^2) is the density of states at E_F
  - n_0 = k_F^3 / (3 pi^2) is the bare electron density (3D Fermi gas)
  - The minus sign: Re Sigma_e(0) > 0 raises the band -> lowers carrier density

Doping is the natural "tuning axis" analogous to chemical doping in cuprates.
Combined with r (distance to bath QCP), the (r, delta_n) plane is the
electronic phase diagram in the spirit of T-x cuprate diagrams.

Quasiparticle pole equation (self-consistent version):
  omega_qp = xi_k + Re Sigma_e(omega_qp)
  Z_pole  = 1 / [1 - dRe Sigma / domega] |_{omega = omega_qp}
"""

from __future__ import annotations
import numpy as np
from typing import Optional
from scipy.optimize import brentq

from bath_dressed import BathRPAModel
from electron_diagnostics import (im_sigma_electron, re_sigma_from_im,
                                   quasiparticle_residue)


# =============================================================================
# Re Sigma_e(omega = 0) — the chemical potential shift
# =============================================================================

def re_sigma_at_zero(p: BathRPAModel, T: float,
                      omega_grid: Optional[np.ndarray] = None,
                      n_q: int = 81, n_theta: int = 41) -> dict:
    """
    Compute Re Sigma_e(omega=0) by Kramers-Kronig from Im Sigma.

    Returns dict with:
        re_sigma_0     : Re Sigma_e at omega = 0
        re_sigma_grid  : full Re Sigma(omega) array
        im_sigma_grid  : full Im Sigma(omega) array
        omega_grid     : grid used
    """
    if omega_grid is None:
        omega_grid = np.geomspace(1e-4, 0.5, 24)

    im_sig = np.array([im_sigma_electron(om, T, p, n_q=n_q, n_theta=n_theta)
                       for om in omega_grid])
    re_sig = re_sigma_from_im(omega_grid, im_sig)

    # Re Sigma at omega = 0: linear extrapolation from low-omega points
    # using Re Sigma(0) = lim_omega->0 Re Sigma(omega)
    # We fit a line to the lowest 4 points and read off the intercept
    n_fit = min(4, len(omega_grid))
    if n_fit >= 2:
        slope, intercept = np.polyfit(omega_grid[:n_fit], re_sig[:n_fit], 1)
        re_sig_0 = float(intercept)
    else:
        re_sig_0 = float(re_sig[0])

    return {
        "re_sigma_0":    re_sig_0,
        "re_sigma_grid": re_sig,
        "im_sigma_grid": im_sig,
        "omega_grid":    omega_grid,
    }


# =============================================================================
# Fractional doping
# =============================================================================

def fractional_doping(p: BathRPAModel, T: float,
                       re_sigma_0: Optional[float] = None,
                       **kwargs) -> dict:
    """
    Compute the bath-induced fractional doping:
        delta_n / n_0 = -N_0 * Re Sigma_e(0) / n_0

    where:
        n_0 = k_F^3 / (3 pi^2)  (bare 3D Fermi gas density)
        N_0 = m k_F / (2 pi^2)  (density of states at E_F)

    With these definitions: N_0 / n_0 = 3 m / (2 k_F^2) = 3 / (4 E_F)
    so equivalently:
        delta_n / n_0 = -3 / (4 E_F) * Re Sigma_e(0)

    Args:
        p           : BathRPAModel
        T           : temperature
        re_sigma_0  : if already computed, pass it; else compute internally

    Returns dict with:
        re_sigma_0   : Re Sigma at omega = 0
        mu_eff       : mu - Re Sigma(0) (effective chemical potential)
        mu_shift     : -Re Sigma(0) (chemical potential shift)
        delta_n_over_n0 : fractional doping
        n_0          : bare density
        N_0          : DOS at Fermi
    """
    if re_sigma_0 is None:
        info = re_sigma_at_zero(p, T, **kwargs)
        re_sigma_0 = info["re_sigma_0"]

    EF = p.kF ** 2 / (2.0 * p.m_e)
    n_0 = p.kF ** 3 / (3.0 * np.pi ** 2)
    N_0 = p.m_e * p.kF / (2.0 * np.pi ** 2)

    delta_n_over_n0 = -N_0 * re_sigma_0 / n_0  # equals -3 Re Sigma / (4 E_F)
    mu_shift = -re_sigma_0
    mu_eff = EF + mu_shift  # original mu = EF in our convention

    return {
        "re_sigma_0":      re_sigma_0,
        "mu_eff":          mu_eff,
        "mu_shift":        mu_shift,
        "delta_n_over_n0": float(delta_n_over_n0),
        "n_0":             n_0,
        "N_0":             N_0,
    }


# =============================================================================
# Self-consistent quasiparticle pole
# =============================================================================

def quasiparticle_pole_self_consistent(p: BathRPAModel, T: float,
                                        xi_k: float = 0.0,
                                        omega_grid: Optional[np.ndarray] = None,
                                        max_iter: int = 30,
                                        tol: float = 1e-6,
                                        n_q: int = 81, n_theta: int = 41
                                        ) -> dict:
    """
    Solve the quasiparticle pole equation
        omega_qp = xi_k + Re Sigma_e(omega_qp)
    self-consistently.

    Strategy: precompute Re Sigma(omega) on a grid, build interpolation,
    and find the fixed point of  F(omega) = xi_k + Re Sigma(omega) - omega = 0.

    Args:
        p          : BathRPAModel
        T          : temperature
        xi_k       : bare dispersion (default: Fermi surface, xi_k = 0)
        max_iter   : Picard iteration max
        tol        : convergence tolerance

    Returns dict with:
        omega_qp      : self-consistent pole frequency
        Z_pole        : Z = 1/(1 - dRe Sigma/domega) at omega_qp
        Z_at_zero     : Z evaluated at omega = 0 (first-order, for comparison)
        re_sigma_at_pole : Re Sigma at omega_qp
        converged     : bool
        n_iter        : iterations used
    """
    if omega_grid is None:
        # Need both positive and negative omega for KK and for pole search
        # Build symmetric grid in log-spaced positive omega, mirror to negative
        omega_pos = np.geomspace(1e-4, 0.5, 30)
        omega_grid = omega_pos  # KK uses only positive, reconstructs negative

    info = re_sigma_at_zero(p, T, omega_grid=omega_grid,
                             n_q=n_q, n_theta=n_theta)
    omega_pos = info["omega_grid"]
    re_sig_pos = info["re_sigma_grid"]
    im_sig_pos = info["im_sigma_grid"]

    # Extend Re Sigma to negative omega: Re Sigma is even in omega
    # for the fermion case with particle-hole symmetric coupling.
    # In general it's not exactly even, but in our model with bath
    # depending on |nu|, we have Re Sigma(-omega) ~ Re Sigma(omega)
    # for the static piece. For the omega-dependent piece, sign matters.
    # We use the simplest extension consistent with KK:
    # Re Sigma is odd in omega for half-filled systems; here we use the
    # explicit data we have on positive omega and extrapolate the slope.

    # Build full symmetric grid by mirroring (assume Re Sigma is approximately
    # odd: Re Sigma(-omega) = -Re Sigma(omega) at half-filling, but for
    # our case with Re Sigma(0) != 0 we use:
    # Re Sigma(-omega) = 2 Re Sigma(0) - Re Sigma(omega)  (reflects around mid-point)
    # This is exact if Re Sigma is linear, approximate otherwise.
    # Actually, the cleanest is to fit Re Sigma(omega) by a 3rd order poly
    # in omega around 0 and use that for the small-omega search:
    n_lowest = min(8, len(omega_pos))
    coeffs = np.polyfit(omega_pos[:n_lowest], re_sig_pos[:n_lowest], 3)
    # coeffs in order [a3, a2, a1, a0] so Re Sigma ~ a0 + a1*omega + a2*omega^2 + a3*omega^3
    a3, a2, a1, a0 = coeffs

    def Re_Sigma(om):
        return a0 + a1 * om + a2 * om ** 2 + a3 * om ** 3

    def dRe_Sigma_domega(om):
        return a1 + 2 * a2 * om + 3 * a3 * om ** 2

    # Z at omega = 0 (first-order, for comparison)
    Z_0_raw = 1.0 / (1.0 - a1) if abs(1.0 - a1) > 1e-12 else np.nan
    Z_0 = max(0.0, min(1.0, float(Z_0_raw))) if np.isfinite(Z_0_raw) else np.nan

    # Self-consistent pole: solve omega = xi_k + Re Sigma(omega)
    # Equivalently: F(omega) = omega - xi_k - Re Sigma(omega) = 0
    def F(om):
        return om - xi_k - Re_Sigma(om)

    # Try bracketing root finder over a reasonable range
    omega_search_max = 5.0 * abs(a0) + 0.5 + abs(xi_k)
    try:
        # Bracketing search
        F_lo = F(-omega_search_max)
        F_hi = F(omega_search_max)
        if F_lo * F_hi > 0:
            # No sign change — use Picard iteration
            om = xi_k + a0  # initial guess: linear
            for it in range(max_iter):
                om_new = xi_k + Re_Sigma(om)
                if abs(om_new - om) < tol:
                    om = om_new
                    converged = True
                    break
                om = om_new
            else:
                converged = False
            omega_qp = om
            n_iter = it + 1
        else:
            omega_qp = brentq(F, -omega_search_max, omega_search_max, xtol=tol)
            converged = True
            n_iter = 0
    except Exception:
        omega_qp = np.nan
        converged = False
        n_iter = -1

    # Z at the pole
    if np.isfinite(omega_qp):
        slope_at_pole = dRe_Sigma_domega(omega_qp)
        Z_pole_raw = 1.0 / (1.0 - slope_at_pole) if abs(1.0 - slope_at_pole) > 1e-12 else np.nan
        Z_pole = max(0.0, min(1.0, float(Z_pole_raw))) if np.isfinite(Z_pole_raw) else np.nan
        re_sig_at_pole = Re_Sigma(omega_qp)
    else:
        Z_pole = np.nan
        re_sig_at_pole = np.nan

    return {
        "omega_qp":         float(omega_qp) if np.isfinite(omega_qp) else np.nan,
        "Z_pole":           Z_pole,
        "Z_at_zero":        Z_0,
        "re_sigma_at_pole": float(re_sig_at_pole) if np.isfinite(re_sig_at_pole) else np.nan,
        "re_sigma_at_zero": a0,
        "slope_at_zero":    a1,
        "converged":        converged,
        "n_iter":           n_iter,
        "polynomial_fit":   {"a0": a0, "a1": a1, "a2": a2, "a3": a3},
    }


# =============================================================================
# Combined diagnostic: Z (both versions) + doping at given (g, T)
# =============================================================================

def full_diagnostic(p: BathRPAModel, T: float,
                     omega_grid: Optional[np.ndarray] = None,
                     n_q: int = 81, n_theta: int = 41) -> dict:
    """
    Compute all main diagnostics for a single (params, T) point:
      - Re Sigma(0), mu_shift, delta_n/n_0
      - omega_qp (self-consistent pole)
      - Z_at_zero (first-order)
      - Z_pole (at the shifted pole)
      - Im Sigma(omega) and Re Sigma(omega) full curves
    """
    if omega_grid is None:
        omega_grid = np.geomspace(1e-4, 0.5, 30)

    pole_info = quasiparticle_pole_self_consistent(
        p, T, xi_k=0.0, omega_grid=omega_grid,
        n_q=n_q, n_theta=n_theta
    )

    dop_info = fractional_doping(p, T,
                                  re_sigma_0=pole_info["re_sigma_at_zero"])

    return {
        "re_sigma_0":      pole_info["re_sigma_at_zero"],
        "mu_shift":        dop_info["mu_shift"],
        "delta_n_over_n0": dop_info["delta_n_over_n0"],
        "omega_qp":        pole_info["omega_qp"],
        "Z_at_zero":       pole_info["Z_at_zero"],
        "Z_pole":          pole_info["Z_pole"],
        "re_sigma_at_pole": pole_info["re_sigma_at_pole"],
        "polynomial_fit":  pole_info["polynomial_fit"],
        "converged":       pole_info["converged"],
    }


# =============================================================================
# Self-test
# =============================================================================

if __name__ == "__main__":
    print("Doping module self-test")
    print("=" * 70)

    # Sweep g and compare Z_at_zero vs Z_pole, look at doping
    base = dict(dispersion="gapped", Omega0=0.5, c_disp=0.5,
                damping="ohmic", gamma0=0.1, kF=1.0, m_e=1.0,
                use_RPA=True)

    print(f"\n{'g':>5s} {'Re Sigma(0)':>14s} {'delta_n/n_0':>14s} "
          f"{'omega_qp':>12s} {'Z(0)':>8s} {'Z(pole)':>10s}")
    print("-" * 70)

    for g in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:
        params = dict(base); params["g"] = g
        p = BathRPAModel(**params)
        try:
            diag = full_diagnostic(p, T=0.05)
            print(f"{g:>5.2f} "
                  f"{diag['re_sigma_0']:>+14.4e} "
                  f"{diag['delta_n_over_n0']:>+14.4e} "
                  f"{diag['omega_qp']:>+12.4e} "
                  f"{diag['Z_at_zero']:>8.3f} "
                  f"{diag['Z_pole']:>10.3f}")
        except Exception as e:
            print(f"{g:>5.2f}  FAILED: {e}")
