"""
self_consistent.py
------------------
Self-consistent extension of the GW one-shot self-energy:

   At each iteration, Re Sigma_e(omega=0) is computed and shifts the
   chemical potential entering the fermion dispersion:
       xi_k^new = xi_k^bare + Re Sigma_e^(it)(0)
   The new xi_k enters the thermal kernel for the next iteration:
       Im Sigma_e^(it+1)(omega, T) = G^2 ∫dq dθ Im D^R(nu) [n_B(nu) + n_F(-xi^new)]
   with nu = omega - xi^new.

This is not full GW self-consistency (we don't dress G internally), but
it does close the loop on the Fermi level shift, which is the dominant
effect at large coupling.

Iteration is by Picard with damping; we stop when |Sigma(0) change| < tol.
"""

from __future__ import annotations
import numpy as np
from typing import Optional
from scipy.integrate import simpson

from bath_dressed import BathRPAModel, Im_D_R_dressed
from electron_diagnostics import re_sigma_from_im
from doping import quasiparticle_pole_self_consistent


# =============================================================================
# Im Sigma with explicit Fermi level shift
# =============================================================================

def im_sigma_electron_shifted(omega: float, T: float, p: BathRPAModel,
                               mu_shift: float = 0.0,
                               n_q: int = 81, n_theta: int = 41,
                               q_max_factor: float = 4.0) -> float:
    """
    Im Sigma_e^R(omega, T) computed with a shifted Fermi level.

    The fermion dispersion entering the integral is

        xi_k(theta, q) = q^2 / (2m) + v_F q cos(theta) - mu_shift

    where mu_shift = Re Sigma_e(0) effectively shifts the chemical
    potential of the gas.

    Setting mu_shift = 0 recovers the standard one-shot result.
    """
    if abs(omega) < 1e-15:
        omega = 1e-15

    if n_q % 2 == 0:
        n_q += 1
    if n_theta % 2 == 0:
        n_theta += 1

    q_grid = np.linspace(1e-6, q_max_factor * p.kF, n_q)
    th_grid = np.linspace(0.0, np.pi, n_theta)
    Q, TH = np.meshgrid(q_grid, th_grid, indexing="ij")

    # Shifted fermion energy (k on Fermi surface)
    xi = Q ** 2 / (2.0 * p.m_e) + p.vF * Q * np.cos(TH) - mu_shift
    nu = omega - xi

    Im_D = Im_D_R_dressed(nu, Q, p)

    # Thermal kernel: n_B(nu) + n_F(-xi)
    nu_reg = nu + np.where(np.abs(nu) < 1e-20, 1e-20, 0.0)
    arg_nu_clipped = np.clip(nu_reg / T, -500, 500)
    n_B = 1.0 / (np.expm1(arg_nu_clipped))

    arg_xi_clipped = np.clip(-xi / T, -500, 500)
    n_F_neg_xi = 1.0 / (np.exp(arg_xi_clipped) + 1.0)

    N_T = n_B + n_F_neg_xi

    measure = Q ** 2 * np.sin(TH) / (4.0 * np.pi ** 2)
    integrand = p.g ** 2 * Im_D * N_T * measure

    int_th = simpson(integrand, x=th_grid, axis=1)
    return float(simpson(int_th, x=q_grid, axis=0))


# =============================================================================
# Self-consistent solution for mu_shift
# =============================================================================

def solve_self_consistent_shift(p: BathRPAModel, T: float,
                                 omega_grid: Optional[np.ndarray] = None,
                                 max_iter: int = 50, tol: float = 1e-4,
                                 damping: float = 0.3,
                                 n_q: int = 61, n_theta: int = 31,
                                 verbose: bool = False) -> dict:
    """
    Iteratively solve for the self-consistent chemical potential shift.

    Uses adaptive damping: if oscillating, reduce damping;
    if monotone slow, increase damping.
    """
    if omega_grid is None:
        omega_grid = np.geomspace(1e-4, 0.5, 18)

    mu_shift = 0.0
    history = []
    prev_change = None
    current_damping = damping

    for it in range(max_iter):
        im_sig = np.array([
            im_sigma_electron_shifted(om, T, p,
                                       mu_shift=mu_shift,
                                       n_q=n_q, n_theta=n_theta)
            for om in omega_grid
        ])
        re_sig = re_sigma_from_im(omega_grid, im_sig)

        n_fit = min(4, len(omega_grid))
        if n_fit >= 2:
            slope, intercept = np.polyfit(omega_grid[:n_fit], re_sig[:n_fit], 1)
            re_sig_0 = float(intercept)
        else:
            re_sig_0 = float(re_sig[0])

        change = re_sig_0 - mu_shift
        history.append({"iter": it, "mu_shift": mu_shift, "re_sig_0": re_sig_0,
                        "change": change, "damping": current_damping})

        if verbose:
            print(f"    iter {it}: mu_shift={mu_shift:+.4e}, "
                  f"re_sig(0)={re_sig_0:+.4e}, change={change:+.4e}, "
                  f"damping={current_damping:.2f}")

        if abs(change) < tol:
            mu_shift = re_sig_0
            return {"mu_shift": mu_shift, "converged": True, "n_iter": it + 1,
                    "history": history, "im_sigma_final": im_sig,
                    "re_sigma_final": re_sig, "omega_grid": omega_grid}

        # Adaptive damping: detect oscillation (sign change) -> reduce damping
        if prev_change is not None and prev_change * change < 0:
            current_damping = max(0.1, current_damping * 0.7)

        mu_shift = mu_shift + current_damping * change
        prev_change = change

    return {"mu_shift": mu_shift, "converged": False, "n_iter": max_iter,
            "history": history, "im_sigma_final": im_sig,
            "re_sigma_final": re_sig, "omega_grid": omega_grid}


# =============================================================================
# Full self-consistent diagnostic
# =============================================================================

def full_sc_diagnostic(p: BathRPAModel, T: float,
                        omega_grid: Optional[np.ndarray] = None,
                        n_q: int = 61, n_theta: int = 31,
                        verbose: bool = False) -> dict:
    """
    Solve self-consistent mu_shift, then compute all observables at that
    fixed point.

    Returns dict with:
      mu_shift, delta_n_over_n0, omega_qp, Z_pole, Z_at_zero,
      converged, n_iter
    """
    sc_info = solve_self_consistent_shift(p, T,
                                            omega_grid=omega_grid,
                                            n_q=n_q, n_theta=n_theta,
                                            verbose=verbose)

    mu_shift = sc_info["mu_shift"]
    im_sig = sc_info["im_sigma_final"]
    re_sig = sc_info["re_sigma_final"]
    omega_grid_used = sc_info["omega_grid"]

    # Reconstruct Z from polynomial fit of Re Sigma at low omega
    n_lowest = min(8, len(omega_grid_used))
    coeffs = np.polyfit(omega_grid_used[:n_lowest], re_sig[:n_lowest], 3)
    a3, a2, a1, a0 = coeffs

    # At self-consistency, the pole is at omega = 0 (by construction:
    # xi_k = 0 at FS with shifted mu, and Re Sigma(0) is absorbed into mu).
    # So the relevant Z is Z_pole = 1/(1 - dRe Sigma/domega) at omega = 0.
    Z_pole_raw = 1.0 / (1.0 - a1) if abs(1.0 - a1) > 1e-12 else np.nan
    Z_pole = max(0.0, min(1.0, float(Z_pole_raw))) if np.isfinite(Z_pole_raw) else np.nan

    # Doping
    n_0 = p.kF ** 3 / (3.0 * np.pi ** 2)
    N_0 = p.m_e * p.kF / (2.0 * np.pi ** 2)
    delta_n_over_n0 = float(-N_0 * mu_shift / n_0)

    return {
        "mu_shift":        float(mu_shift),
        "delta_n_over_n0": delta_n_over_n0,
        "Z_pole":          Z_pole,
        "Z_at_zero":       Z_pole,  # same after self-consistency
        "omega_qp":        0.0,     # pole is at omega = 0 by construction
        "converged":       sc_info["converged"],
        "n_iter":          sc_info["n_iter"],
        "im_sigma_final":  im_sig,
        "re_sigma_final":  re_sig,
        "omega_grid":      omega_grid_used,
    }


# =============================================================================
# Self-test
# =============================================================================

if __name__ == "__main__":
    print("Self-consistent module test")
    print("=" * 70)

    base = dict(dispersion="gapped", Omega0=0.5, c_disp=0.5,
                damping="ohmic", gamma0=0.1, kF=1.0, m_e=1.0,
                use_RPA=True)

    print(f"\n{'g':>5s} {'mu_shift':>14s} {'delta_n/n0':>14s} {'Z':>8s} "
          f"{'iter':>5s} {'conv':>5s}")
    print("-" * 70)

    for g in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]:
        params = dict(base); params["g"] = g
        p = BathRPAModel(**params)
        diag = full_sc_diagnostic(p, T=0.05, verbose=False)
        print(f"{g:>5.2f} {diag['mu_shift']:>+14.4e} "
              f"{diag['delta_n_over_n0']:>+14.4e} "
              f"{diag['Z_pole']:>8.3f} "
              f"{diag['n_iter']:>5d} {str(diag['converged']):>5s}")
