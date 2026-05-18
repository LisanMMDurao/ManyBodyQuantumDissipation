"""
electron_diagnostics.py
-----------------------
Electron-gas observables computed with RPA-dressed bath propagator.

Quantities:
  - Im Sigma_e(omega, T)  via GW one-shot integral with D^R_dressed
  - Re Sigma_e(omega)     via Kramers-Kronig from Im Sigma
  - Z = 1/(1 - dRe Sigma/domega)|_omega=0  — quasiparticle residue
  - m*/m via 1 - dRe Sigma/domega (and momentum-dependence)
  - Gamma_qp/T            — Planckian diagnostic
  - rho(T) ~ <|Im Sigma|>_FW   resistivity proxy
  - rho exponent alpha = d ln rho / d ln T

All observables computed as functions of the bath control parameter r
(distance to QCP) and temperature T. The aim is to map all electronic
observables on the (r, T) plane and identify universal signatures.
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Union
from scipy.integrate import simpson
from scipy.interpolate import interp1d

from bath_dressed import (BathRPAModel, D_R_dressed, Im_D_R_dressed,
                          control_parameter_r, mass_function_corrected,
                          Omega_q_bare_sq)
from rpa_polarization import lindhard_3D, re_Pi_static

ArrayLike = Union[float, np.ndarray]


# =============================================================================
# Im Sigma_e via GW one-shot with dressed bath
# =============================================================================

def im_sigma_electron(omega: float, T: float, p: BathRPAModel,
                      n_q: int = 121, n_theta: int = 61,
                      q_max_factor: float = 4.0) -> float:
    """
    Im Sigma_e^R(omega, T) on the Fermi surface, computed by one-shot GW:

        Im Sigma_e^R(omega, T) =
            g^2 * Integral d^3 q / (2pi)^3 * Im D^R_dressed(q, omega - xi_{k+q}) *
            [n_F(omega - xi_{k+q}) + n_B(xi_{k+q})]

    where xi_{k+q} = q^2/(2m) + v_F q cos(theta) (k on Fermi surface).

    Args:
        omega : external frequency
        T     : temperature (T > 0; for T = 0 use a small positive value)
        p     : BathRPAModel
        n_q   : number of q points (odd for Simpson)
        n_theta : number of theta points (odd for Simpson)
        q_max_factor : q_max = q_max_factor * k_F

    Returns:
        Im Sigma_e^R(omega, T) (real number, should be <= 0 for omega > 0)
    """
    if abs(omega) < 1e-15:
        omega = np.sign(omega) * 1e-15 if omega != 0 else 1e-15

    # Ensure odd number of points for Simpson's rule
    if n_q % 2 == 0:
        n_q += 1
    if n_theta % 2 == 0:
        n_theta += 1

    q_grid = np.linspace(1e-6, q_max_factor * p.kF, n_q)
    th_grid = np.linspace(0.0, np.pi, n_theta)
    Q, TH = np.meshgrid(q_grid, th_grid, indexing="ij")

    # Fermion energy shift (k on Fermi surface)
    xi = Q ** 2 / (2.0 * p.m_e) + p.vF * Q * np.cos(TH)
    nu = omega - xi  # frequency entering the bath propagator

    # Im D^R_dressed at (q, nu)
    Im_D = Im_D_R_dressed(nu, Q, p)

    # Thermal kernel: N_T(omega, xi) = n_B(nu) + n_F(-xi),  nu = omega - xi
    # This equals n_B(nu) + 1 - n_F(xi) and reduces to Theta(xi)Theta(omega-xi)
    # in the T -> 0 limit for omega > 0.
    #
    # Numerical care: n_B(nu) diverges at nu = 0. Standard trick is to combine
    # n_B(nu) * Im D(nu, q) using the identity
    #    n_B(nu) Im D(nu) ~ (T/nu) * Im D(nu) for small nu,
    # but Im D(nu) ~ nu (from causality), so the product is finite.
    # We use direct evaluation with a small offset to nu (regularization).
    nu_reg = nu + np.where(np.abs(nu) < 1e-20, 1e-20, 0.0)
    arg_nu = nu_reg / T
    arg_xi = -xi / T

    # Stable n_B: 1/(exp(x) - 1) with overflow protection.
    # For |arg| large positive: n_B -> exp(-arg) (small positive)
    # For |arg| large negative: n_B -> -1 - exp(arg) (close to -1)
    # For |arg| small: n_B -> 1/arg - 1/2 + arg/12 (Laurent series)
    # We use direct expression in stable form:
    arg_nu_clipped = np.clip(arg_nu, -500, 500)
    n_B = 1.0 / (np.expm1(arg_nu_clipped))  # expm1 stable near zero

    # n_F(-xi) = 1/(exp(-xi/T) + 1)
    arg_xi_clipped = np.clip(arg_xi, -500, 500)
    n_F_neg_xi = 1.0 / (np.exp(arg_xi_clipped) + 1.0)

    N_T = n_B + n_F_neg_xi

    # Angular measure d^3q / (2pi)^3 = q^2 sin(theta) dq dtheta dphi / (2pi)^3
    # After phi integration: * 2pi -> q^2 sin(theta) dq dtheta / (2pi)^2 / (2pi)
    # Conventional factor: 1/(2pi)^3 * 2pi = 1/(4 pi^2)
    measure = Q ** 2 * np.sin(TH) / (4.0 * np.pi ** 2)

    integrand = p.g ** 2 * Im_D * N_T * measure

    # Integrate
    int_th = simpson(integrand, x=th_grid, axis=1)
    result = simpson(int_th, x=q_grid, axis=0)

    return float(result)


# =============================================================================
# Compute Im Sigma on a grid of (omega, T)
# =============================================================================

def im_sigma_grid(omega_grid: np.ndarray, T: float, p: BathRPAModel,
                  **kwargs) -> np.ndarray:
    """Compute Im Sigma_e(omega) at fixed T over a frequency grid."""
    return np.array([im_sigma_electron(om, T, p, **kwargs)
                     for om in omega_grid])


# =============================================================================
# Re Sigma via Kramers-Kronig
# =============================================================================

def re_sigma_from_im(omega_grid: np.ndarray, im_sigma: np.ndarray,
                     omega_principal_window: Optional[float] = None) -> np.ndarray:
    """
    Re Sigma^R(omega) = (1/pi) P int d omega' Im Sigma^R(omega') / (omega' - omega)

    omega_grid must be positive and sorted. Extends to negative omega by
    Im Sigma(-omega) = -Im Sigma(omega) (retarded fermion convention).
    """
    omega_grid = np.asarray(omega_grid)
    im_sigma = np.asarray(im_sigma)
    n = len(omega_grid)

    omegas_full = np.concatenate([-omega_grid[::-1], omega_grid])
    imsig_full = np.concatenate([-im_sigma[::-1], im_sigma])

    if omega_principal_window is None:
        d_om = np.diff(omega_grid).mean()
        omega_principal_window = 2.0 * d_om

    re_sigma = np.zeros(n)
    for i, om in enumerate(omega_grid):
        mask = np.abs(omegas_full - om) > omega_principal_window
        if not mask.any():
            re_sigma[i] = np.nan
            continue
        integrand = imsig_full[mask] / (omegas_full[mask] - om)
        re_sigma[i] = np.trapezoid(integrand, omegas_full[mask]) / np.pi
    return re_sigma


# =============================================================================
# Quasiparticle residue Z = 1 / (1 - dRe Sigma / domega)|_omega=0
# =============================================================================

def quasiparticle_residue(omega_grid: np.ndarray, re_sigma: np.ndarray,
                          window: float = 5e-3) -> dict:
    """
    Compute Z at omega = 0 from numerical derivative of Re Sigma.

    Returns:
        Z       : the residue
        slope   : dRe Sigma / domega at 0
        info    : diagnostic dictionary
    """
    mask = np.abs(omega_grid) < window
    if mask.sum() < 3:
        return {"Z": np.nan, "slope": np.nan, "n_fit": int(mask.sum()),
                "converged": False}

    om_fit = omega_grid[mask]
    re_fit = re_sigma[mask]
    # Linear fit: Re Sigma ~ a + b*omega
    slope, intercept = np.polyfit(om_fit, re_fit, 1)

    Z_raw = 1.0 / (1.0 - slope)
    # Physical bound: Z in (0, 1]
    Z = max(0.0, min(1.0, float(Z_raw))) if np.isfinite(Z_raw) else np.nan

    return {
        "Z":            Z,
        "Z_raw":        float(Z_raw) if np.isfinite(Z_raw) else np.nan,
        "slope":        float(slope),
        "intercept":    float(intercept),
        "n_fit":        int(mask.sum()),
        "converged":    np.isfinite(Z) and Z > 0,
    }


# =============================================================================
# Quasiparticle decay rate Gamma_qp(T) and Planckian diagnostic
# =============================================================================

def quasiparticle_gamma(T: float, p: BathRPAModel,
                        omega_eval: float = None) -> float:
    """
    Gamma_qp(T) = 2 |Im Sigma_e^R(omega = 0+, T)|, evaluated at small omega.

    If omega_eval is None, use omega_eval = T (Planckian scaling: ~ T).
    """
    if omega_eval is None:
        omega_eval = T
    omega_eval = max(omega_eval, 1e-6)
    im_sig = im_sigma_electron(omega_eval, T, p)
    return 2.0 * abs(im_sig)


def planckian_ratio(T: float, p: BathRPAModel) -> float:
    """
    Planckian ratio: Gamma_qp(T) / (k_B T) = Gamma_qp(T) / T (natural units).

    Diagnostic:
        << 1 :  Fermi liquid
        ~ 1  :  Planckian / strange metal
        >> 1 :  bad metal / strongly incoherent
    """
    Gamma = quasiparticle_gamma(T, p)
    return Gamma / max(T, 1e-12)


# =============================================================================
# Resistivity rho(T) via Fermi window average
# =============================================================================

def resistivity_proxy(T: float, p: BathRPAModel,
                       n_omega: int = 41, omega_max_factor: float = 5.0) -> float:
    """
    rho(T) ~ int d omega (-df/d omega) (-Im Sigma_e^R(omega, T))

    Up to constants and transport vertex factors. We compute the dimensionful
    proxy (not absolute units).
    """
    if n_omega % 2 == 0:
        n_omega += 1
    om_max = max(omega_max_factor * T, 1e-3)
    omega_grid = np.linspace(1e-6, om_max, n_omega)

    df_dom = 1.0 / (4.0 * T * np.cosh(omega_grid / (2.0 * T)) ** 2)
    im_sig = np.array([im_sigma_electron(om, T, p) for om in omega_grid])
    integrand = df_dom * np.abs(im_sig)
    rho = simpson(integrand, x=omega_grid)
    return float(rho)


def resistivity_exponent(T: float, p: BathRPAModel,
                          delta_frac: float = 0.2) -> float:
    """
    alpha(T) = d ln rho / d ln T, computed by 3-point log-log fit
    around T with width delta_frac * T.
    """
    dT = max(T * delta_frac, 1e-4)
    T_pts = np.array([max(T - dT, 1e-5), T, T + dT])
    rho_pts = np.array([resistivity_proxy(t, p) for t in T_pts])
    mask = rho_pts > 1e-20
    if mask.sum() < 2:
        return np.nan
    slope = np.polyfit(np.log(T_pts[mask]), np.log(rho_pts[mask]), 1)[0]
    return float(slope)


# =============================================================================
# Spectral function exponent (Im Sigma ~ omega^vth)
# =============================================================================

def fit_imsigma_exponent(omega_grid: np.ndarray, im_sigma: np.ndarray,
                          fit_range: tuple = (1e-3, 1e-2)) -> dict:
    """Log-log fit of |Im Sigma(omega)| vs omega in a window."""
    mask = ((omega_grid >= fit_range[0]) & (omega_grid <= fit_range[1])
            & (np.abs(im_sigma) > 0))
    if mask.sum() < 3:
        return {"vth": np.nan, "A": np.nan, "r_squared": np.nan,
                "converged": False}
    log_om = np.log(omega_grid[mask])
    log_imS = np.log(np.abs(im_sigma[mask]))
    slope, intercept = np.polyfit(log_om, log_imS, 1)
    y_pred = slope * log_om + intercept
    ss_res = np.sum((log_imS - y_pred) ** 2)
    ss_tot = np.sum((log_imS - log_imS.mean()) ** 2)
    r2 = 1.0 - ss_res / max(ss_tot, 1e-30)
    return {
        "vth":       float(slope),
        "A":         float(np.exp(intercept)),
        "r_squared": float(r2),
        "converged": r2 > 0.9,
        "n_fit":     int(mask.sum()),
    }


# =============================================================================
# Self-test
# =============================================================================

if __name__ == "__main__":
    print("Electron diagnostics self-test")
    print("=" * 70)

    # Reference case: gapped bath, moderate coupling, low T
    p = BathRPAModel(dispersion="gapped", Omega0=0.5, c_disp=0.5,
                      damping="ohmic", gamma0=0.1, g=1.0,
                      kF=1.0, m_e=1.0, use_RPA=True)
    print(f"Model: {p.label()}")

    info_r = control_parameter_r(p)
    print(f"  Control parameter r = {info_r['r']:+.4f}  at q* = {info_r['q_star']:.3f}")

    # Im Sigma on omega grid
    omega_grid = np.geomspace(1e-4, 0.3, 16)
    T = 0.01
    print(f"\nIm Sigma vs omega at T = {T}:")
    im_sig = np.array([im_sigma_electron(om, T, p) for om in omega_grid])
    for om, val in zip(omega_grid[::3], im_sig[::3]):
        print(f"  omega={om:.3e}:  Im Sigma = {val:+.4e}")

    # Exponent
    fit = fit_imsigma_exponent(omega_grid, im_sig, fit_range=(1e-3, 1e-2))
    print(f"  Fitted exponent vth = {fit['vth']:.4f}  (R^2 = {fit['r_squared']:.4f})")

    # Z via KK
    re_sig = re_sigma_from_im(omega_grid, im_sig)
    Z_info = quasiparticle_residue(omega_grid, re_sig, window=5e-3)
    print(f"  Z = {Z_info['Z']:.4f}  (slope dRe Sigma/domega = {Z_info['slope']:.4f})")

    # Planckian ratio
    T_test = [0.01, 0.05, 0.1, 0.2]
    print(f"\nPlanckian ratio Gamma_qp / T:")
    for T_v in T_test:
        ratio = planckian_ratio(T_v, p)
        print(f"  T={T_v:.3f}:  Gamma/T = {ratio:.4f}")

    # Resistivity at a couple of T
    print(f"\nResistivity rho(T):")
    T_arr = np.geomspace(0.005, 0.1, 6)
    rho_arr = np.array([resistivity_proxy(T_v, p) for T_v in T_arr])
    for t, r_ in zip(T_arr, rho_arr):
        print(f"  T={t:.4f}:  rho = {r_:.4e}")
    # Fit
    log_t = np.log(T_arr)
    log_r = np.log(rho_arr)
    slope = np.polyfit(log_t, log_r, 1)[0]
    print(f"  rho ~ T^{slope:.4f}")
