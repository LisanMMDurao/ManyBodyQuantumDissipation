"""
transport.py
------------
Tier 3.2: Resistivity rho(T) from the self-energy.

In the relaxation-time approximation (RTA), the dc resistivity is

    rho(T) ~ <tau^{-1}(omega, T)>_{Fermi window}

where the Fermi window <...> = int d omega (-df/d omega) ... is the
thermal average around the Fermi level, and

    tau^{-1}(omega, T) = -2 * Im Sigma^R(omega, T)

For T = 0 in the IR, Im Sigma^R(omega) ~ |omega|^vth, so the
omega-integrated rate scales as

    rho(T) ~ int d omega (-df/d omega) |omega|^vth ~ T^vth

up to a logarithmic prefactor depending on the regime.

Key predictions:
    - Ohmic + beta=1 (MFL): rho ~ T   (cuprate strange metal)
    - Sub-ohmic + beta=1: rho ~ T     (still MFL, by the degeneracy)
    - Sub-ohmic + beta=2: rho ~ T^vth with vth < 1 (heavy-fermion-like)
    - Super-ohmic: rho ~ T^vth with vth > 1 (towards FL)
"""

from __future__ import annotations
import numpy as np
from scipy.interpolate import interp1d
from self_energy import BathModel, imsigma_trapezoidal


# =============================================================================
# Fermi function and its derivative
# =============================================================================

def fermi_neg_derivative(omega: np.ndarray, T: float) -> np.ndarray:
    """
    -df/d omega = 1/(4 T cosh^2(omega/(2T)))
    """
    x = omega / (2.0 * T)
    # Avoid overflow in cosh for large x
    x_clip = np.clip(x, -50, 50)
    return 1.0 / (4.0 * T * np.cosh(x_clip) ** 2)


# =============================================================================
# Resistivity in the relaxation-time approximation
# =============================================================================

def resistivity_RTA(T_array: np.ndarray,
                    omegas: np.ndarray, im_sigma_T0: np.ndarray,
                    omega_T_scaling: bool = True) -> np.ndarray:
    """
    Compute rho(T) ~ int d omega (-df/d omega) |Im Sigma^R(omega)|

    Using the T=0 self-energy and treating T as the smearing scale of
    the Fermi window. This is the leading-order behavior.

    If omega_T_scaling is True, we replace Im Sigma^R(omega) with
    Im Sigma^R(max(omega, T)) to mimic the omega <-> T crossover
    expected in scaling regimes.

    Args:
        T_array         : (NT,) array of temperatures.
        omegas          : (Nw,) frequency grid (positive) where Im Sigma was computed.
        im_sigma_T0     : (Nw,) Im Sigma^R(omega) at T=0.
        omega_T_scaling : if True, apply the simple omega <-> T scaling.

    Returns:
        rho : (NT,) array of resistivity values.
    """
    # Extend Im Sigma to negative omegas by Im Sigma(-omega) = -Im Sigma(omega)
    omega_full = np.concatenate([-omegas[::-1], omegas])
    imsig_full = np.concatenate([-im_sigma_T0[::-1], im_sigma_T0])

    # Build interpolation for |Im Sigma| as function of |omega|
    abs_omega = np.abs(omega_full)
    abs_imsig = np.abs(imsig_full)
    # Average over +/- to enforce evenness in |omega|
    # Simple approach: use the positive-omega data only
    interp = interp1d(omegas, np.abs(im_sigma_T0),
                       bounds_error=False, fill_value=(np.abs(im_sigma_T0[0]),
                                                       np.abs(im_sigma_T0[-1])),
                       kind="linear")

    rho = np.empty(len(T_array))
    for i, T in enumerate(T_array):
        # Integration grid: from -10T to 10T
        om_grid = np.linspace(-10 * T, 10 * T, 2001)
        df_dom = fermi_neg_derivative(om_grid, T)
        if omega_T_scaling:
            scaled_om = np.maximum(np.abs(om_grid), T)
            tau_inv = 2.0 * interp(scaled_om)
        else:
            tau_inv = 2.0 * interp(np.abs(om_grid))
        rho[i] = np.trapezoid(df_dom * tau_inv, om_grid)
    return rho


# =============================================================================
# Extract resistivity exponent
# =============================================================================

def fit_resistivity_exponent(T_array, rho_array, T_range=(1e-3, 1e-1)):
    """
    Fit rho(T) ~ A T^n, return (n, A, info).
    """
    mask = (T_array >= T_range[0]) & (T_array <= T_range[1]) & (rho_array > 0)
    if mask.sum() < 3:
        return np.nan, np.nan, {"converged": False}
    logT = np.log(T_array[mask])
    logR = np.log(rho_array[mask])
    slope, intercept = np.polyfit(logT, logR, 1)
    # R^2
    y_pred = slope * logT + intercept
    ss_res = ((logR - y_pred) ** 2).sum()
    ss_tot = ((logR - logR.mean()) ** 2).sum()
    r2 = 1.0 - ss_res / max(ss_tot, 1e-30)
    return float(slope), float(np.exp(intercept)), {
        "converged": r2 > 0.9, "r_squared": float(r2)
    }


# =============================================================================
# Self-test
# =============================================================================

if __name__ == "__main__":
    from backends import get_backend
    xp = get_backend("numpy")

    print("Transport module self-test")
    print("=" * 60)

    # Build Im Sigma for ohmic + linear (MFL) and compute rho(T)
    omegas = np.geomspace(1e-4, 1.0, 80)
    model = BathModel(s=1.0, alpha=0.0, beta=1.0, eta0=0.1,
                       gapped=False, include_real_KK=False)
    print(f"Model: {model.name()}")
    im_sig = np.array([imsigma_trapezoidal(model, om, xp,
                                            n_p=120, n_nu=240,
                                            nu_floor=1e-7) for om in omegas])

    T_array = np.geomspace(1e-4, 0.1, 30)
    rho = resistivity_RTA(T_array, omegas, im_sig, omega_T_scaling=True)
    print("\nT, rho:")
    for i in range(0, len(T_array), 5):
        print(f"  T={T_array[i]:.2e}  rho={rho[i]:.4e}")

    n_fit, A_fit, info = fit_resistivity_exponent(T_array, rho,
                                                   T_range=(1e-3, 1e-2))
    print(f"\nFitted rho ~ T^{n_fit:.4f}  (R^2 = {info['r_squared']:.4f})")
    print(f"Predicted n = 1 (MFL T-linear)")
