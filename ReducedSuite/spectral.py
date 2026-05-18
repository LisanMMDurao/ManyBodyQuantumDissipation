"""
spectral.py
-----------
Tier 2.2: Spectral function A(k, omega).

Once Im Sigma^R(omega) is computed on the Fermi surface, we obtain Re Sigma
via Kramers-Kronig and then the spectral function

    A(k, omega) = -(1/pi) Im G^R(k, omega)
                = -(1/pi) Im Sigma^R(omega) /
                  [ (omega - xi_k - Re Sigma^R(omega))^2 + (Im Sigma^R(omega))^2 ]

Output: 2D map A(k, omega) suitable for comparison with ARPES MDC widths
and EDC peak positions.
"""

from __future__ import annotations
import numpy as np
from typing import Callable

from backends import Backend
from self_energy import BathModel, imsigma_trapezoidal


# =============================================================================
# Kramers-Kronig: Re Sigma from Im Sigma
# =============================================================================

def re_sigma_from_KK(omegas: np.ndarray, im_sigma: np.ndarray,
                     omega_principal_window: float = None) -> np.ndarray:
    """
    Compute Re Sigma^R(omega) from Im Sigma^R(omega) via the principal-value
    Kramers-Kronig relation:

        Re Sigma^R(omega) = (1/pi) PV int_{-infty}^{+infty}
                            d omega' Im Sigma^R(omega') / (omega' - omega)

    For omega > 0 (retarded), and using Im Sigma odd in omega (for fermions
    in the wide-band limit on the Fermi surface), we extend the input
    omegas to negative values by Im Sigma(-omega) = -Im Sigma(omega).

    Args:
        omegas    : 1D array of positive omega values (sorted ascending).
        im_sigma  : 1D array of Im Sigma values at those omegas.
        omega_principal_window : optional, exclude |omega' - omega| < this
            from the integral (handles the singularity). Defaults to 2*d_omega.

    Returns:
        re_sigma  : 1D array of Re Sigma at the same omega values.
    """
    omegas = np.asarray(omegas)
    im_sigma = np.asarray(im_sigma)
    n = len(omegas)

    # Build full symmetric grid (negative + positive)
    omegas_full = np.concatenate([-omegas[::-1], omegas])
    im_full = np.concatenate([-im_sigma[::-1], im_sigma])

    if omega_principal_window is None:
        # Use 2 * the local spacing as default exclusion window
        d_omega = np.diff(omegas).mean()
        omega_principal_window = 2.0 * d_omega

    re_sigma = np.empty(n)
    for i, om in enumerate(omegas):
        # Exclude points near omega' = omega
        mask = np.abs(omegas_full - om) > omega_principal_window
        if not mask.any():
            re_sigma[i] = np.nan
            continue
        integrand = im_full[mask] / (omegas_full[mask] - om)
        re_sigma[i] = np.trapezoid(integrand, omegas_full[mask]) / np.pi
    return re_sigma


# =============================================================================
# Spectral function A(k, omega)
# =============================================================================

def spectral_function(k_array: np.ndarray, omega_array: np.ndarray,
                      im_sigma_omega: np.ndarray, re_sigma_omega: np.ndarray,
                      kF: float = 1.0, vF: float = 1.0,
                      Z_factor: float = 1.0) -> np.ndarray:
    """
    Build A(k, omega) on a 2D grid using the Fermi-surface self-energy.

    Approximation: we use Sigma(k_F, omega) for all k near k_F, which is
    standard in low-energy NFL phenomenology (Sigma depends weakly on k
    perpendicular to FS in the IR).

    xi_k = vF * (k - kF) is the linearized dispersion.

    Args:
        k_array       : (Nk,) momentum grid.
        omega_array   : (Nw,) frequency grid.
        im_sigma_omega: (Nw,) Im Sigma at omega_array.
        re_sigma_omega: (Nw,) Re Sigma at omega_array.
        kF, vF        : Fermi momentum and velocity.
        Z_factor      : optional explicit quasi-particle residue.

    Returns:
        A : (Nk, Nw) spectral function.
    """
    K, W = np.meshgrid(k_array, omega_array, indexing="ij")
    xi_k = vF * (K - kF)

    # Broadcast Sigma along k dimension
    Im_S = im_sigma_omega[np.newaxis, :]  # (1, Nw)
    Re_S = re_sigma_omega[np.newaxis, :]

    denom = (W - xi_k - Re_S) ** 2 + Im_S ** 2
    A = -(1.0 / np.pi) * Z_factor * Im_S / np.maximum(denom, 1e-30)
    return A


def quasiparticle_residue(omegas: np.ndarray, re_sigma: np.ndarray,
                          omega_fit_window: float = 1e-3) -> tuple[float, dict]:
    """
    Compute Z^{-1} = 1 - dRe Sigma/d omega at omega=0 by linear fit
    of Re Sigma in a small window around zero.

    Returns (Z, info_dict).
    """
    mask = np.abs(omegas) < omega_fit_window
    if mask.sum() < 3:
        return np.nan, {"converged": False, "n_fit": int(mask.sum())}
    om_fit = omegas[mask]
    re_fit = re_sigma[mask]
    # Linear fit Re Sigma ~ a + b * omega
    slope, intercept = np.polyfit(om_fit, re_fit, 1)
    Z = 1.0 / (1.0 - slope)
    return float(Z), {"converged": True, "n_fit": int(mask.sum()),
                       "slope": float(slope), "intercept": float(intercept)}


# =============================================================================
# MDC width extraction (analog to ARPES)
# =============================================================================

def mdc_width(A: np.ndarray, k_array: np.ndarray, omega_array: np.ndarray,
              omega_target: float) -> tuple[float, float]:
    """
    For a fixed omega_target, find the momentum-distribution-curve (MDC)
    peak and its FWHM. Analog of ARPES MDC analysis.

    Returns (k_peak, FWHM).
    """
    # Find nearest omega index
    iw = int(np.argmin(np.abs(omega_array - omega_target)))
    mdc = A[:, iw]

    # Peak location
    ipeak = int(np.argmax(mdc))
    k_peak = float(k_array[ipeak])
    peak_val = float(mdc[ipeak])

    # FWHM: half-max points on either side
    half = 0.5 * peak_val
    if peak_val <= 0:
        return k_peak, np.nan

    # Search left
    left_idx = ipeak
    while left_idx > 0 and mdc[left_idx] > half:
        left_idx -= 1
    if left_idx == 0 and mdc[left_idx] > half:
        k_left = float(k_array[0])
    else:
        # Linear interpolation
        k1, k2 = k_array[left_idx], k_array[left_idx + 1]
        m1, m2 = mdc[left_idx], mdc[left_idx + 1]
        k_left = float(k1 + (half - m1) / (m2 - m1) * (k2 - k1))

    # Search right
    right_idx = ipeak
    while right_idx < len(mdc) - 1 and mdc[right_idx] > half:
        right_idx += 1
    if right_idx == len(mdc) - 1 and mdc[right_idx] > half:
        k_right = float(k_array[-1])
    else:
        k1, k2 = k_array[right_idx - 1], k_array[right_idx]
        m1, m2 = mdc[right_idx - 1], mdc[right_idx]
        k_right = float(k1 + (half - m1) / (m2 - m1) * (k2 - k1))

    return k_peak, k_right - k_left


# =============================================================================
# Self-test
# =============================================================================

if __name__ == "__main__":
    from backends import get_backend
    xp = get_backend("numpy")

    # Build a small test
    omegas = np.geomspace(1e-4, 1.0, 80)
    model = BathModel(s=1.0, alpha=0.0, beta=1.0,
                       eta0=0.1, gapped=False, include_real_KK=False)
    print(f"Model: {model.name()}")

    print("Computing Im Sigma on omega grid...")
    im_sig = np.array([imsigma_trapezoidal(model, om, xp,
                                            n_p=120, n_nu=240,
                                            nu_floor=1e-7) for om in omegas])

    print("Computing Re Sigma via Kramers-Kronig...")
    re_sig = re_sigma_from_KK(omegas, im_sig)

    print("Computing Z residue...")
    Z, info = quasiparticle_residue(omegas, re_sig, omega_fit_window=1e-2)
    print(f"  Z = {Z:.4f}   slope dRe/domega = {info['slope']:.4f}")

    print("Building A(k, omega)...")
    k_array = np.linspace(0.5, 1.5, 80)
    A = spectral_function(k_array, omegas, im_sig, re_sig, kF=1.0, vF=1.0)
    print(f"  A.shape = {A.shape}")
    print(f"  max A = {A.max():.4e}   min A = {A.min():.4e}")

    print("\nMDC widths at several omegas:")
    for om in [0.001, 0.01, 0.05, 0.1]:
        k_pk, fwhm = mdc_width(A, k_array, omegas, om)
        print(f"  omega={om:.3f}:  k_peak={k_pk:.4f}  FWHM={fwhm:.4f}")
