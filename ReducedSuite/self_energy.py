"""
self_energy.py
--------------
Tier 1: Direct numerical evaluation of the electronic self-energy

    Im Sigma^R(k_F, omega) = -C * int_0^{2 k_F} dp p * int_0^omega dnu Im D^R(p, nu)

where the bath spectral function is

    Im D^R(p, nu) = -Gamma(p, nu) sgn(nu) / [(nu^2 - E_p^2 - Delta(p,nu))^2 + Gamma(p,nu)^2]

with parametrized damping families (Ohmic, frequency, momentum, mixed) and
dispersion family E_p = c p^beta.

Units throughout: natural (hbar = k_B = 1). Momenta in units of k_F, energies
in units of E_F.

Validates the corrected exponent expression from the manuscript:
    vartheta_Sigma = 1 + s*(2 - alpha - 4*beta)/(2*beta + alpha) + s

against direct numerical integration of Im Sigma over the (p, nu) plane.
"""

from __future__ import annotations
import numpy as np
from typing import Callable
from dataclasses import dataclass, field

from backends import Backend
from integrators import trapezoid_2d, romberg_1d, simpson_adaptive, monte_carlo, StratifiedMC


# =============================================================================
# Model parameters and damping/dispersion ansatze
# =============================================================================

@dataclass
class BathModel:
    """
    Container for the parametrized bath model.

    Damping family is specified by (s, alpha):
        eta(p, nu) = eta0 * (|nu|/Lambda_omega)^(s-1) * (k0/p)^alpha
        Gamma(p, nu) = pi * |nu| * eta(p, nu)

    Dispersion family by beta:
        E_p = c * p^beta   (polynomial)
        OR
        E_p = sqrt(Omega0^2 + (c*p)^2)   (gapped, if gapped=True)

    Real part Delta(p,nu) from Kramers-Kronig of Gamma:
        Delta(p, omega) = -(hbar eta0 / Lambda_omega^(s-1)) * cot(pi s / 2) * |omega|^s
    """
    s:          float = 1.0
    alpha:      float = 0.0
    beta:       float = 2.0
    eta0:       float = 0.1
    c:          float = 1.0
    k0:         float = 1.0
    Lambda_w:   float = 1.0
    Omega0:     float = 0.1
    gapped:     bool  = False
    include_real_KK: bool = True

    def name(self) -> str:
        disp = "gapped" if self.gapped else f"p^{self.beta:g}"
        return (f"s={self.s:g} alpha={self.alpha:g} disp={disp} "
                f"eta0={self.eta0:g} Omega0={self.Omega0:g}")


def eta_function(model: BathModel, p, nu, xp: Backend):
    """eta(p, nu) damping intensity."""
    abs_nu = xp.abs(nu) + 1e-30
    p_safe = xp.where(p > 1e-30, p, 1e-30 * xp.ones_like(p) if hasattr(p, 'shape') else 1e-30)
    return (model.eta0
            * (abs_nu / model.Lambda_w) ** (model.s - 1.0)
            * (model.k0 / p_safe) ** model.alpha)


def Gamma_function(model: BathModel, p, nu, xp: Backend):
    """Gamma(p, nu) = pi * |nu| * eta(p, nu).  >= 0."""
    return np.pi * xp.abs(nu) * eta_function(model, p, nu, xp)


def Delta_KK(model: BathModel, p, nu, xp: Backend):
    """
    Kramers-Kronig partner of Gamma in the pure frequency-damping family:
        Delta(omega) = -(hbar eta_0 / Lambda^(s-1)) cot(pi s/2) |omega|^s

    For mixed damping, we use the same form, treating the momentum factor
    as carried through the integral (a working approximation).
    """
    if not model.include_real_KK:
        return xp.zeros_like(nu) if hasattr(nu, 'shape') else 0.0
    s = model.s
    if abs(s - 1.0) < 1e-10:
        # Ohmic limit: cot(pi/2) = 0 -> no real shift
        return xp.zeros_like(nu) if hasattr(nu, 'shape') else 0.0
    cot_factor = 1.0 / np.tan(np.pi * s / 2.0)
    p_safe = xp.where(p > 1e-30, p, 1e-30 * xp.ones_like(p) if hasattr(p, 'shape') else 1e-30)
    momentum_factor = (model.k0 / p_safe) ** model.alpha
    return -(model.eta0 / model.Lambda_w ** (s - 1.0)) * cot_factor \
           * xp.abs(nu) ** s * momentum_factor


def Ep_dispersion(model: BathModel, p, xp: Backend):
    """Bath dispersion E_p."""
    if model.gapped:
        return xp.sqrt(model.Omega0 ** 2 + (model.c * p) ** 2)
    return model.c * p ** model.beta


# =============================================================================
# Bath spectral function
# =============================================================================

def ImD_R(model: BathModel, p, nu, xp: Backend):
    """
    Im D^R(p, nu) = -Gamma * sgn(nu) / [(nu^2 - E_p^2 - Delta)^2 + Gamma^2]

    Returns a tensor with the same shape as (p, nu) broadcast.
    """
    Gamma = Gamma_function(model, p, nu, xp)
    Delta = Delta_KK(model, p, nu, xp)
    Ep = Ep_dispersion(model, p, xp)

    denom = (nu ** 2 - Ep ** 2 - Delta) ** 2 + Gamma ** 2
    sgn_nu = xp.where(nu >= 0, xp.ones_like(nu), -xp.ones_like(nu)) \
        if hasattr(nu, 'shape') else (1.0 if nu >= 0 else -1.0)
    return -Gamma * sgn_nu / denom


# =============================================================================
# Self-energy via 2D trapezoidal (deterministic baseline)
# =============================================================================

def imsigma_trapezoidal(model: BathModel, omega: float,
                        xp: Backend,
                        kF: float = 1.0, m: float = 1.0,
                        n_p: int = 200, n_nu: int = 400,
                        p_max_factor: float = 2.0,
                        nu_floor: float = 1e-6) -> float:
    """
    Direct trapezoidal evaluation of

        Im Sigma^R(k_F, omega) = -C * int_0^{p_max} dp p int_0^omega dnu Im D^R(p, nu)

    where C = m / (4 pi^2 k_F) and p_max = p_max_factor * k_F.

    Args:
        model     : BathModel.
        omega     : external frequency (positive).
        xp        : backend (numpy / cupy / torch).
        kF, m     : Fermi momentum and effective mass.
        n_p, n_nu : grid points in momentum and frequency.
        p_max_factor : multiplier of k_F for momentum cutoff.
        nu_floor  : lower limit on nu integration (avoid nu=0 singularity).

    Returns:
        float (Im Sigma at given omega). Negative for omega > 0.
    """
    if omega <= nu_floor:
        return 0.0

    p_grid  = xp.linspace(1e-6, p_max_factor * kF, n_p)
    nu_grid = xp.linspace(nu_floor, omega, n_nu)

    P, NU = xp.meshgrid(p_grid, nu_grid, indexing="ij")
    integrand = ImD_R(model, P, NU, xp) * P    # p * Im D^R

    inner = trapezoid_2d(integrand, p_grid, nu_grid, xp)
    # Sign convention: with Im D^R < 0 for nu > 0 (causality), the inner
    # integral is negative; multiplying by +C gives Im Sigma^R < 0 as required.
    # The "-C" in the manuscript Eq. (8) is correct only under a non-standard
    # convention where Im D^R is taken positive; we use the physical convention
    # consistently here.
    C = m / (4.0 * np.pi ** 2 * kF)
    result = C * inner
    return float(xp.to_numpy(result))


# =============================================================================
# Self-energy via Romberg in nu, trapezoidal in p
# =============================================================================

def imsigma_romberg(model: BathModel, omega: float,
                    xp: Backend,
                    kF: float = 1.0, m: float = 1.0,
                    n_p: int = 200,
                    p_max_factor: float = 2.0,
                    nu_floor: float = 1e-6,
                    romberg_tol: float = 1e-8,
                    max_levels: int = 14) -> tuple[float, dict]:
    """
    Higher-accuracy evaluation: inner nu integral by Romberg, outer p by trapezoid.

    Useful for validating trapezoid_2d at coarse grids and for resolving
    the IR behavior with high precision (needed for log-log fits).
    """
    if omega <= nu_floor:
        return 0.0, {"converged": True, "levels": 0}

    p_grid = np.linspace(1e-6, p_max_factor * kF, n_p)
    inner_vals = np.empty(n_p)
    levels_total = 0
    converged_all = True

    for i, p_val in enumerate(p_grid):
        def integrand_nu(nu, p=p_val):
            p_arr = np.array(p)
            nu_arr = np.array(nu)
            return float(xp.to_numpy(ImD_R(model, xp.asarray(p_arr),
                                            xp.asarray(nu_arr), xp)))
        val, info = romberg_1d(integrand_nu, nu_floor, omega,
                               max_levels=max_levels, tol=romberg_tol)
        inner_vals[i] = val
        levels_total += info["levels"]
        if not info["converged"]:
            converged_all = False

    # Outer integral
    outer = np.trapezoid(p_grid * inner_vals, p_grid)
    C = m / (4.0 * np.pi ** 2 * kF)
    result = C * outer  # sign: see note in imsigma_trapezoidal
    return float(result), {
        "converged": converged_all,
        "levels_total": levels_total,
        "n_p": n_p,
    }


# =============================================================================
# Monte Carlo evaluation
# =============================================================================

def imsigma_mc(model: BathModel, omega: float,
               xp: Backend,
               kF: float = 1.0, m: float = 1.0,
               n_samples: int = 200_000,
               p_max_factor: float = 2.0,
               nu_floor: float = 1e-6,
               seed: int = 0) -> tuple[float, float]:
    """
    Plain MC integration of the (p, nu) integral. Returns (estimate, std_error).
    """
    if omega <= nu_floor:
        return 0.0, 0.0

    p_max = p_max_factor * kF

    def integrand(pts, *args):
        p_arr = xp.asarray(pts[:, 0])
        nu_arr = xp.asarray(pts[:, 1])
        return ImD_R(model, p_arr, nu_arr, xp) * p_arr

    est, sem, _ = monte_carlo(integrand,
                              [(1e-6, p_max), (nu_floor, omega)],
                              n_samples=n_samples, xp=xp, seed=seed)
    C = m / (4.0 * np.pi ** 2 * kF)
    return float(C * est), float(C * sem)


# =============================================================================
# Stratified MC evaluation
# =============================================================================

def imsigma_stratified_mc(model: BathModel, omega: float,
                          xp: Backend,
                          kF: float = 1.0, m: float = 1.0,
                          n_strata: int = 16,
                          n_per_iter: int = 20_000, n_iters: int = 5,
                          p_max_factor: float = 2.0,
                          nu_floor: float = 1e-6,
                          seed: int = 0) -> tuple[float, float]:
    """
    Stratified (Vegas-like) MC integration. Better for peaked integrands.
    """
    if omega <= nu_floor:
        return 0.0, 0.0

    p_max = p_max_factor * kF

    def integrand(pts, *args):
        p_arr = xp.asarray(pts[:, 0])
        nu_arr = xp.asarray(pts[:, 1])
        return ImD_R(model, p_arr, nu_arr, xp) * p_arr

    smc = StratifiedMC([(1e-6, p_max), (nu_floor, omega)],
                       n_strata=n_strata, seed=seed)
    est, err, _ = smc.integrate(integrand, n_per_iter=n_per_iter,
                                n_iters=n_iters)
    C = m / (4.0 * np.pi ** 2 * kF)
    return float(C * est), float(C * err)


# =============================================================================
# Exponent extraction by log-log fit
# =============================================================================

def fit_exponent(omegas, imsigmas, omega_range=(1e-4, 1e-2)):
    """
    Fit Im Sigma ~ |Im Sigma| ~ A * omega^vartheta in a window of omegas.

    Returns (vartheta_fit, A_fit, info_dict).
    """
    omegas = np.asarray(omegas)
    imsigmas = np.asarray(imsigmas)
    mask = (omegas >= omega_range[0]) & (omegas <= omega_range[1]) \
           & (imsigmas != 0.0) & np.isfinite(imsigmas)
    if mask.sum() < 3:
        return np.nan, np.nan, {"n_fit": int(mask.sum()), "converged": False}

    log_omega = np.log(omegas[mask])
    log_abs   = np.log(np.abs(imsigmas[mask]))

    # Linear regression
    n = len(log_omega)
    Sx = log_omega.sum()
    Sy = log_abs.sum()
    Sxx = (log_omega ** 2).sum()
    Sxy = (log_omega * log_abs).sum()

    slope = (n * Sxy - Sx * Sy) / (n * Sxx - Sx ** 2)
    intercept = (Sy - slope * Sx) / n

    # R^2
    y_pred = slope * log_omega + intercept
    ss_res = ((log_abs - y_pred) ** 2).sum()
    ss_tot = ((log_abs - log_abs.mean()) ** 2).sum()
    r2 = 1.0 - ss_res / max(ss_tot, 1e-30)

    return float(slope), float(np.exp(intercept)), {
        "n_fit": int(mask.sum()),
        "r_squared": float(r2),
        "converged": r2 > 0.9,
    }


# =============================================================================
# Analytical reference exponent (for validation)
# =============================================================================

def vartheta_analytical(s: float, alpha: float, beta: float) -> float:
    """
    Corrected analytical exponent from manuscript Eq. (vth):
        vth = s + 1 + s*(2 - alpha - 4*beta)/(2*beta + alpha)
    Equivalent simplified form:
        vth = 1 + s*(3 - alpha)/(2*beta + alpha) + s - s = ...
        leading to NFL condition s*(3 - alpha) < beta + alpha
    """
    return s + 1.0 + s * (2.0 - alpha - 4.0 * beta) / (2.0 * beta + alpha)


# =============================================================================
# Self-test
# =============================================================================

if __name__ == "__main__":
    from backends import get_backend
    xp = get_backend("numpy")

    print("Self-energy module self-test")
    print("=" * 70)

    # Test 1: Ohmic + quadratic (canonical Caldeira-Leggett)
    model = BathModel(s=1.0, alpha=0.0, beta=2.0,
                       eta0=0.1, Omega0=0.0, gapped=False,
                       include_real_KK=False)
    print(f"\nModel: {model.name()}")
    print(f"Analytical vartheta = {vartheta_analytical(1.0, 0.0, 2.0):.4f}")

    omegas = np.geomspace(1e-4, 0.1, 12)
    imsig_vals = []
    for om in omegas:
        val = imsigma_trapezoidal(model, om, xp,
                                  n_p=200, n_nu=400, nu_floor=1e-7)
        imsig_vals.append(val)
        print(f"  omega={om:.2e}  Im Sigma = {val:+.6e}")

    vth_fit, A_fit, info = fit_exponent(omegas, imsig_vals,
                                          omega_range=(1e-4, 1e-2))
    print(f"\n  Fitted vartheta = {vth_fit:.4f}  (R^2 = {info['r_squared']:.4f})")
    print(f"  Amplitude       = {A_fit:.4e}")

    # Test 2: MFL degenerate case (beta = 1)
    print("\n" + "-" * 70)
    print("Test beta=1 (MFL generic prediction)")
    for (s_test, alpha_test) in [(0.5, 0.0), (1.0, 0.5), (1.5, 1.0)]:
        m2 = BathModel(s=s_test, alpha=alpha_test, beta=1.0,
                       eta0=0.1, gapped=False, include_real_KK=False)
        vals = [imsigma_trapezoidal(m2, om, xp, n_p=200, n_nu=400,
                                     nu_floor=1e-7) for om in omegas]
        vth_fit, _, info = fit_exponent(omegas, vals, omega_range=(1e-4, 1e-2))
        print(f"  s={s_test}, alpha={alpha_test}, beta=1: "
              f"vth_fit={vth_fit:.4f}, R^2={info['r_squared']:.4f}  "
              f"(predicted: 1.0)")
