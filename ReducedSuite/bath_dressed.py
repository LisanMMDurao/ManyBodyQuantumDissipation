"""
bath_dressed.py
---------------
Bosonic bath propagator dressed by RPA polarization of the electron gas.

The RPA-dressed retarded propagator is:

    D^R_dressed(q, omega) = 1 / [omega^2 - Omega_q^2 - g^2 Pi^R(q, omega) 
                                  + i*omega*Gamma_intrinsic(q, omega)]

where:
  - Omega_q is the bare bath dispersion (constant / linear / quadratic / gapped)
  - g is the electron-boson coupling
  - Pi^R(q, omega) is the full Lindhard polarization (from rpa_polarization.py)
  - Gamma_intrinsic is any additional bath self-damping (e.g. ohmic)

The key feature: g^2 * Re Pi(q, 0) shifts Omega_q^2 -> Omega_q^2 - g^2 |Pi(q,0)|.
Since Re Pi(q, 0) < 0 (with magnitude maximal near q = 2 k_F due to the Kohn
anomaly), this RPA correction *reduces* the squared frequency, and large enough
g drives Omega_q^2_dressed -> 0 at some q* (RPA instability / soft mode).

The bosonic QPT is defined by

    r(g, k0, beta) := min_q [Omega_q^2 - g^2 Re Pi(q, 0)]

with:
    r > 0  : stable bath  (gapped, FL-like regime for the electrons)
    r = 0  : QPT          (soft mode)
    r < 0  : instability  (formal -- requires interpretation as ordered phase)

Standard scaling: T* ~ sqrt(|r|).
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Union, Optional

from rpa_polarization import lindhard_3D, re_Pi_static, F_static

ArrayLike = Union[float, np.ndarray]


# =============================================================================
# Bath model parameters
# =============================================================================

@dataclass
class BathRPAModel:
    """
    Parameters specifying the bath + RPA model.

    Parametrization options:
      dispersion ∈ {"constant", "linear", "gapped", "quadratic"}
      damping    ∈ {"ohmic", "drude", "none"}

    All quantities in natural units E_F = 1, k_F = 1, hbar = 1.
    """
    # Bath dispersion
    dispersion:    str    = "gapped"   # "constant", "linear", "gapped", "quadratic"
    Omega0:        float  = 0.5        # bare gap (constant / gapped)
    c_disp:        float  = 1.0        # slope (linear / quadratic)
    M:             float  = 1.0        # bath mass (quadratic)
    k0:            float  = 1.0        # IR cutoff scale (where damping is normalized)
    # Damping
    damping:       str    = "ohmic"    # "ohmic", "drude", "none"
    gamma0:        float  = 0.1        # ohmic damping strength
    GammaD:        float  = 0.1        # drude amplitude
    omegaD:        float  = 1.0        # drude frequency
    # Coupling to electron gas
    g:             float  = 1.0        # electron-boson vertex
    # Electron gas
    kF:            float  = 1.0
    m_e:           float  = 1.0
    # RPA flag
    use_RPA:       bool   = True       # include g^2 Pi(q, omega) in denominator

    # Numerical
    eps:           float  = 1e-12

    def __post_init__(self):
        self.vF = self.kF / self.m_e
        self.EF = self.kF ** 2 / (2.0 * self.m_e)
        self.N0 = self.m_e * self.kF / (2.0 * np.pi ** 2)

    def label(self) -> str:
        return (f"{self.dispersion}/{self.damping} "
                f"Omega0={self.Omega0:g} g={self.g:g} k0={self.k0:g}")


# =============================================================================
# Bare bath dispersion (Omega_q, BEFORE RPA dressing)
# =============================================================================

def Omega_q_bare(q: ArrayLike, p: BathRPAModel) -> np.ndarray:
    """Bare bath frequency Omega_q (positive square root)."""
    q = np.asarray(q, dtype=float)
    if p.dispersion == "constant":
        return np.full_like(q, p.Omega0)
    elif p.dispersion == "linear":
        return p.c_disp * q
    elif p.dispersion == "gapped":
        return np.sqrt(p.Omega0 ** 2 + (p.c_disp * q) ** 2)
    elif p.dispersion == "quadratic":
        return q ** 2 / (2.0 * p.M)
    else:
        raise ValueError(f"Unknown dispersion: {p.dispersion}")


def Omega_q_bare_sq(q: ArrayLike, p: BathRPAModel) -> np.ndarray:
    """Bare Omega_q^2 (same as Omega_q_bare**2, but explicit)."""
    return Omega_q_bare(q, p) ** 2


# =============================================================================
# Intrinsic bath damping (BEFORE RPA)
# =============================================================================

def Gamma_intrinsic(omega: ArrayLike, p: BathRPAModel) -> np.ndarray:
    """
    Intrinsic bath damping function (independent of fermionic RPA).

    Ohmic:   Gamma(omega) = gamma0 * omega (linear in omega)
    Drude:   Gamma(omega) = GammaD / (1 + i omega/omegaD)  -- complex!
    None:    Gamma = 0
    """
    omega = np.asarray(omega, dtype=float)
    if p.damping == "ohmic":
        return p.gamma0 * omega
    elif p.damping == "drude":
        # NB: Drude damping is intrinsically complex (frequency-dependent)
        return p.GammaD / (1.0 + 1j * (omega / p.omegaD))
    elif p.damping == "none":
        return np.zeros_like(omega)
    else:
        raise ValueError(f"Unknown damping: {p.damping}")


# =============================================================================
# Dressed bath propagator
# =============================================================================

def D_R_dressed(omega: ArrayLike, q: ArrayLike,
                p: BathRPAModel) -> np.ndarray:
    """
    Retarded bath propagator with RPA dressing:

        D^R(q, omega) = 1 / [omega^2 - Omega_q^2 - g^2 Pi^R(q, omega) 
                              + i omega Gamma_intrinsic(omega)]

    Returns complex array shape = broadcast(omega, q).

    NOTE on convention: the +i omega Gamma_intrinsic carries the sign of
    omega (since Gamma_intrinsic ~ gamma0 * omega is odd), so for omega > 0
    the imaginary part of the denominator is positive, and Im D^R < 0
    (causal retarded -- consistent with our manuscript conventions).
    """
    omega = np.asarray(omega, dtype=float)
    q = np.asarray(q, dtype=float)

    Omega_sq = Omega_q_bare_sq(q, p)

    # RPA correction
    if p.use_RPA:
        Pi = lindhard_3D(omega, q, kF=p.kF, m=p.m_e)
        rpa_term = p.g ** 2 * Pi  # complex
    else:
        rpa_term = np.zeros_like(Omega_sq, dtype=complex)

    # Intrinsic damping (could be complex for Drude)
    Gam = Gamma_intrinsic(omega, p)

    # Build denominator. Note: + i*omega*Gam_intrinsic, where Gam_intrinsic
    # may itself be complex (Drude). For ohmic Gam is real, so the imaginary
    # part of the term is omega*Gam = gamma0*omega^2 (positive for any omega).
    denom = omega ** 2 - Omega_sq - rpa_term + 1j * omega * Gam

    return 1.0 / (denom + p.eps * 1j)


def Im_D_R_dressed(omega: ArrayLike, q: ArrayLike,
                   p: BathRPAModel) -> np.ndarray:
    """Im D^R_dressed(q, omega). Should be <= 0 for omega > 0."""
    return np.imag(D_R_dressed(omega, q, p))


# =============================================================================
# Mass parameter r(p) and QPT locus
# =============================================================================

def mass_function(q: ArrayLike, p: BathRPAModel) -> np.ndarray:
    """
    The 'mass function' M(q) := Omega_q^2 - g^2 Re Pi(q, omega=0).

    The minimum of M(q) over q is the control parameter r.
    M(q) > 0 everywhere  : stable bath
    M(q*) = 0 at some q* : QPT critical point
    M(q*) < 0           : unstable (ordered phase)
    """
    q = np.asarray(q, dtype=float)
    Omega_sq = Omega_q_bare_sq(q, p)
    re_Pi = re_Pi_static(q, kF=p.kF, m=p.m_e)
    return Omega_sq - p.g ** 2 * re_Pi  # note: re_Pi < 0, so subtracting *adds*... wait

    # Wait: Re Pi(q, 0) = -N_0 F_static(q/2kF) < 0
    # The denominator of D^R is omega^2 - Omega_q^2 - g^2 Pi^R
    # = omega^2 - Omega_q^2 - g^2*(negative real) - i*g^2*Im Pi
    # = omega^2 - Omega_q^2 + g^2 |Pi_R(q,0)| - i*g^2*Im Pi
    # So the effective mass term is M(q) = Omega_q^2 - g^2 Re Pi
    # With Re Pi < 0, this *increases* M(q) -- repulsive, no instability!
    # If g^2 Re Pi is to drive an instability, we'd need Re Pi > 0, which
    # is NOT the case for the bare Lindhard.
    # So with this sign convention, the RPA dressing STABILIZES the bath,
    # not destabilizes it. To get a QPT, we need an attractive sign somewhere.


def mass_function_corrected(q: ArrayLike, p: BathRPAModel) -> np.ndarray:
    """
    The mass function with the convention that DESTABILIZING RPA correction
    requires careful sign analysis.

    For a phonon-like coupling with vertex g where the interaction is
    *attractive* in some channel, we may have an effective +g^2 |Pi|
    correction lowering Omega_q^2. Specifically, in the standard Bardeen-Pines
    structure, the corrected boson frequency obeys:

        Omega_q^2_dressed = Omega_q^2 / [1 - g^2 |Pi(q,0)| / Omega_q^2]^? 

    or, in linearized form: Omega_q^2 - g^2 |Pi|. Here |Pi| = -Re Pi > 0.

    The CONVENTION USED by the user's framework (damped_phasediagram_torch.py)
    is: M(q) = Omega_q^2 - g^2 |Pi(q,0)|, with the soft mode at
        Omega_q^2 = g^2 |Pi(q,0)|.

    We adopt this convention here, returning:
        M_corrected(q) = Omega_q^2 - g^2 * (-Re Pi(q, 0))
                       = Omega_q^2 + g^2 * Re Pi(q, 0)

    With Re Pi < 0, the correction is NEGATIVE, can drive M to zero.
    """
    q = np.asarray(q, dtype=float)
    Omega_sq = Omega_q_bare_sq(q, p)
    re_Pi = re_Pi_static(q, kF=p.kF, m=p.m_e)
    # |Re Pi| = -Re Pi (since Re Pi < 0)
    return Omega_sq + p.g ** 2 * re_Pi  # adds a negative number -> reduces M


def control_parameter_r(p: BathRPAModel,
                        q_grid: Optional[np.ndarray] = None) -> dict:
    """
    Compute r = min_q M(q) and locate the critical momentum q*.

    Returns dict with:
        r    : minimum value of M(q)
        q_star : the q where minimum occurs
        M_curve : the full M(q) curve over q_grid
        q_grid : the grid used
    """
    if q_grid is None:
        q_grid = np.linspace(0.01, 3.0 * p.kF, 500)

    M = mass_function_corrected(q_grid, p)
    idx = int(np.argmin(M))
    return {
        "r":       float(M[idx]),
        "q_star":  float(q_grid[idx]),
        "M_curve": M,
        "q_grid":  q_grid,
    }


# =============================================================================
# Self-test
# =============================================================================

if __name__ == "__main__":
    print("Dressed bath propagator self-test")
    print("=" * 70)

    # Case 1: gapped bath, weak coupling -- expect r > 0 (stable)
    p1 = BathRPAModel(dispersion="gapped", Omega0=0.5, c_disp=0.5,
                       damping="ohmic", gamma0=0.1, g=0.5,
                       kF=1.0, m_e=1.0, use_RPA=True)
    info1 = control_parameter_r(p1)
    print(f"\nCase 1: {p1.label()}")
    print(f"  r = {info1['r']:+.6f}  at q* = {info1['q_star']:.4f}")
    Omega_sq_qstar = float(Omega_q_bare_sq(np.array([info1['q_star']]), p1)[0])
    re_Pi_qstar = float(re_Pi_static(np.array([info1['q_star']]), kF=1)[0])
    print(f"  Omega_q^2 at q* = {Omega_sq_qstar:.4f}")
    print(f"  g^2 * Re Pi(q*,0) = {p1.g**2 * re_Pi_qstar:+.4f}")

    # Case 2: tune g upward until r approaches zero
    print(f"\nCase 2: g-sweep at Omega0={p1.Omega0}, c={p1.c_disp}, dispersion=gapped")
    g_values = np.linspace(0.1, 10.0, 11)
    for g in g_values:
        p_test = BathRPAModel(dispersion="gapped", Omega0=0.5, c_disp=0.5,
                               damping="ohmic", gamma0=0.1, g=g,
                               kF=1.0, m_e=1.0, use_RPA=True)
        info = control_parameter_r(p_test)
        print(f"  g={g:5.2f}:  r={info['r']:+.4f}  q*={info['q_star']:.3f}")

    # Case 3: Verify Im D < 0 for omega > 0 (causality)
    print(f"\nCase 3: Im D^R at (omega=0.1, q=1.0) for stable case:")
    p_stable = BathRPAModel(dispersion="gapped", Omega0=1.0, c_disp=0.5,
                             damping="ohmic", gamma0=0.1, g=0.5,
                             kF=1.0, m_e=1.0, use_RPA=True)
    D_val = D_R_dressed(np.array([0.1]), np.array([1.0]), p_stable)[0]
    print(f"  D^R = {D_val:+.6f}  (Im part: {D_val.imag:+.6e})")
    print(f"  Causality: {'OK' if D_val.imag < 0 else 'FAIL'}")

    # Case 4: Compare bare vs RPA-dressed near q = 2 kF (Kohn anomaly region)
    print(f"\nCase 4: Bare vs RPA-dressed at q = 2 k_F = 2.0:")
    p_bare = BathRPAModel(dispersion="gapped", Omega0=0.5, c_disp=0.5,
                           damping="ohmic", gamma0=0.1, g=2.0,
                           kF=1.0, m_e=1.0, use_RPA=False)
    p_dress = BathRPAModel(dispersion="gapped", Omega0=0.5, c_disp=0.5,
                            damping="ohmic", gamma0=0.1, g=2.0,
                            kF=1.0, m_e=1.0, use_RPA=True)
    for omega_test in [0.01, 0.1, 0.5, 1.0]:
        D_bare = D_R_dressed(np.array([omega_test]), np.array([2.0]), p_bare)[0]
        D_dr = D_R_dressed(np.array([omega_test]), np.array([2.0]), p_dress)[0]
        print(f"  omega={omega_test:.2f}:  D_bare={D_bare:+.4f}  D_dressed={D_dr:+.4f}")
