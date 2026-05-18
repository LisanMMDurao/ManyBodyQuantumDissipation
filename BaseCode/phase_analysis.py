"""
core.phase_analysis
===================

Análise de fases do gás de elétrons acoplado ao banho de quasipartículas.

A convenção A produz Z = 1 estritamente em omega = 0 (porque
Re Sigma ~ omega^2 ln omega tem derivada nula em zero). O diagnóstico
fisicamente relevante é portanto:

(i)  A ESCALA DE LARGURA: Gamma(omega) = 2|Im Sigma| = 2 pi g_eff^2 alpha |omega| exp(-|omega|/E_0).
     Define a "fronteira de coerência" omega_coh acima da qual Gamma ~ omega.

(ii) A ESCALA DE COERÊNCIA: omega_coh determinada por g_eff^2 alpha pi = eta_threshold.
     Para g_eff^2 alpha pi > 1, NÃO existe regime coerente em escala alguma.

(iii) Z PRÓXIMO MAS NÃO EM zero: avaliando Z em omega = 0.1 E_0 (longe do logaritmo),
      capturamos a redução real do peso espectral.

Regimes físicos:
    FL clássico ("Fermi liquid coerente"):
        - g_eff^2 alpha pi << 1, omega_coh ~ E_0
        - Gamma(omega) << omega para omega << omega_coh
        - polos bem definidos com largura pequena

    MFL (Marginal Fermi Liquid):
        - g_eff^2 alpha pi ~ 0.5: linhas de coerência marginal
        - Gamma(omega) ~ omega: estrutura linear sem janela

    Incoerente / Strange-metal-like:
        - g_eff^2 alpha pi >> 1
        - Gamma(omega) > omega em toda a janela ate E_0
        - quasipartículas mal definidas

VARIÁVEIS DE CONTROLE:
    - alpha: acoplamento sistema-banho
    - g: acoplamento elétron-mediador
    - k0/kF: cutoff em momentum (afeta F)
    - m0/m_e: massa do banho (afeta E_0)
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple

from .bath import BathParams
from .electron_gas import (
    CouplingParams, im_sigma_e, re_sigma_e, sigma_e,
    momentum_cutoff_factor, quasiparticle_Z_at_FS,
    quasiparticle_pole, coherence_window, decay_rate
)


# =============================================================
# MÉTRICAS DE FASE
# =============================================================

def dimensionless_coupling(bath: BathParams, coupling: CouplingParams) -> float:
    """
    Acoplamento efetivo adimensional:
        lambda = g_eff^2 * alpha * pi
              = g^2 * N(0) * F(k_0/k_F) * alpha * pi

    Critério de fase:
        lambda < 0.1: FL coerente
        0.1 <= lambda < 1: MFL / marginal
        lambda >= 1: incoerente
    """
    g_eff_sq = coupling.g**2 * coupling.N0 * momentum_cutoff_factor(bath, coupling)
    return g_eff_sq * bath.alpha * np.pi


def classify_phase(bath: BathParams, coupling: CouplingParams,
                   thresh_coh: float = 0.1,
                   thresh_incoh: float = 1.0) -> str:
    """Retorna 'FL', 'MFL', 'incoherent' com base em lambda."""
    lam = dimensionless_coupling(bath, coupling)
    if lam < thresh_coh:
        return 'FL'
    elif lam < thresh_incoh:
        return 'MFL'
    else:
        return 'incoherent'


def quasiparticle_Z_at_scale(omega_scale: float, bath: BathParams,
                              coupling: CouplingParams) -> float:
    """
    Z avaliado em uma escala finita omega_scale > 0.

    Captura redução do peso espectral em frequências finitas,
    onde Re Sigma diverge logaritmicamente.
    """
    from .electron_gas import dRe_sigma_e_domega
    dRe = dRe_sigma_e_domega(np.array([omega_scale]), bath, coupling)[0]
    denom = 1.0 - dRe
    if denom <= 0:
        return np.nan
    Z = 1.0 / denom
    return min(Z, 1.0)


def spectral_weight_at_omega_E0(bath: BathParams, coupling: CouplingParams) -> float:
    """
    Z em omega = E_0/10 (escala intermediária).
    """
    return quasiparticle_Z_at_scale(0.1 * bath.E0, bath, coupling)


def gamma_over_omega_ratio(omega: float, bath: BathParams,
                            coupling: CouplingParams) -> float:
    """eta(omega) = Gamma(omega) / (2 omega) — parâmetro de coerência."""
    if abs(omega) < 1e-12:
        return np.inf
    Gamma = decay_rate(np.array([omega]), bath, coupling)[0]
    return Gamma / (2.0 * abs(omega))


# =============================================================
# VARREDURAS BIDIMENSIONAIS
# =============================================================

def scan_alpha_g(alpha_grid: np.ndarray, g_grid: np.ndarray,
                 m0: float = 1.0, k0_over_kF: float = 1.0,
                 kF: float = 1.0) -> dict:
    """
    Varredura no plano (alpha, g) com m_0 e k_0 fixos.

    Returns
    -------
    dict com matrizes:
        'lambda'   : lambda(alpha, g)
        'Z_omega'  : Z avaliado em omega = E_0/10
        'omega_coh': janela de coerência (em unidades de E_0)
        'phase'    : código (0=FL, 1=MFL, 2=incoherent)
    """
    n_a, n_g = len(alpha_grid), len(g_grid)
    lam_map = np.zeros((n_a, n_g))
    Z_map = np.zeros((n_a, n_g))
    coh_map = np.zeros((n_a, n_g))
    phase_map = np.zeros((n_a, n_g), dtype=int)

    for i, alpha in enumerate(alpha_grid):
        bath = BathParams(m0=m0, k0=k0_over_kF*kF, alpha=alpha, kF=kF)
        for j, g in enumerate(g_grid):
            coupling = CouplingParams(g=g, kF=kF)
            lam_map[i, j] = dimensionless_coupling(bath, coupling)
            Z_map[i, j] = spectral_weight_at_omega_E0(bath, coupling)
            coh = coherence_window(bath, coupling)
            coh_map[i, j] = coh if np.isfinite(coh) else 10*bath.E0
            phase_str = classify_phase(bath, coupling)
            phase_map[i, j] = {'FL': 0, 'MFL': 1, 'incoherent': 2}[phase_str]

    return {
        'alpha_grid': alpha_grid, 'g_grid': g_grid,
        'lambda': lam_map, 'Z_omega': Z_map,
        'omega_coh': coh_map, 'phase': phase_map,
        'm0': m0, 'k0_over_kF': k0_over_kF,
    }


def scan_k0_m0(k0_over_kF_grid: np.ndarray, m0_grid: np.ndarray,
               alpha: float = 0.3, g: float = 1.0,
               kF: float = 1.0) -> dict:
    """
    Varredura no plano (k_0/k_F, m_0/m_e) com alpha e g fixos.

    Captura como a GEOMETRIA do banho (escala de momentum e massa)
    afeta o regime do gás de elétrons.
    """
    n_k, n_m = len(k0_over_kF_grid), len(m0_grid)
    lam_map = np.zeros((n_k, n_m))
    Z_map = np.zeros((n_k, n_m))
    E0_map = np.zeros((n_k, n_m))
    F_map = np.zeros((n_k, n_m))
    phase_map = np.zeros((n_k, n_m), dtype=int)

    coupling = CouplingParams(g=g, kF=kF)

    for i, k0r in enumerate(k0_over_kF_grid):
        for j, m0 in enumerate(m0_grid):
            bath = BathParams(m0=m0, k0=k0r*kF, alpha=alpha, kF=kF)
            lam_map[i, j] = dimensionless_coupling(bath, coupling)
            Z_map[i, j] = spectral_weight_at_omega_E0(bath, coupling)
            E0_map[i, j] = bath.E0
            F_map[i, j] = momentum_cutoff_factor(bath, coupling)
            phase_str = classify_phase(bath, coupling)
            phase_map[i, j] = {'FL': 0, 'MFL': 1, 'incoherent': 2}[phase_str]

    return {
        'k0_grid': k0_over_kF_grid, 'm0_grid': m0_grid,
        'lambda': lam_map, 'Z_omega': Z_map,
        'E0': E0_map, 'F': F_map, 'phase': phase_map,
        'alpha': alpha, 'g': g,
    }


def scan_omega_alpha(omega_grid: np.ndarray, alpha_grid: np.ndarray,
                     bath_kwargs: Optional[dict] = None,
                     coupling_kwargs: Optional[dict] = None) -> dict:
    """
    Varredura na razão Gamma/omega no plano (omega, alpha).

    Permite visualizar onde a estrutura de quasipartícula sobrevive:
        eta < 0.5: coerente
        eta > 1: incoerente
    """
    bath_kwargs = bath_kwargs or {'m0': 1.0, 'k0': 1.0}
    coupling_kwargs = coupling_kwargs or {'g': 1.0}

    eta_map = np.zeros((len(omega_grid), len(alpha_grid)))

    for j, alpha in enumerate(alpha_grid):
        bath = BathParams(alpha=alpha, **bath_kwargs)
        coupling = CouplingParams(**coupling_kwargs)
        for i, omega in enumerate(omega_grid):
            eta_map[i, j] = gamma_over_omega_ratio(omega, bath, coupling)

    return {
        'omega_grid': omega_grid, 'alpha_grid': alpha_grid,
        'eta': eta_map,
    }


if __name__ == "__main__":
    bath = BathParams(m0=1.0, k0=1.0, alpha=0.1)
    coupling = CouplingParams(g=1.0)
    print('lambda =', dimensionless_coupling(bath, coupling))
    print('Fase  =', classify_phase(bath, coupling))
    print('Z@(omega=0.1 E0):', spectral_weight_at_omega_E0(bath, coupling))
