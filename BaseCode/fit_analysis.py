"""
core.fit_analysis
=================

Análise por ajuste log-log do expoente n em |Im Sigma_e^R(omega)| ~ omega^n
no infravermelho.

Esta análise replica a metodologia do código original (Rota DEF) e é
COMPLEMENTAR à classificação baseada em lambda. As duas devem concordar,
mas o fit dá uma assinatura EMPÍRICA mais próxima do que se mede em
experimentos de ARPES/condutividade.

CONVENÇÕES de classificação:
    n > 1.7 : Fermi Liquid (Im Sigma ~ omega^2)
    1.3 < n < 1.7 : Marginal Fermi Liquid (intermediário)
    n < 1.3 : Strange Metal-like (Im Sigma ~ omega^1, linear)

Note que na Convenção A com banho ôhmico exponencialmente regularizado,
o expoente EFETIVO depende da janela:
    - Janela perto de zero: n -> 1 (linear, ômico puro)
    - Janela próxima de E_0: n cai (cutoff em ação)
    - Janela muito acima de E_0: n -> negativo (decaimento exponencial)

A escolha da janela é crucial e define o que se está medindo.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Optional

from .bath import BathParams
from .electron_gas import CouplingParams, im_sigma_e, re_sigma_e


# =============================================================
# AJUSTE DE LEI DE POTÊNCIA
# =============================================================

def power_law_fit(omega: np.ndarray, im_sigma: np.ndarray,
                  window: Tuple[float, float]) -> dict:
    """
    Ajusta |Im Sigma(omega)| = A * omega^n em uma janela [omega_min, omega_max].

    Returns
    -------
    dict com 'n' (expoente), 'A' (pré-fator), 'R2' (qualidade do fit),
    e 'window' (janela efetiva usada).
    """
    omega = np.atleast_1d(omega)
    mask = (omega > window[0]) & (omega < window[1])

    if mask.sum() < 5:
        return {'n': np.nan, 'A': np.nan, 'R2': np.nan,
                'window': window, 'n_points': mask.sum()}

    om_fit = omega[mask]
    imS_fit = np.abs(im_sigma[mask])

    # remover pontos onde Im Sigma é essencialmente zero (problemas log)
    nonzero = imS_fit > 1e-15
    if nonzero.sum() < 5:
        return {'n': np.nan, 'A': np.nan, 'R2': np.nan,
                'window': window, 'n_points': nonzero.sum()}

    log_om = np.log(om_fit[nonzero])
    log_imS = np.log(imS_fit[nonzero])

    # ajuste linear
    coeffs, cov = np.polyfit(log_om, log_imS, 1, cov=True)
    n, log_A = coeffs

    # R^2
    log_imS_pred = n * log_om + log_A
    ss_res = np.sum((log_imS - log_imS_pred)**2)
    ss_tot = np.sum((log_imS - log_imS.mean())**2)
    R2 = 1.0 - ss_res/ss_tot if ss_tot > 0 else np.nan

    return {
        'n': float(n), 'A': float(np.exp(log_A)),
        'R2': float(R2), 'n_err': float(np.sqrt(cov[0,0])),
        'window': window, 'n_points': int(nonzero.sum()),
    }


def classify_by_amplitude(A_prefactor: float,
                          thresh_FL: float = 0.1,
                          thresh_incoh: float = 1.0) -> str:
    """
    Classificação por AMPLITUDE do fit Im Sigma = A * omega^n.

    Na convenção A onde Im Sigma_e = lambda * |omega| * exp(-omega/E_0),
    o expoente n é universal ~1, mas A é exatamente lambda.

    Portanto A é o melhor diagnóstico:
        A < 0.1: FL coerente
        0.1 <= A < 1: MFL marginal
        A >= 1: incoerente
    """
    if np.isnan(A_prefactor):
        return 'undefined'
    if A_prefactor < thresh_FL:
        return 'FL'
    elif A_prefactor < thresh_incoh:
        return 'MFL'
    else:
        return 'incoherent'


def classify_by_exponent(n: float,
                         thresh_FL: float = 1.7,
                         thresh_SM: float = 1.3) -> str:
    """Classificação empírica via expoente n."""
    if np.isnan(n):
        return 'undefined'
    if n > thresh_FL:
        return 'FL'
    elif n < thresh_SM:
        return 'SM'
    else:
        return 'MFL'


def extract_n_for_params(bath: BathParams, coupling: CouplingParams,
                          window_fraction: Tuple[float, float] = (0.02, 0.3),
                          n_points: int = 200) -> dict:
    """
    Extrai expoente n para um par (banho, acoplamento).

    A janela é especificada como FRAÇÃO de E_0:
        window = (frac_low * E_0, frac_high * E_0).
    Janela default (0.02, 0.3)*E_0 mede o regime claramente IR.
    """
    omega_low = window_fraction[0] * bath.E0
    omega_high = window_fraction[1] * bath.E0
    # grade fina dentro da janela e um pouco além
    omega = np.linspace(omega_low * 0.5, omega_high * 2.0, n_points)
    im_S = im_sigma_e(omega, bath, coupling)
    fit_result = power_law_fit(omega, im_S, (omega_low, omega_high))
    return fit_result


# =============================================================
# VARREDURAS COM CLASSIFICAÇÃO POR EXPOENTE
# =============================================================

def scan_exponent_alpha_g(alpha_grid: np.ndarray, g_grid: np.ndarray,
                          m0: float = 1.0, k0_over_kF: float = 1.0,
                          kF: float = 1.0,
                          window_fraction: Tuple[float, float] = (0.02, 0.3)) -> dict:
    """
    Varredura no plano (alpha, g) extraindo expoente n por fit.

    Returns
    -------
    dict com 'n_map', 'phase_n' (código 0=FL, 1=MFL, 2=SM), 'R2_map'.
    """
    n_a, n_g = len(alpha_grid), len(g_grid)
    n_map = np.zeros((n_a, n_g))
    R2_map = np.zeros((n_a, n_g))
    phase_n = np.zeros((n_a, n_g), dtype=int)

    code_dict = {'FL': 0, 'MFL': 1, 'SM': 2, 'undefined': 3}

    for i, alpha in enumerate(alpha_grid):
        bath = BathParams(m0=m0, k0=k0_over_kF*kF, alpha=alpha, kF=kF)
        for j, g in enumerate(g_grid):
            coupling = CouplingParams(g=g, kF=kF)
            fit = extract_n_for_params(bath, coupling, window_fraction)
            n_map[i, j] = fit['n']
            R2_map[i, j] = fit['R2']
            phase_n[i, j] = code_dict[classify_by_exponent(fit['n'])]

    return {
        'alpha_grid': alpha_grid, 'g_grid': g_grid,
        'n_map': n_map, 'R2_map': R2_map, 'phase_n': phase_n,
        'm0': m0, 'k0_over_kF': k0_over_kF,
        'window_fraction': window_fraction,
    }


def temperature_proxy_resistivity(T_grid: np.ndarray,
                                   bath: BathParams,
                                   coupling: CouplingParams) -> dict:
    """
    Proxy térmico: rho(T) ~ |Im Sigma_e^R(omega = T)|

    Permite extrair expoente rho ~ T^p, onde:
        p ≈ 2: comportamento FL
        p ≈ 1: comportamento linear (strange metal)

    Em escala log-log, a inclinação é o expoente p.
    """
    rho_T = np.abs(im_sigma_e(T_grid, bath, coupling))
    mask = (T_grid > 0.02 * bath.E0) & (T_grid < 0.3 * bath.E0)
    if mask.sum() >= 5:
        slope, _ = np.polyfit(np.log(T_grid[mask]), np.log(rho_T[mask]), 1)
    else:
        slope = np.nan
    return {'T_grid': T_grid, 'rho_T': rho_T, 'exponent_p': float(slope)}


if __name__ == "__main__":
    # Teste rápido
    bath = BathParams(m0=1.0, k0=1.0, alpha=0.3)
    coupling = CouplingParams(g=2.0)
    fit = extract_n_for_params(bath, coupling)
    print(f"n = {fit['n']:.3f} ± {fit['n_err']:.3f}, R² = {fit['R2']:.4f}")
    print(f"Fase (por n): {classify_by_exponent(fit['n'])}")
