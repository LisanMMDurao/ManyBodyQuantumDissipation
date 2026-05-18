"""
core.electron_gas_GW
====================

Self-energia eletrônica via integral GW completa (correção da abordagem ingênua
de electron_gas.py, que faz apenas multiplicação por g_eff^2).

A self-energia eletrônica em 1-loop é:

    Sigma_e^R(omega) = i ∫ (d^3q/(2π)^3) (dω'/(2π)) |g(q)|^2 G_0^R(k_F+q, omega+ω') D^R_bath(q, ω')

Projetando na superfície de Fermi e integrando angularmente:

    Im Sigma_e^R(omega) = -g^2 * sgn(omega) ∫_0^{q_max} dq q^2/(2 v_F q) (BP factor)
                         × ∫_0^{|omega|} dω' [-Im D^R_bath(q, omega - ω')]

A integral em ω' é A CHAVE: ela CONVOLVE o expoente do banho.

Para banho ômico Im D^R_bath ~ |ω| no IR:
    ∫_0^|omega| dω' |omega - ω'| ~ |omega|^2 / 2
    => Im Sigma_e ~ |omega|^2 (FERMI LIQUID!)

Para banho saturado/incoerente Im D^R_bath ~ const no IR:
    ∫_0^|omega| dω' const ~ |omega|
    => Im Sigma_e ~ |omega| (MARGINAL / STRANGE METAL)

Esta é a propagação correta do expoente, perdida na implementação
multiplicativa anterior.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import simpson
from typing import Union

from .bath import BathParams, sigma_bath, im_sigma_bath, re_sigma_bath
from .electron_gas import CouplingParams

ArrayLike = Union[float, np.ndarray]


# =============================================================
# PROPAGADOR DO BANHO (sem dispersão em q — banho local)
# =============================================================

def D_bath_R_local(omega: np.ndarray, bath: BathParams) -> np.ndarray:
    """
    Propagador retardado do banho local (modo de baixa-q, sem dispersão).

        D^R(omega) = 1 / [omega^2 - omega_0^2 - Sigma_bath^R(omega)]

    onde omega_0^2 representa a massa típica do modo do banho. Para banho
    ôhmico "puro" usamos omega_0 = 0 (modo de Goldstone), gerando

        D^R(omega) = 1 / [omega^2 - Sigma_bath^R(omega)]

    No IR, omega^2 << Sigma_bath, então
        D^R(omega) ≈ -1/Sigma_bath^R(omega)
        Im D^R(omega) ≈ Im Sigma_bath^R / |Sigma_bath|^2 ~ -|omega|/algumconstante

    Linear em omega no IR! (regime ômico do banho).
    """
    sig = sigma_bath(omega, bath)
    # Modo com massa zero (Goldstone-like) para reproduzir banho ômico no elétron
    denom = omega**2 - sig
    # Adicionar pequena regularização em omega = 0
    denom = np.where(np.abs(denom) > 1e-12, denom, 1e-12)
    return 1.0 / denom


def D_bath_R_massive(omega: np.ndarray, bath: BathParams,
                     omega_0_sq: float = None) -> np.ndarray:
    """
    Propagador com massa explícita omega_0^2.
    Para banho com dispersão omega_q = q^2/(2 m_0), avaliando em q ~ k_0:
        omega_0^2 = (k_0^2/(2 m_0))^2 = E_0^2
    """
    if omega_0_sq is None:
        omega_0_sq = bath.E0**2
    sig = sigma_bath(omega, bath)
    denom = omega**2 - omega_0_sq - sig
    denom = np.where(np.abs(denom) > 1e-12, denom, 1e-12)
    return 1.0 / denom


# =============================================================
# SELF-ENERGIA ELETRÔNICA VIA INTEGRAL GW
# =============================================================

def im_sigma_e_GW(omega: ArrayLike, bath: BathParams,
                  coupling: CouplingParams,
                  q_max: float = None,
                  n_q: int = 80, n_omp: int = 200,
                  bath_mode: str = 'massless') -> np.ndarray:
    """
    Im Sigma_e^R(omega) via integral GW completa.

    Im Sigma_e^R(omega) = -g^2 * sgn(omega) * ∫dq q²/(2v_F q) BP(q)/(4π²)
                         * ∫_0^|omega| dω' [-Im D^R_bath(omega - ω')]

    Notas:
        - q_max = 2 k_F (back-scattering máximo) se k_0 >= 2 k_F,
          senão q_max = k_0 (suporte do banho).
        - v_F = k_F/m_e.
        - BP(q) = q^2 / (q^2 + k_TF^2): fator Bardeen-Pines/Thomas-Fermi.
        - bath_mode='massless': D = 1/(omega² - Sigma_bath), banho ômico puro
          (expoente de Im D ~ |omega| no IR, leva a Im Sigma_e ~ omega²)
        - bath_mode='massive': D = 1/(omega² - E_0² - Sigma_bath), modo de banho
          com gap (Im D ~ |omega|·constante no IR)

    Returns
    -------
    Im Sigma_e^R(omega), array da mesma forma que omega.
    """
    omega = np.atleast_1d(np.asarray(omega, dtype=float))
    kF = coupling.kF
    me = coupling.me
    g = coupling.g
    vF = kF / me

    if q_max is None:
        q_max = min(2.0 * kF, bath.k0)

    # grade em q
    q_grid = np.linspace(0.01 * kF, q_max, n_q)
    # Fator Bardeen-Pines com k_TF^2 ~ 4π N(0)
    k_TF_sq = 4 * np.pi * coupling.N0
    BP = q_grid**2 / (q_grid**2 + k_TF_sq)
    # peso angular: q^2 / (2 v_F q) = q/2
    angular_weight = q_grid / (2.0 * vF)

    result = np.zeros_like(omega)

    for i, om in enumerate(omega):
        if np.abs(om) < 1e-10:
            result[i] = 0.0
            continue

        sign_om = np.sign(om)
        # frequências internas omega' indo de 0 a |omega|
        omp_grid = np.linspace(1e-6, np.abs(om), n_omp)
        # argumento do propagador: omega - omega'
        diff_omega = sign_om * (np.abs(om) - omp_grid)

        # Im D^R_bath ao longo do segmento (não depende de q neste modelo simplificado)
        if bath_mode == 'massless':
            D_b = D_bath_R_local(diff_omega, bath)
        else:
            D_b = D_bath_R_massive(diff_omega, bath)
        im_D = np.imag(D_b)

        # integral em omega'
        integral_omp = simpson(-im_D, x=omp_grid)

        # integral em q
        integrand_q = angular_weight * BP / (4.0 * np.pi**2) * integral_omp
        integral_q = simpson(integrand_q, x=q_grid)

        result[i] = -g**2 * integral_q * sign_om

    return result


def re_sigma_e_GW(omega: ArrayLike, im_sigma_e_array: np.ndarray) -> np.ndarray:
    """
    Re Sigma_e^R via transformada de Hilbert (Kramers-Kronig) de Im Sigma_e^R.

    Re Sigma(omega) = (1/π) P∫ Im Sigma(omega')/(omega' - omega) dω'.

    Implementação numérica via scipy.signal.hilbert.
    Requer omega simétrica e uniforme.
    """
    from scipy.signal import hilbert as hilbert_transform
    # Convenção: Im(hilbert(f)) = (1/π) P∫ f(t')/(t-t') dt'
    # Nossa Re Sigma tem -1 vs essa convenção
    return -np.imag(hilbert_transform(im_sigma_e_array))


def sigma_e_GW_full(omega: np.ndarray, bath: BathParams,
                    coupling: CouplingParams, **kwargs):
    """
    Computa (Re Sigma_e^R, Im Sigma_e^R) via integral GW + KK.

    omega deve ser ordenada e uniforme; idealmente simétrica em torno de 0
    para KK preciso.
    """
    if not np.array_equal(omega, np.sort(omega)):
        raise ValueError("omega deve ser ordenada")
    im_S = im_sigma_e_GW(omega, bath, coupling, **kwargs)
    re_S = re_sigma_e_GW(omega, im_S)
    return re_S, im_S


# =============================================================
# DIAGNÓSTICOS
# =============================================================

def extract_exponent_GW(bath: BathParams, coupling: CouplingParams,
                        window_fraction: tuple = (0.01, 0.1),
                        n_points: int = 100,
                        bath_mode: str = 'massless') -> dict:
    """
    Extrai expoente n do fit |Im Sigma_e^R| = A omega^n em janela IR.
    """
    om_low = window_fraction[0] * bath.E0
    om_high = window_fraction[1] * bath.E0
    omega_t = np.linspace(om_low * 0.5, om_high * 2.0, n_points)
    im_S = -im_sigma_e_GW(omega_t, bath, coupling, bath_mode=bath_mode)
    mask = (omega_t > om_low) & (omega_t < om_high) & (im_S > 1e-15)
    if mask.sum() < 5:
        return {'n': np.nan, 'A': np.nan, 'R2': np.nan}

    log_om = np.log(omega_t[mask])
    log_imS = np.log(im_S[mask])
    coeffs = np.polyfit(log_om, log_imS, 1)
    n, log_A = coeffs

    pred = n * log_om + log_A
    ss_res = np.sum((log_imS - pred)**2)
    ss_tot = np.sum((log_imS - log_imS.mean())**2)
    R2 = 1 - ss_res/ss_tot if ss_tot > 0 else np.nan

    return {'n': float(n), 'A': float(np.exp(log_A)), 'R2': float(R2)}


if __name__ == "__main__":
    # Teste rápido
    bath = BathParams(m0=1.0, k0=1.0, alpha=0.1)
    coupling = CouplingParams(g=1.0)
    omega_t = np.linspace(0.001, 0.1, 20)
    im_S = im_sigma_e_GW(omega_t, bath, coupling)
    print("Omega vs Im Sigma_e^R (modo banho massless):")
    for o, s in zip(omega_t, im_S):
        print(f"  omega={o:.4f}: Im Sigma = {s:.4e}")
    fit = extract_exponent_GW(bath, coupling)
    print(f"\nExpoente n = {fit['n']:.3f}, A = {fit['A']:.4e}, R² = {fit['R2']:.4f}")
