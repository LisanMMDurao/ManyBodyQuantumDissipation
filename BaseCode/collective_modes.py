"""
core.collective_modes
=====================

Modos coletivos do gás de elétrons acoplado ao banho de quasipartículas.

Estrutura física:

1. O banho de quasipartículas, com self-energia Sigma_bath^R (Convenção A),
   dá origem a um PROPAGADOR EFETIVO renormalizado:
        D^R_bath(omega, q) = 1 / [omega^2 - omega_q^2 - Sigma_bath^R(omega)]

   onde omega_q = hbar^2 q^2 / (2 m_0) é a dispersão do banho.

2. Este banho MEDEIA uma interação efetiva entre elétrons:
        V_eff(omega, q) = g^2 * D^R_bath(omega, q) * f_q

   onde f_q é um fator de forma do acoplamento elétron-banho.

3. A resposta de densidade do gás de elétrons na presença dessa
   interação efetiva é (em RPA):
        chi(omega, q) = chi_0(omega, q) / [1 - V_eff(omega, q) * chi_0]

   onde chi_0 é a função de Lindhard livre.

4. Os modos coletivos são os polos de chi (zeros do denominador):
        1 - V_eff(omega, q) * chi_0(omega, q) = 0

Para banho ôhmico fortemente acoplado, surgem três classes de modos:
    - PLASMON: existente em q pequeno acima do continuum particle-hole.
    - MODO BÔSON DRESSADO: análogo do fônon dressado pelas
      flutuações eletrônicas (Bardeen-Pines).
    - MODOS HÍBRIDOS: combinações lineares plasmon-banho com gap.

A análise inclui:
    - Dispersão omega_modo(q) por busca de polos.
    - Largura Gamma_modo(q) por FWHM da função espectral.
    - Mapas A_col(omega, q) = -Im chi/pi para visualização.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from typing import Tuple, Optional

from .bath import BathParams, sigma_bath
from .electron_gas import CouplingParams


# =============================================================
# FUNÇÃO DE LINDHARD 3D
# =============================================================

def lindhard_3d(omega: np.ndarray, q: np.ndarray,
                kF: float = 1.0, me: float = 1.0,
                eps: float = 1e-10) -> np.ndarray:
    """
    Função de Lindhard 3D para gás de Fermi livre.

    chi_0(omega, q) / N(0) = -1/2 - (1/(8 qL)) * [
        (1 - (qL - nu)^2) ln|((qL-nu)+1)/((qL-nu)-1)|
      + (1 - (qL + nu)^2) ln|((qL+nu)+1)/((qL+nu)-1)|
    ] - i pi/(8 qL) * [(1 - (qL-nu)^2) chi_inf + ...]

    onde nu = omega / (v_F q), qL = q/(2 k_F).

    A normalização aqui é chi_0/N(0) (adimensional).
    """
    omega = np.asarray(omega, dtype=float)
    q = np.asarray(q, dtype=float)

    vF = kF / me  # velocidade de Fermi (hbar = 1)
    EF = kF**2 / (2*me)

    q_safe = np.where(np.abs(q) > eps, q, eps)
    qL = q_safe / (2.0 * kF)  # adimensional
    nu = omega / (vF * q_safe)

    # ----- Parte real -----
    a1 = qL - nu
    a2 = qL + nu

    def safe_log(a):
        num = a + 1.0
        den = a - 1.0
        den_safe = np.where(np.abs(den) < eps, np.sign(den + eps) * eps, den)
        return np.log(np.abs(num / den_safe))

    term1 = (1.0 - a1**2) * safe_log(a1)
    term2 = (1.0 - a2**2) * safe_log(a2)
    re_chi = -0.5 - (term1 + term2) / (8.0 * qL)

    # ----- Parte imaginária -----
    abs_nu = np.abs(nu)
    sign_nu = np.sign(nu)
    im_chi = np.zeros_like(nu)

    # Região 1: regime IR |nu| + qL < 1
    region_IR = (abs_nu + qL < 1.0) & (qL > eps)
    im_chi = np.where(region_IR, -np.pi * nu / 2.0, im_chi)

    # Região 2: borda alta |1 - qL| < |nu| < 1 + qL
    region_edge = (abs_nu < 1.0 + qL) & (abs_nu > np.abs(1.0 - qL))
    edge_val = -np.pi / (8.0 * qL) * (1.0 - (abs_nu - qL)**2)
    im_chi = np.where(region_edge & ~region_IR, sign_nu * edge_val, im_chi)

    return re_chi + 1j * im_chi


def continuum_boundaries(q: np.ndarray, kF: float = 1.0,
                          me: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Limites do continuum particle-hole em (omega, q).

    omega_+(q) = v_F * q + q^2/(2m) (borda superior)
    omega_-(q) = |v_F * q - q^2/(2m)|  (borda inferior)
    """
    vF = kF / me
    omega_plus = vF * q + q**2 / (2*me)
    omega_minus = np.abs(vF * q - q**2 / (2*me))
    return omega_minus, omega_plus


# =============================================================
# PROPAGADOR DO BANHO DEPENDENTE DE MOMENTUM
# =============================================================

def bath_propagator_q(omega: np.ndarray, q: np.ndarray,
                      bath: BathParams) -> np.ndarray:
    """
    Propagador retardado do banho com dispersão omega_q = hbar^2 q^2/(2 m_0).

        D^R_bath(omega, q) = 1 / [omega^2 - omega_q^2 - Sigma_bath^R(omega)]

    Para q > k_0, o modo não existe (fora do suporte do banho).
    Implementamos isso via fator de corte theta(k_0 - q).
    """
    sig = sigma_bath(omega, bath)
    omega_q_sq = (q**2 / (2*bath.m0))**2  # omega_q^2 com omega_q = q^2/(2m_0)
    # Mais corretamente, energia do modo bosônico em q:
    omega_q = q**2 / (2*bath.m0)

    denom = omega**2 - omega_q**2 - sig
    D = 1.0 / denom

    # Suporte: zero fora de |q| <= k_0
    support = np.abs(q) <= bath.k0
    return np.where(support, D, 0.0 + 0j)


# =============================================================
# INTERAÇÃO EFETIVA E RESPOSTA RPA
# =============================================================

def V_effective(omega: np.ndarray, q: np.ndarray,
                bath: BathParams, coupling: CouplingParams,
                form_factor: str = 'bardeen_pines') -> np.ndarray:
    """
    Interação efetiva mediada pelo banho:
        V_eff(omega, q) = g^2 * f_q * D^R_bath(omega, q)

    form_factor:
        'bardeen_pines': f_q = q^2 / (q^2 + k_TF^2), k_TF = sqrt(4 pi e^2 N(0))
                         estrutura típica de blindagem.
        'constant': f_q = 1, acoplamento local.
        'long_range': f_q = 4 pi / q^2, Coulomb não-blindado.
    """
    D = bath_propagator_q(omega, q, bath)

    if form_factor == 'bardeen_pines':
        # k_TF^2 = 4 pi e^2 N(0); aqui usamos 1 como escala
        k_TF_sq = 4 * np.pi * coupling.N0  # em unidades naturais
        f_q = q**2 / (q**2 + k_TF_sq)
    elif form_factor == 'constant':
        f_q = np.ones_like(q)
    elif form_factor == 'long_range':
        f_q = np.where(np.abs(q) > 1e-6, 4*np.pi / q**2, 0)
    else:
        raise ValueError(f"form_factor desconhecido: {form_factor}")

    return coupling.g**2 * f_q * D


def dielectric_function(omega: np.ndarray, q: np.ndarray,
                        bath: BathParams, coupling: CouplingParams,
                        **kwargs) -> np.ndarray:
    """
    epsilon(omega, q) = 1 - V_eff(omega, q) * chi_0(omega, q)
    """
    V = V_effective(omega, q, bath, coupling, **kwargs)
    chi0 = lindhard_3d(omega, q, kF=coupling.kF, me=coupling.me)
    # chi_0 está normalizada por N(0); precisamos multiplicar
    chi0_phys = chi0 * coupling.N0
    return 1.0 - V * chi0_phys


def response_function(omega: np.ndarray, q: np.ndarray,
                      bath: BathParams, coupling: CouplingParams,
                      **kwargs) -> np.ndarray:
    """
    Resposta de densidade RPA: chi = chi_0 / (1 - V_eff * chi_0)
    """
    chi0 = lindhard_3d(omega, q, kF=coupling.kF, me=coupling.me) * coupling.N0
    V = V_effective(omega, q, bath, coupling, **kwargs)
    eps = 1.0 - V * chi0
    return chi0 / eps


def collective_spectral_function(omega: np.ndarray, q: np.ndarray,
                                  bath: BathParams,
                                  coupling: CouplingParams,
                                  **kwargs) -> np.ndarray:
    """
    A_col(omega, q) = -Im chi(omega, q) / pi

    Função positiva relacionada ao fator de estrutura S(omega, q).
    """
    chi = response_function(omega, q, bath, coupling, **kwargs)
    return -np.imag(chi) / np.pi


def loss_function(omega: np.ndarray, q: np.ndarray,
                  bath: BathParams, coupling: CouplingParams,
                  **kwargs) -> np.ndarray:
    """
    -Im[1/epsilon(omega, q)] — função de perda (EELS).

    Pode ser negativa quando V_eff é atrativa (mediação por banho dressado).
    """
    eps = dielectric_function(omega, q, bath, coupling, **kwargs)
    return -np.imag(1.0 / eps)


# =============================================================
# BUSCA DE MODOS COLETIVOS (POLOS)
# =============================================================

def find_collective_mode(q_value: float, bath: BathParams,
                         coupling: CouplingParams,
                         omega_range: Tuple[float, float] = (0.01, 5.0),
                         n_omega: int = 1000,
                         above_continuum_only: bool = True,
                         **kwargs) -> dict:
    """
    Localiza o modo coletivo em momentum q por busca de pico em A_col.

    Returns
    -------
    dict com 'omega_mode', 'amplitude', 'fwhm', 'found'.
    """
    omega = np.linspace(omega_range[0], omega_range[1], n_omega)
    q_arr = q_value * np.ones_like(omega)

    A = collective_spectral_function(omega, q_arr, bath, coupling, **kwargs)

    if above_continuum_only:
        # Buscar apenas acima do continuum particle-hole
        omega_min_pc, omega_max_pc = continuum_boundaries(
            np.array([q_value]), kF=coupling.kF, me=coupling.me)
        mask = omega > omega_max_pc[0] * 1.005
        if not np.any(mask):
            return {'omega_mode': np.nan, 'amplitude': np.nan,
                    'fwhm': np.nan, 'found': False}
        A_search = A[mask]
        om_search = omega[mask]
    else:
        A_search = A
        om_search = omega

    if A_search.max() < 1e-5:
        return {'omega_mode': np.nan, 'amplitude': np.nan,
                'fwhm': np.nan, 'found': False}

    peak_idx = np.argmax(A_search)
    omega_mode = om_search[peak_idx]
    amp = A_search[peak_idx]

    # FWHM
    half = amp / 2.0
    left = np.where(A_search[:peak_idx] < half)[0]
    right = np.where(A_search[peak_idx:] < half)[0]

    if len(left) > 0 and len(right) > 0:
        fwhm = om_search[peak_idx + right[0]] - om_search[left[-1]]
    else:
        fwhm = np.nan

    return {
        'omega_mode': float(omega_mode), 'amplitude': float(amp),
        'fwhm': float(fwhm) if not np.isnan(fwhm) else np.nan,
        'found': True
    }


def collective_dispersion(q_grid: np.ndarray, bath: BathParams,
                          coupling: CouplingParams,
                          omega_max: float = 5.0,
                          **kwargs) -> dict:
    """
    Dispersão do modo coletivo: omega_mode(q) e fwhm(q).
    """
    omega_modes = np.zeros_like(q_grid)
    fwhms = np.zeros_like(q_grid)
    amps = np.zeros_like(q_grid)

    for i, q in enumerate(q_grid):
        result = find_collective_mode(q, bath, coupling,
                                       omega_range=(0.01, omega_max), **kwargs)
        omega_modes[i] = result['omega_mode']
        fwhms[i] = result['fwhm']
        amps[i] = result['amplitude']

    return {'q_grid': q_grid, 'omega_mode': omega_modes,
            'fwhm': fwhms, 'amplitude': amps}


if __name__ == "__main__":
    bath = BathParams(m0=1.0, k0=2.0, alpha=0.3)
    coupling = CouplingParams(g=1.5)

    # teste de modos
    q_test = np.linspace(0.2, 1.5, 5)
    for q in q_test:
        mode = find_collective_mode(q, bath, coupling)
        print(f"q={q:.2f}: omega={mode['omega_mode']:.3f}, "
              f"FWHM={mode['fwhm']:.3f}, A={mode['amplitude']:.3f}")
