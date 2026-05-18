"""
core.rpa_response
=================

Resposta linear RPA do gás de elétrons. A ÚNICA interação é mediada pelo banho:

    V_eff(omega, q) = g^2 * D^R_bath(omega)

(não há Coulomb explícito nem outras interações)

Estrutura RPA:

    chi^R(omega, q) = chi_0(omega, q) / [1 - V_eff(omega) * chi_0(omega, q)]

    epsilon(omega, q) = 1 - V_eff(omega) * chi_0(omega, q)

onde chi_0 é a função de Lindhard 3D do gás livre.

Como o banho não tem dispersão explícita em q neste modelo (modo local),
V_eff depende SÓ de omega. A dependência em q vem inteiramente de chi_0.

Modos coletivos: zeros de epsilon(omega, q) ou polos de chi.
    1 - V_eff(omega) * Re chi_0(omega, q) = 0  AND  V_eff * Im chi_0 = 0
        (condições para polo bem-definido)

Tempo de vida do plasmon:
    Para omega complexo com chi_0(omega - i Gamma_p/2, q) = 1/V_eff(omega),
    Gamma_p é a largura do modo, controlada pela dissipação ôhmica.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from typing import Tuple, Optional, Union

from .bath import BathParams, sigma_bath
from .electron_gas import CouplingParams
from .electron_gas_GW import D_bath_R_massive, D_bath_R_local


# =============================================================
# FUNÇÃO DE LINDHARD 3D
# =============================================================

def lindhard_3d(omega: np.ndarray, q: np.ndarray,
                kF: float = 1.0, me: float = 1.0,
                eps: float = 1e-10) -> np.ndarray:
    """
    Função de Lindhard 3D normalizada por N(0) = m kF/(2 pi^2).

    chi_0(omega, q) / N(0):
        Real part: forma fechada via combinações log.
        Imag part: zero fora do continuum p-h; -pi nu/2 dentro do regime IR.

    Variáveis internas:
        nu = omega / (v_F q)
        qL = q / (2 kF)
    """
    omega = np.asarray(omega, dtype=float)
    q = np.asarray(q, dtype=float)
    vF = kF / me

    q_safe = np.where(np.abs(q) > eps, q, eps)
    qL = q_safe / (2.0 * kF)
    nu = omega / (vF * q_safe)

    # ------ Real part ------
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

    # ------ Imaginary part ------
    abs_nu = np.abs(nu)
    sign_nu = np.sign(nu)
    im_chi = np.zeros_like(nu)

    region_IR = (abs_nu + qL < 1.0) & (qL > eps)
    im_chi = np.where(region_IR, -np.pi * nu / 2.0, im_chi)

    region_edge = (abs_nu < 1.0 + qL) & (abs_nu > np.abs(1.0 - qL))
    edge_val = -np.pi / (8.0 * qL) * (1.0 - (abs_nu - qL)**2)
    im_chi = np.where(region_edge & ~region_IR, sign_nu * edge_val, im_chi)

    return re_chi + 1j * im_chi


def continuum_boundaries(q: np.ndarray, kF: float = 1.0,
                          me: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Borda inferior/superior do continuum particle-hole 3D."""
    vF = kF / me
    omega_plus = vF * q + q**2 / (2*me)
    omega_minus = np.abs(vF * q - q**2 / (2*me))
    return omega_minus, omega_plus


# =============================================================
# INTERAÇÃO MEDIADA PELO BANHO (única interação)
# =============================================================

def V_bath_mediated(omega: np.ndarray, bath: BathParams,
                    coupling: CouplingParams,
                    bath_mode: str = 'massive') -> np.ndarray:
    """
    Interação efetiva mediada pelo banho — ÚNICA interação no modelo.

        V_eff(omega) = g^2 * D^R_bath(omega)

    Note que NÃO depende de q (banho local, sem dispersão em q).

    bath_mode:
        'massive': D = 1/(omega^2 - E_0^2 - Sigma_bath)
        'massless': D = 1/(omega^2 - Sigma_bath)
    """
    if bath_mode == 'massive':
        D = D_bath_R_massive(omega, bath)
    else:
        D = D_bath_R_local(omega, bath)
    return coupling.g**2 * D


# =============================================================
# RESPOSTA RPA E FUNÇÃO DIELÉTRICA
# =============================================================

def dielectric_RPA(omega: np.ndarray, q: np.ndarray,
                   bath: BathParams, coupling: CouplingParams,
                   bath_mode: str = 'massive') -> np.ndarray:
    """
    Função dielétrica RPA com interação mediada pelo banho:

        epsilon(omega, q) = 1 - V_eff(omega) * chi_0(omega, q)
    """
    omega = np.asarray(omega, dtype=float)
    q = np.asarray(q, dtype=float)
    V = V_bath_mediated(omega, bath, coupling, bath_mode)
    chi0 = lindhard_3d(omega, q, kF=coupling.kF, me=coupling.me) * coupling.N0
    return 1.0 - V * chi0


def response_RPA(omega: np.ndarray, q: np.ndarray,
                 bath: BathParams, coupling: CouplingParams,
                 bath_mode: str = 'massive') -> np.ndarray:
    """
    Resposta de densidade RPA:

        chi(omega, q) = chi_0 / (1 - V_eff * chi_0)
    """
    omega = np.asarray(omega, dtype=float)
    q = np.asarray(q, dtype=float)
    chi0 = lindhard_3d(omega, q, kF=coupling.kF, me=coupling.me) * coupling.N0
    V = V_bath_mediated(omega, bath, coupling, bath_mode)
    eps_rpa = 1.0 - V * chi0
    return chi0 / eps_rpa


def spectral_density(omega: np.ndarray, q: np.ndarray,
                     bath: BathParams, coupling: CouplingParams,
                     bath_mode: str = 'massive') -> np.ndarray:
    """
    A(omega, q) = -Im chi(omega, q) / pi (positiva).
    """
    chi = response_RPA(omega, q, bath, coupling, bath_mode)
    return -np.imag(chi) / np.pi


def loss_function_EELS(omega: np.ndarray, q: np.ndarray,
                       bath: BathParams, coupling: CouplingParams,
                       bath_mode: str = 'massive') -> np.ndarray:
    """
    -Im[1/epsilon] (EELS). Pode mudar de sinal se V_eff for atrativa.
    """
    eps = dielectric_RPA(omega, q, bath, coupling, bath_mode)
    return -np.imag(1.0 / eps)


# =============================================================
# DISPERSÃO DE PLASMONS
# =============================================================

def plasmon_dispersion(q_grid: np.ndarray, bath: BathParams,
                       coupling: CouplingParams,
                       omega_search_range: Tuple[float, float] = None,
                       bath_mode: str = 'massive',
                       n_omega: int = 800,
                       method: str = 'peak') -> dict:
    """
    Encontra dispersão omega_p(q).

    method:
        'peak'  — busca pico de A(omega, q) = -Im chi/pi acima do continuum.
                  Robusto, captura modos amortecidos.
        'zero'  — busca zero de Re epsilon (plasmon estrito).
                  Mais preciso quando o modo existe, mas pode falhar.

    Retorna omega_p, fwhm (medido), Gamma_p (formal via derivada de eps),
    e eta_p = Gamma_p / (2 omega_p).
    """
    if omega_search_range is None:
        omega_search_range = (0.01, 10.0 * bath.E0)

    omega_p_vals = []
    fwhm_vals = []
    gamma_p_vals = []
    Z_p_vals = []
    found_mask = []

    for q in q_grid:
        om_low, om_high = continuum_boundaries(np.array([q]),
                                                kF=coupling.kF, me=coupling.me)
        om_start = max(om_high[0] * 1.005, omega_search_range[0])
        om_end = omega_search_range[1]

        if om_start >= om_end:
            omega_p_vals.append(np.nan); fwhm_vals.append(np.nan)
            gamma_p_vals.append(np.nan); Z_p_vals.append(np.nan)
            found_mask.append(False); continue

        omega_test = np.linspace(om_start, om_end, n_omega)
        q_arr = q * np.ones_like(omega_test)

        omega_p = np.nan; fwhm = np.nan
        gamma_p = np.nan; Z_p = np.nan
        found = False

        if method == 'peak':
            # buscar pico de A_col
            A = spectral_density(omega_test, q_arr, bath, coupling, bath_mode)
            if A.max() < 1e-5:
                omega_p_vals.append(np.nan); fwhm_vals.append(np.nan)
                gamma_p_vals.append(np.nan); Z_p_vals.append(np.nan)
                found_mask.append(False); continue
            peak_idx = np.argmax(A)
            omega_p = omega_test[peak_idx]
            amp = A[peak_idx]
            # FWHM
            half = amp / 2.0
            left = np.where(A[:peak_idx] < half)[0]
            right = np.where(A[peak_idx:] < half)[0]
            if len(left) > 0 and len(right) > 0:
                fwhm = omega_test[peak_idx + right[0]] - omega_test[left[-1]]
            else:
                fwhm = np.nan
            found = True

        elif method == 'zero':
            eps = dielectric_RPA(omega_test, q_arr, bath, coupling, bath_mode)
            re_eps = np.real(eps)
            sign_changes = np.where(np.diff(np.sign(re_eps)))[0]
            if len(sign_changes) == 0:
                omega_p_vals.append(np.nan); fwhm_vals.append(np.nan)
                gamma_p_vals.append(np.nan); Z_p_vals.append(np.nan)
                found_mask.append(False); continue
            idx = sign_changes[0]
            try:
                def f_re_eps(om):
                    return float(np.real(
                        dielectric_RPA(np.array([om]), np.array([q]),
                                        bath, coupling, bath_mode)[0]
                    ))
                omega_p = brentq(f_re_eps, omega_test[idx], omega_test[idx+1],
                                  xtol=1e-6)
                found = True
            except (ValueError, RuntimeError):
                omega_p_vals.append(np.nan); fwhm_vals.append(np.nan)
                gamma_p_vals.append(np.nan); Z_p_vals.append(np.nan)
                found_mask.append(False); continue

        # Computar Gamma_p formal (válido para ambos métodos)
        if found:
            om_p_arr = np.array([omega_p])
            q_arr_p = np.array([q])
            eps_at_p = dielectric_RPA(om_p_arr, q_arr_p, bath, coupling, bath_mode)[0]
            dom = 1e-4
            eps_plus = dielectric_RPA(om_p_arr + dom, q_arr_p, bath, coupling, bath_mode)[0]
            eps_minus = dielectric_RPA(om_p_arr - dom, q_arr_p, bath, coupling, bath_mode)[0]
            deps_domega = (eps_plus - eps_minus) / (2 * dom)
            d_re_eps = np.real(deps_domega)
            if abs(d_re_eps) > 1e-10:
                gamma_p = 2.0 * np.imag(eps_at_p) / d_re_eps
                Z_p = 1.0 / d_re_eps

        omega_p_vals.append(omega_p)
        fwhm_vals.append(fwhm)
        gamma_p_vals.append(abs(gamma_p) if not np.isnan(gamma_p) else np.nan)
        Z_p_vals.append(abs(Z_p) if not np.isnan(Z_p) else np.nan)
        found_mask.append(found)

    return {
        'q_grid': q_grid,
        'omega_p': np.array(omega_p_vals),
        'fwhm': np.array(fwhm_vals),
        'gamma_p': np.array(gamma_p_vals),
        'Z_p': np.array(Z_p_vals),
        'found': np.array(found_mask),
    }


# =============================================================
# TEMPOS DE VIDA E DIAGNÓSTICOS DE FASE
# =============================================================

def plasmon_lifetime_ratio(q_grid: np.ndarray, bath: BathParams,
                            coupling: CouplingParams,
                            **kwargs) -> dict:
    """
    Razão tempo de vida / período: eta_p = Gamma_p / (2 omega_p)
        eta_p < 0.1: plasmon coerente
        eta_p > 1:   plasmon mal-definido (incoerente)
    """
    disp = plasmon_dispersion(q_grid, bath, coupling, **kwargs)
    eta_p = disp['gamma_p'] / (2.0 * disp['omega_p'])
    disp['eta_p'] = eta_p
    return disp


def classify_plasmon_regime(eta_p: np.ndarray) -> np.ndarray:
    """
    0 = coerente (eta_p < 0.1)
    1 = subamortecido (0.1 <= eta_p < 0.5)
    2 = superamortecido (eta_p >= 0.5)
    """
    regime = np.full_like(eta_p, 3, dtype=int)  # default: undefined
    regime[~np.isnan(eta_p) & (eta_p < 0.1)] = 0
    regime[~np.isnan(eta_p) & (eta_p >= 0.1) & (eta_p < 0.5)] = 1
    regime[~np.isnan(eta_p) & (eta_p >= 0.5)] = 2
    return regime


if __name__ == "__main__":
    bath = BathParams(m0=1.0, k0=2.0, alpha=0.3)
    coupling = CouplingParams(g=1.5)

    q_test = np.linspace(0.1, 1.5, 8)
    disp = plasmon_lifetime_ratio(q_test, bath, coupling)
    print("q  |  omega_p  |  Gamma_p  |  eta_p")
    print("-" * 50)
    for i, q in enumerate(q_test):
        print(f"{q:.2f}  |  {disp['omega_p'][i]:.3f}  |  "
              f"{disp['gamma_p'][i]:.3f}  |  {disp['eta_p'][i]:.3f}")
