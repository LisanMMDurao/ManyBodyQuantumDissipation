"""
core.bath
=========

Self-energia do banho ôhmico de quasipartículas.

Modelo: banho de osciladores com dispersão quadrática de cutoff
    E_0 = hbar^2 k_0^2 / (2 m_0)
e função espectral ôhmica J(omega) = alpha omega exp(-|omega|/E_0).

Constantes do modelo: (omega, m_0, k, k_0, alpha, E_0).

MODELIZAÇÃO DA AUTO-ENERGIA:
    Im Sigma_bath^R(omega) = -pi * alpha * |omega| * exp(-|omega|/E_0)
    (par em omega, fisicamente positiva-definida em magnitude)

    Re Sigma_bath^R(omega) = -alpha * omega * [e^(-x) Ei(x) + e^x E_1(x)]
    com x = |omega|/E_0 (ímpar em omega, via Kramers-Kronig).

Propriedades:
    * Im Sigma é par em omega; Re Sigma é ímpar em omega.
    * No IR (omega -> 0): Im Sigma ~ -pi*alpha*|omega|, Re Sigma ~ omega^2 ln(omega).
    * No UV (omega >> E_0): ambas decaem exponencialmente.
    * Causalidade retardada: Im Sigma^R(omega > 0) < 0. ✓
"""

from __future__ import annotations

import numpy as np
from scipy.special import expi, exp1
from typing import Union

ArrayLike = Union[float, np.ndarray]


# =============================================================
# CONSTANTES DO MODELO
# =============================================================

class BathParams:
    """
    Parâmetros do banho de quasipartículas.

    Attributes
    ----------
    m0 : float
        Massa do banho (em unidades da massa eletrônica m_e).
    k0 : float
        Escala de tamanho/momentum do banho (em unidades de k_F).
    alpha : float
        Acoplamento adimensional sistema-banho.
    E0 : float
        Cutoff energético do banho, computado como hbar^2 k0^2 / (2 m0)
        em unidades de E_F = hbar^2 k_F^2 / (2 m_e).

    Em unidades naturais (hbar = m_e = k_F = 1), temos E_F = 1/2 e
        E_0 = k_0^2 / (2 m_0).
    """

    def __init__(self, m0: float = 1.0, k0: float = 1.0,
                 alpha: float = 0.1, hbar: float = 1.0, kF: float = 1.0):
        self.m0 = m0
        self.k0 = k0
        self.alpha = alpha
        self.hbar = hbar
        self.kF = kF
        # cutoff: E_0 = hbar^2 k_0^2 / (2 m_0)
        self.E0 = hbar**2 * k0**2 / (2.0 * m0)
        # Energia de Fermi de referência (m_e = 1)
        self.EF = hbar**2 * kF**2 / 2.0

    @property
    def E0_over_EF(self) -> float:
        """Razão E_0 / E_F — controla 'rigidez' do banho."""
        return self.E0 / self.EF

    @property
    def mass_ratio(self) -> float:
        """m_0 / m_e."""
        return self.m0

    @property
    def momentum_ratio(self) -> float:
        """k_0 / k_F."""
        return self.k0 / self.kF

    def __repr__(self) -> str:
        return (f"BathParams(m0={self.m0:.3g}, k0={self.k0:.3g}, "
                f"alpha={self.alpha:.3g}, E0={self.E0:.4g}, "
                f"E0/EF={self.E0_over_EF:.3g})")


# =============================================================
# AUTOENERGIA DO BANHO 
# =============================================================

def im_sigma_bath(omega: ArrayLike, params: BathParams) -> ArrayLike:
    """
    Parte imaginária da self-energia retardada do banho.

        Im Sigma_bath^R(omega) = -pi * alpha * |omega| * exp(-|omega|/E_0)

    Forma par em omega; magnitude representa taxa de dissipação ôhmica
    regularizada por cutoff exponencial em E_0.
    """
    omega = np.asarray(omega, dtype=float)
    return -np.pi * params.alpha * np.abs(omega) #* np.exp(-np.abs(omega) / params.E0)


def re_sigma_bath(omega: ArrayLike, params: BathParams) -> ArrayLike:
    """
    Parte real da self-energia retardada do banho (via Kramers-Kronig).

        Re Sigma_bath^R(omega) = -alpha * omega * [e^(-x) Ei(x) + e^x E_1(x)]

    com x = |omega|/E_0. Ímpar em omega.

    No limite IR (x -> 0): Re Sigma ~ -2 alpha omega * x * [ln(x) + gamma_E - 1]/E_0
                                   = -2 alpha omega^2/E_0 * [ln(omega/E_0) + gamma_E - 1]
    No limite UV (x -> infty): Re Sigma -> -alpha omega * (2/x) = -2 alpha E_0/omega -> 0.
    """
    omega = np.asarray(omega, dtype=float)
    x = np.abs(omega) / params.E0

    # tratamento cuidadoso de x = 0
    result = np.zeros_like(omega)
    small = x < 1e-8
    large = ~small

    if np.any(large):
        xl = x[large]
        # Para xl muito grande (xl > 700), exp(xl) overflows.
        # Usar limite assintótico: e^x E_1(x) -> 1/x para x grande
        # e^(-x) Ei(x) -> 1/x  para x grande
        # então bracket -> 2/x = 2 E_0/|omega|
        # Re Sigma -> -alpha omega * 2 E_0 / |omega| = -2 alpha sign(omega) E_0
        # Mas isto é constante! Para x intermediário, calcular exatamente.
        very_large = xl > 500
        normal = ~very_large
        bracket = np.zeros_like(xl)
        if np.any(normal):
            xn = xl[normal]
            bracket[normal] = np.exp(-xn) * expi(xn) + np.exp(xn) * exp1(xn)
        if np.any(very_large):
            # série assintótica: e^(-x)Ei(x) ~ 1/x + 1/x^2 + 2/x^3 + ...
            #                   e^x E_1(x)  ~ 1/x - 1/x^2 + 2/x^3 - ...
            # soma: 2/x + 4/x^3 + ...
            xv = xl[very_large]
            bracket[very_large] = 2.0/xv + 4.0/xv**3
        result[large] = -params.alpha * omega[large] * bracket

    if np.any(small):
        # No limite x -> 0: bracket ~ -2 x [ln x + gamma - 1]
        # então Re Sigma ~ -alpha * omega * (-2 x [ln x + gamma - 1])
        #                = 2 alpha omega * x * [ln x + gamma - 1]
        # que vai a zero como x ln x — finito mas com derivada divergente.
        # Para x estritamente 0, Re Sigma = 0 (limite).
        ws = omega[small]
        xs = x[small]
        gamma_E = 0.5772156649015329
        # cuidado: para xs = 0, ln(xs) é -infty mas xs*ln(xs) -> 0
        with np.errstate(divide='ignore', invalid='ignore'):
            log_term = np.where(xs > 0, np.log(xs), 0.0)
            result[small] = 2.0 * params.alpha * ws * xs * (log_term + gamma_E - 1.0)
        # forçar zero estrito em omega = 0
        result[small & (omega == 0)] = 0.0

    return result


def dRe_sigma_bath_domega(omega: ArrayLike, params: BathParams) -> ArrayLike:
    """
    Derivada analítica d Re Sigma / d omega.

    Útil para computar Z = 1 / (1 - d Re Sigma / d omega) sem ruído numérico.

    Derivando Re Sigma = -alpha * omega * F(|omega|/E_0), onde
        F(x) = e^(-x) Ei(x) + e^x E_1(x),
    temos
        dF/dx = -e^(-x) Ei(x) + e^(-x)/x + e^x E_1(x) - e^x/x = -F(x) extra...

    Mais simples: usar identidades
        d/dx [e^(-x) Ei(x)] = -e^(-x) Ei(x) + 1/x
        d/dx [e^x E_1(x)] = e^x E_1(x) - 1/x
    Logo dF/dx = -e^(-x) Ei(x) + e^x E_1(x) = G(x).

    Então
        d Re Sigma / d omega = -alpha * F(|omega|/E_0)
                              - alpha * omega * sign(omega)/E_0 * G(|omega|/E_0)
                            = -alpha * [F(x) + x * G(x)]
    onde a paridade par é preservada.
    """
    omega = np.asarray(omega, dtype=float)
    x = np.abs(omega) / params.E0

    result = np.zeros_like(omega)
    small = x < 1e-8
    large = ~small

    if np.any(large):
        xl = x[large]
        very_large = xl > 500
        normal = ~very_large
        F = np.zeros_like(xl)
        G = np.zeros_like(xl)
        if np.any(normal):
            xn = xl[normal]
            F[normal] = np.exp(-xn) * expi(xn) + np.exp(xn) * exp1(xn)
            G[normal] = -np.exp(-xn) * expi(xn) + np.exp(xn) * exp1(xn)
        if np.any(very_large):
            xv = xl[very_large]
            # F ~ 2/x + 4/x^3, G ~ -2/x^2 (diferença das séries)
            F[very_large] = 2.0/xv + 4.0/xv**3
            G[very_large] = -2.0/xv**2
        result[large] = -params.alpha * (F + xl * G)

    if np.any(small):
        # F(x) ~ -2x[ln x + gamma - 1], G(x) ~ -2[ln x + gamma] + O(x)
        # F + x G ~ -2x[ln x + gamma - 1] - 2x[ln x + gamma] = -2x[2 ln x + 2 gamma - 1]
        xs = x[small]
        gamma_E = 0.5772156649015329
        with np.errstate(divide='ignore', invalid='ignore'):
            log_term = np.where(xs > 0, np.log(xs), 0.0)
            bracket = -2.0 * xs * (2.0 * log_term + 2.0 * gamma_E - 1.0)
            result[small] = -params.alpha * bracket
        result[small & (omega == 0)] = 0.0

    return result


def sigma_bath(omega: ArrayLike, params: BathParams) -> np.ndarray:
    """
    Self-energia retardada complexa: Sigma_bath^R(omega) = Re + i Im.
    """
    return re_sigma_bath(omega, params) + 1j * im_sigma_bath(omega, params)


# =============================================================
# PROPAGADOR DO BANHO (mediador massivo dressado)
# =============================================================

def bath_propagator(omega: ArrayLike, k: ArrayLike, params: BathParams,
                    mass_term: bool = True) -> np.ndarray:
    """
    Propagador retardado do banho:
        D^R_bath(omega) = 1 / (omega^2 - omega_0^2 - Sigma_bath^R(omega))

    onde omega_0^2 = E_0 é a "massa" do mediador (em unidades de E_F),
    obtida de hbar^2 k_0^2 / (2 m_0) = E_0.

    Se mass_term=False, retorna propagador "puro ôhmico" sem massa
    (banho de osciladores na origem):
        D^R(omega) = 1 / (omega^2 - Sigma_bath^R(omega))
    """
    omega = np.asarray(omega, dtype=float)
    sig = sigma_bath(omega, params)
    omega0_sq = 0.5*(1/params.m0)*(params.hbar*k)**2 if mass_term else 0.0
    denom = omega**2 - omega0_sq - sig
    return 1.0 / denom


# =============================================================
# DIAGNÓSTICOS BÁSICOS
# =============================================================

def check_causality(params: BathParams, omega_test: np.ndarray = None) -> dict:
    """
    Verifica causalidade retardada: Im Sigma^R(omega > 0) < 0.
    """
    if omega_test is None:
        omega_test = np.linspace(-2*params.E0, 2*params.E0, 201)
    im_s = im_sigma_bath(omega_test, params)
    re_s = re_sigma_bath(omega_test, params)
    pos_omega = omega_test > 0
    neg_omega = omega_test < 0
    # Para Im Sigma par (convenção A), causalidade retardada exige
    # Im Sigma^R(omega) <= 0 PARA TODO omega (não muda de sinal).
    return {
        'causal_all_omega': np.all(im_s <= 1e-12),
        'im_parity_even': np.allclose(im_s, im_s[::-1], atol=1e-10),
        're_parity_odd': np.allclose(re_s, -re_s[::-1], atol=1e-10),
        'sigma_at_zero': (float(re_sigma_bath(np.array([0.0]), params)[0]),
                          float(im_sigma_bath(np.array([0.0]), params)[0])),
    }


if __name__ == "__main__":
    # Teste rápido
    p = BathParams(m0=1.0, k0=1.0, alpha=0.1)
    print(p)
    print("E_0/E_F =", p.E0_over_EF)
    omega = np.linspace(-3, 3, 11)
    k = np.linspace(-2.0*p.k0,2.0*p.k0,100)
    print("omega:", omega)
    print("Im Sigma:", im_sigma_bath(omega, p))
    print("Re Sigma:", re_sigma_bath(omega, p))
    print("Causality check:", check_causality(p))
