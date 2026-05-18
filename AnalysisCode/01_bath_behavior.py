"""
analysis.01_bath_behavior
==========================

Análise do comportamento do banho de quasipartículas isoladamente.

Gera figuras:
    1. Im Sigma_bath, Re Sigma_bath em função de omega/E_0 para vários alpha
    2. Comparação de E_0 para diferentes (m_0, k_0) — cutoff variável
    3. Verificação de paridade e causalidade
    4. Comportamento assintótico IR e UV
"""

import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, '/home/lisan/Documentos/QuantumDissipation/qp_bath_project /qp_bath_project/')

from core.bath import BathParams, im_sigma_bath, re_sigma_bath, sigma_bath

FIGDIR = 'home/lisan/Documentos/QuantumDissipation/qp_bath_project /qp_bath_project/figures'
os.makedirs(FIGDIR, exist_ok=True)

plt.rcParams.update({
    'font.size': 11, 'axes.labelsize': 12, 'legend.fontsize': 9,
    'lines.linewidth': 1.8, 'axes.grid': True, 'grid.alpha': 0.35,
})


# =============================================================
# FIG 1: Self-energia em função de omega para vários alpha
# =============================================================

print("[FIG 1] Self-energia do banho para vários alpha (m0=k0=1)")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
alphas = [0.05, 0.1, 0.3, 0.5, 1.0]
omega = np.linspace(-4, 4, 801)
colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(alphas)))

for alpha, c in zip(alphas, colors):
    p = BathParams(m0=1.0, k0=1.0, alpha=alpha)
    axes[0].plot(omega/p.E0, im_sigma_bath(omega, p)/p.E0,
                 color=c, label=rf'$\alpha={alpha}$')
    axes[1].plot(omega/p.E0, re_sigma_bath(omega, p)/p.E0,
                 color=c, label=rf'$\alpha={alpha}$')

axes[0].set_xlabel(r'$\omega/E_0$'); axes[0].set_ylabel(r'$\mathrm{Im}\,\Sigma_{\rm bath}/E_0$')
axes[0].set_title(r'Parte imaginária (par em $\omega$)')
axes[0].legend(loc='upper right'); axes[0].axhline(0, color='k', lw=0.5)

axes[1].set_xlabel(r'$\omega/E_0$'); axes[1].set_ylabel(r'$\mathrm{Re}\,\Sigma_{\rm bath}/E_0$')
axes[1].set_title(r'Parte real (ímpar em $\omega$)')
axes[1].legend(loc='upper left'); axes[1].axhline(0, color='k', lw=0.5)
axes[1].axvline(0, color='k', lw=0.5)

fig.suptitle('Self-energia do banho — Convenção A (m₀=mₑ, k₀=k_F)', fontsize=13)
plt.tight_layout()
plt.show()
plt.savefig(f'{FIGDIR}/fig01_bath_self_energy.png', dpi=140, bbox_inches='tight')
plt.close()
print(f"    salvo: fig01_bath_self_energy.png")


# =============================================================
# FIG 2: Variação de E_0 com (m_0, k_0)
# =============================================================

print("[FIG 2] Cutoff E_0 em função de (m0, k0)")

m0_vals = np.logspace(0, 3, 5)  # 1 a 1000
k0_vals = np.linspace(0.3, 3.0, 5)  # 0.3 kF a 3 kF

fig, ax = plt.subplots(figsize=(7.5, 5.5))
omega = np.linspace(0, 6, 400)
alpha_fixed = 0.3

# fixar k0, variar m0
for m0, c in zip(m0_vals, plt.cm.plasma(np.linspace(0.1, 0.85, len(m0_vals)))):
    p = BathParams(m0=m0, k0=1.0, alpha=alpha_fixed)
    label = rf'$m_0/m_e = {m0:.0f}$ ($E_0={p.E0:.2g}$)'
    ax.plot(omega, -im_sigma_bath(omega, p), color=c, label=label)

ax.set_xlabel(r'$\omega$ (em unidades de $E_F$)')
ax.set_ylabel(r'$|\mathrm{Im}\,\Sigma_{\rm bath}|$ (em unidades de $E_F$)')
ax.set_yscale('log')
ax.set_title(rf'Variação de $E_0$ com $m_0$ (fixo $k_0=k_F$, $\alpha={alpha_fixed}$)')
ax.legend(loc='best', fontsize=8)
plt.tight_layout()
plt.savefig(f'{FIGDIR}/fig02_E0_variation_m0.png', dpi=140, bbox_inches='tight')
plt.show()
plt.close()
print(f"    salvo: fig02_E0_variation_m0.png")


# =============================================================
# FIG 3: Comportamento assintótico IR e UV
# =============================================================

print("[FIG 3] Comportamento assintótico — verificação analítica")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
p = BathParams(m0=1.0, k0=1.0, alpha=0.3)
gamma_E = 0.5772156649

# IR: omega -> 0
omega_IR = np.logspace(-5, 0, 200)
im_S_IR = -im_sigma_bath(omega_IR, p)  # |Im Sigma|
re_S_IR = re_sigma_bath(omega_IR, p)

# previsões analíticas
# IR: |Im Sigma| ~ pi*alpha*|omega| (leading)
im_predicted = np.pi * p.alpha * omega_IR
# IR: Re Sigma ~ -2 alpha omega^2 / E_0 * [ln(omega/E_0) + gamma_E - 1]
re_predicted = -2 * p.alpha * omega_IR**2 / p.E0 * (np.log(omega_IR/p.E0) + gamma_E - 1)

axes[0].loglog(omega_IR, im_S_IR, 'b-', label=r'$|\mathrm{Im}\,\Sigma|$ numérico')
axes[0].loglog(omega_IR, im_predicted, 'k--', label=r'$\pi\alpha|\omega|$ (IR)')
axes[0].set_xlabel(r'$\omega$')
axes[0].set_ylabel(r'$|\mathrm{Im}\,\Sigma_{\rm bath}|$')
axes[0].set_title('Comportamento IR: $|\\mathrm{Im}\\,\\Sigma| \\sim \\pi\\alpha|\\omega|$')
axes[0].legend()

# Re Sigma — escala linear, ressaltar oscilação ω² ln ω
axes[1].plot(omega_IR, re_S_IR, 'b-', label=r'$\mathrm{Re}\,\Sigma$ numérico')
axes[1].plot(omega_IR, re_predicted, 'k--',
             label=r'$-(2\alpha\omega^2/E_0)[\ln(\omega/E_0)+\gamma_E-1]$')
axes[1].set_xlabel(r'$\omega$')
axes[1].set_ylabel(r'$\mathrm{Re}\,\Sigma_{\rm bath}$')
axes[1].set_title(r'Comportamento IR: $\mathrm{Re}\,\Sigma \sim \omega^2\ln\omega$')
axes[1].legend()
axes[1].set_xlim(0, 0.5)

plt.tight_layout()
plt.savefig(f'{FIGDIR}/fig03_asymptotic_IR.png', dpi=140, bbox_inches='tight')
plt.show()
plt.close()
print(f"    salvo: fig03_asymptotic_IR.png")


# =============================================================
# FIG 4: Propagador do banho |D^R(omega)|^2 (densidade espectral)
# =============================================================

print("[FIG 4] Propagador retardado do banho")
from core.bath import bath_propagator

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
p = BathParams(m0=1.0, k0=1.0, alpha=alpha)
omega = np.linspace(-2.0*p.E0, -1.85*p.E0, 600)
k = np.linspace(0.01, 4, 600)

alphas = [0.05, 0.2, 0.5, 1.0]
for alpha, c in zip(alphas, plt.cm.viridis(np.linspace(0.15, 0.85, len(alphas)))):
    p = BathParams(m0=1.0, k0=1.0, alpha=alpha)
    D = bath_propagator(omega, k,p, mass_term=True)
    A_bath = -np.imag(D) / np.pi
    axes[0].plot(omega/p.E0, A_bath, color=c, label=rf'$\alpha={alpha}$')
    axes[1].plot(omega/p.E0, np.real(D), color=c, label=rf'$\alpha={alpha}$')

# linha vertical no polo nominal omega^2 = E_0
#axes[0].axvline(1.0, color='k', ls=':', alpha=0.5, label=r'$\omega = \sqrt{E_0}$')
axes[0].set_xlabel(r'$\omega/E_0$')
axes[0].set_ylabel(r'$A_{\rm bath}(\omega) = -\mathrm{Im}\,D^R/\pi$')
axes[0].set_title('Densidade espectral do banho (com massa)')
axes[0].legend()

axes[1].set_xlabel(r'$\omega/E_0$')
axes[1].set_ylabel(r'$\mathrm{Re}\,D^R(\omega)$')
axes[1].set_title('Parte real do propagador')
axes[1].legend()
axes[1].axhline(0, color='k', lw=0.5)

plt.tight_layout()
plt.show()
plt.savefig(f'{FIGDIR}/fig04_bath_propagator.png', dpi=140, bbox_inches='tight')
plt.close()
print(f"    salvo: fig04_bath_propagator.png")


print("\nAnálise do banho concluída.")
