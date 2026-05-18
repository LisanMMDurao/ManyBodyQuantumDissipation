"""
analysis.04_summary
===================

Figura-resumo do projeto: síntese dos resultados principais.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, ListedColormap
import matplotlib.patches as mpatches
import sys, os
sys.path.insert(0, '/home/claude/qp_bath_project')

from core.bath import BathParams, im_sigma_bath, re_sigma_bath
from core.electron_gas import CouplingParams, spectral_function, momentum_cutoff_factor
from core.phase_analysis import scan_alpha_g, dimensionless_coupling

FIGDIR = '/home/claude/qp_bath_project/figures'

plt.rcParams.update({
    'font.size': 11, 'axes.labelsize': 12, 'legend.fontsize': 9,
    'lines.linewidth': 1.8, 'axes.grid': True, 'grid.alpha': 0.35,
})

phase_cmap = ListedColormap(['#3b73af', '#e8a13a', '#c34440'])


fig = plt.figure(figsize=(15, 11))
gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.35)

# =============================================================
# (A) Self-energia do banho
# =============================================================
ax_A = fig.add_subplot(gs[0, 0])
omega = np.linspace(-3, 3, 400)
for alpha, c in zip([0.1, 0.3, 0.5, 1.0],
                     plt.cm.viridis(np.linspace(0.2, 0.85, 4))):
    p = BathParams(m0=1.0, k0=1.0, alpha=alpha)
    ax_A.plot(omega/p.E0, im_sigma_bath(omega, p)/p.E0,
              color=c, label=rf'$\alpha={alpha}$')
ax_A.set_xlabel(r'$\omega/E_0$')
ax_A.set_ylabel(r'$\mathrm{Im}\,\Sigma_{\rm bath}/E_0$')
ax_A.set_title(r'(A) Banho: $\mathrm{Im}\,\Sigma \propto -\alpha|\omega|e^{-|\omega|/E_0}$')
ax_A.legend(fontsize=8)
ax_A.axhline(0, color='k', lw=0.5)

# =============================================================
# (B) Re Sigma — paridade ímpar
# =============================================================
ax_B = fig.add_subplot(gs[0, 1])
for alpha, c in zip([0.1, 0.3, 0.5, 1.0],
                     plt.cm.viridis(np.linspace(0.2, 0.85, 4))):
    p = BathParams(m0=1.0, k0=1.0, alpha=alpha)
    ax_B.plot(omega/p.E0, re_sigma_bath(omega, p)/p.E0,
              color=c, label=rf'$\alpha={alpha}$')
ax_B.set_xlabel(r'$\omega/E_0$')
ax_B.set_ylabel(r'$\mathrm{Re}\,\Sigma_{\rm bath}/E_0$')
ax_B.set_title(r'(B) Re $\Sigma$ ímpar (via KK)')
ax_B.legend(fontsize=8)
ax_B.axhline(0, color='k', lw=0.5)
ax_B.axvline(0, color='k', lw=0.5)

# =============================================================
# (C) Fator F(k_0/k_F)
# =============================================================
ax_C = fig.add_subplot(gs[0, 2])
k0_range = np.linspace(0.1, 4, 100)
F = np.where(k0_range >= 2.0, 1.0, (k0_range/2)**2)
ax_C.plot(k0_range, F, 'b-', lw=2.5)
ax_C.axvline(2.0, color='k', ls='--', alpha=0.5, label=r'$k_0=2k_F$')
ax_C.set_xlabel(r'$k_0/k_F$')
ax_C.set_ylabel(r'$F(k_0/k_F)$')
ax_C.set_title(r'(C) Espaço de fase: $F$')
ax_C.legend()

# =============================================================
# (D) Função espectral (3 regimes)
# =============================================================
ax_D = fig.add_subplot(gs[1, 0])
omega_sp = np.linspace(-1.5, 1.5, 400)

cases_sp = [
    (0.1, 1.0, 'FL', 'tab:blue'),
    (0.5, 2.5, 'MFL', 'tab:orange'),
    (1.5, 3.5, 'Incoh.', 'tab:red'),
]
for alpha, g, label, c in cases_sp:
    bath = BathParams(m0=1.0, k0=1.0, alpha=alpha)
    coupling = CouplingParams(g=g)
    lam = dimensionless_coupling(bath, coupling)
    A = spectral_function(omega_sp, 0.0, bath, coupling, broadening=8e-3)
    ax_D.plot(omega_sp, A, color=c, label=rf'{label} ($\lambda={lam:.2f}$)')
ax_D.set_xlabel(r'$\omega$')
ax_D.set_ylabel(r'$A(k_F,\omega)$')
ax_D.set_title('(D) Função espectral nos 3 regimes')
ax_D.legend(fontsize=8)
ax_D.axvline(0, color='k', lw=0.5)

# =============================================================
# (E) Diagrama de fases (alpha, g) — m_0=m_e, k_0=k_F
# =============================================================
ax_E = fig.add_subplot(gs[1, 1])
alpha_grid = np.logspace(-2, 1, 25)
g_grid = np.linspace(0.1, 5.0, 25)
result = scan_alpha_g(alpha_grid, g_grid, m0=1.0, k0_over_kF=1.0)
AG, GG = np.meshgrid(result['alpha_grid'], result['g_grid'], indexing='ij')
im_E = ax_E.pcolormesh(AG, GG, np.clip(result['phase'], -0.5, 2.5),
                        shading='auto', cmap=phase_cmap, vmin=-0.5, vmax=2.5)
cs = ax_E.contour(AG, GG, result['lambda'], levels=[0.1, 1.0],
                   colors='white', linewidths=1.5, alpha=0.85)
ax_E.clabel(cs, inline=True, fontsize=8, fmt={0.1:'0.1', 1.0:'1'})
ax_E.set_xscale('log')
ax_E.set_xlabel(r'$\alpha$')
ax_E.set_ylabel(r'$g$')
ax_E.set_title(r'(E) Fases em $(\alpha, g)$ — $m_0=m_e$, $k_0=k_F$')

# =============================================================
# (F) Diagrama de fases (alpha, g) — m_0=1000, k_0=k_F
# =============================================================
ax_F = fig.add_subplot(gs[1, 2])
result = scan_alpha_g(alpha_grid, g_grid, m0=1000.0, k0_over_kF=1.0)
AG, GG = np.meshgrid(result['alpha_grid'], result['g_grid'], indexing='ij')
im_F = ax_F.pcolormesh(AG, GG, np.clip(result['phase'], -0.5, 2.5),
                        shading='auto', cmap=phase_cmap, vmin=-0.5, vmax=2.5)
cs = ax_F.contour(AG, GG, result['lambda'], levels=[0.1, 1.0],
                   colors='white', linewidths=1.5, alpha=0.85)
ax_F.clabel(cs, inline=True, fontsize=8, fmt={0.1:'0.1', 1.0:'1'})
ax_F.set_xscale('log')
ax_F.set_xlabel(r'$\alpha$')
ax_F.set_ylabel(r'$g$')
ax_F.set_title(r'(F) $m_0=1000\,m_e$ (íon): mesma fase, $E_0\to 5\times 10^{-4}$')

# =============================================================
# (G) Cutoff E_0(m_0, k_0)
# =============================================================
ax_G = fig.add_subplot(gs[2, 0])
m0_range = np.logspace(0, 3.5, 80)
for k0r, c in zip([0.5, 1.0, 2.0, 4.0],
                   plt.cm.plasma(np.linspace(0.15, 0.85, 4))):
    E0_vals = [(k0r)**2 / (2*m0) for m0 in m0_range]
    ax_G.loglog(m0_range, E0_vals, color=c, label=rf'$k_0/k_F={k0r}$', lw=2)
ax_G.axhline(0.5, color='k', ls='--', alpha=0.5, label=r'$E_F$')
ax_G.set_xlabel(r'$m_0/m_e$')
ax_G.set_ylabel(r'$E_0/E_F$')
ax_G.set_title(r'(G) Cutoff $E_0 = \hbar^2 k_0^2/(2m_0)$')
ax_G.legend(fontsize=8)

# =============================================================
# (H) Im Sigma_e em diferentes m_0
# =============================================================
ax_H = fig.add_subplot(gs[2, 1])
from core.electron_gas import im_sigma_e
omega_pos = np.logspace(-5, 0.5, 200)
for m0, c in zip([1, 10, 100, 1000],
                  plt.cm.plasma(np.linspace(0.15, 0.85, 4))):
    bath = BathParams(m0=m0, k0=1.0, alpha=0.5)
    coupling = CouplingParams(g=2.0)
    im_S = -im_sigma_e(omega_pos, bath, coupling)
    ax_H.loglog(omega_pos, im_S, color=c,
                label=rf'$m_0={m0}$ ($E_0={bath.E0:.2g}$)')
ax_H.set_xlabel(r'$\omega$')
ax_H.set_ylabel(r'$|\mathrm{Im}\,\Sigma_e|$')
ax_H.set_title(r'(H) Cutoff adiabático: $E_0(m_0)$')
ax_H.legend(fontsize=7, loc='lower right')

# =============================================================
# (I) Texto explicativo
# =============================================================
ax_I = fig.add_subplot(gs[2, 2])
ax_I.axis('off')

text = (
    r'$\mathbf{Convenção\ A}$' + '\n\n'
    r'$\mathrm{Im}\,\Sigma_{\rm bath}^R = -\pi\alpha|\omega|e^{-|\omega|/E_0}$' + '\n'
    r'$\mathrm{Re}\,\Sigma_{\rm bath}^R = -\alpha\omega[e^{-x}\mathrm{Ei}(x)+e^x E_1(x)]$' + '\n\n'
    r'$E_0 = \hbar^2 k_0^2/(2m_0)$' + '\n\n'
    r'$\mathbf{Acoplamento:}$' + '\n'
    r'$\lambda = \pi\alpha g^2 N(0) F(k_0/k_F)$' + '\n\n'
    r'$\mathbf{Fases:}$' + '\n'
    r'• FL:    $\lambda < 0.1$' + '\n'
    r'• MFL:   $0.1 \leq \lambda < 1$' + '\n'
    r'• Incoh: $\lambda \geq 1$' + '\n\n'
    r'$\mathbf{Obs:}$ $m_0$ não entra em $\lambda$,' + '\n'
    r'só em $E_0$. Mesma fenomenologia' + '\n'
    r'em escala adiabática.'
)
ax_I.text(0.05, 0.95, text, transform=ax_I.transAxes,
          verticalalignment='top', fontsize=10,
          bbox=dict(boxstyle='round', facecolor='#f5f5f5',
                    edgecolor='gray', alpha=0.9))

fig.suptitle('Banho de Quasipartículas — Resumo do projeto (Convenção A)',
             fontsize=14, fontweight='bold', y=0.995)

# legenda comum
patches = [mpatches.Patch(color='#3b73af', label='FL coerente'),
           mpatches.Patch(color='#e8a13a', label='Marginal FL'),
           mpatches.Patch(color='#c34440', label='Incoerente')]
fig.legend(handles=patches, loc='lower center', ncol=3, fontsize=10,
           bbox_to_anchor=(0.5, -0.005))

plt.savefig(f'{FIGDIR}/fig00_summary.png', dpi=140, bbox_inches='tight')
plt.close()
print(f"salvo: fig00_summary.png")
