"""
analysis.05_collective_modes
============================

Análise dos modos coletivos do gás de elétrons acoplado ao banho.

Inclui:
    - Mapa A_col(omega, q) — função espectral de densidade
    - Comparação A_col vs. loss function -Im[1/eps]
    - Dispersão omega_mode(q)
    - Variação dos modos com (alpha, g, k_0, m_0)
    - Identificação de plasmon vs modo bosônico dressado
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm
import sys, os
sys.path.insert(0, '/home/claude/qp_bath_project')

from core.bath import BathParams
from core.electron_gas import CouplingParams
from core.collective_modes import (
    lindhard_3d, V_effective, dielectric_function,
    response_function, collective_spectral_function, loss_function,
    find_collective_mode, collective_dispersion, continuum_boundaries
)

FIGDIR = '/home/claude/qp_bath_project/figures'
os.makedirs(FIGDIR, exist_ok=True)

plt.rcParams.update({
    'font.size': 11, 'axes.labelsize': 12, 'legend.fontsize': 9,
    'lines.linewidth': 1.8, 'axes.grid': True, 'grid.alpha': 0.35,
})


# =============================================================
# FIG 14: Mapa A_col(omega, q) — caso de referência
# =============================================================

print("[FIG 14] Mapa A_col — caso de referência")

bath = BathParams(m0=1.0, k0=2.0, alpha=0.3)
coupling = CouplingParams(g=2.5)

omega = np.linspace(0.01, 4.0, 400)
q_grid = np.linspace(0.05, 2.5, 150)
OMG, QG = np.meshgrid(omega, q_grid, indexing='ij')
A = collective_spectral_function(OMG, QG, bath, coupling)

fig, ax = plt.subplots(figsize=(9, 7))
A_safe = np.maximum(A, 1e-5)
im = ax.pcolormesh(QG, OMG, A_safe,
                    norm=LogNorm(vmin=1e-3, vmax=A_safe.max()),
                    shading='auto', cmap='inferno')
plt.colorbar(im, ax=ax, label=r'$A_{\rm col}(\omega,q)$')

# Continuum particle-hole
q_overlay = np.linspace(0.05, 2.5, 200)
om_low, om_high = continuum_boundaries(q_overlay, kF=coupling.kF, me=coupling.me)
ax.plot(q_overlay, om_high, 'w--', lw=1.5, alpha=0.7, label='borda sup. continuum')
ax.plot(q_overlay, om_low, 'w:', lw=1.5, alpha=0.7, label='borda inf. continuum')

# Dispersão do modo (busca por pico)
q_disp = np.linspace(0.1, 2.0, 40)
disp = collective_dispersion(q_disp, bath, coupling, above_continuum_only=True)
valid = ~np.isnan(disp['omega_mode'])
ax.plot(q_disp[valid], disp['omega_mode'][valid], 'cyan', lw=2.5,
        label='modo coletivo')

# Linha k0 (suporte do banho)
ax.axvline(bath.k0, color='magenta', ls=':', lw=2, alpha=0.7,
           label=rf'$k_0={bath.k0}\,k_F$')

ax.set_xlabel(r'$q/k_F$'); ax.set_ylabel(r'$\omega$ (em $E_F$)')
ax.set_title(rf'$A_{{\rm col}}$: $\alpha={bath.alpha}$, $g={coupling.g}$, '
             rf'$k_0={bath.k0}\,k_F$, $m_0=m_e$')
ax.legend(loc='upper left', framealpha=0.9, fontsize=9)
plt.tight_layout()
plt.savefig(f'{FIGDIR}/fig14_A_col_main.png', dpi=140, bbox_inches='tight')
plt.close()
print(f"    salvo: fig14_A_col_main.png")


# =============================================================
# FIG 15: A_col vs loss function — caracterização dual
# =============================================================

print("[FIG 15] A_col vs -Im[1/eps]")

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# (a) A_col
A_plot = np.maximum(A, 1e-5)
im0 = axes[0].pcolormesh(QG, OMG, A_plot,
                          norm=LogNorm(vmin=1e-3, vmax=A_plot.max()),
                          shading='auto', cmap='inferno')
plt.colorbar(im0, ax=axes[0], label=r'$A_{\rm col} = -\mathrm{Im}\,\chi/\pi$')
axes[0].plot(q_overlay, om_high, 'w--', lw=1, alpha=0.7)
axes[0].plot(q_overlay, om_low, 'w:', lw=1, alpha=0.7)
axes[0].set_xlabel(r'$q/k_F$'); axes[0].set_ylabel(r'$\omega$')
axes[0].set_title(r'(a) Função resposta de densidade')

# (b) Loss function -Im[1/eps]
loss = loss_function(OMG, QG, bath, coupling)
vmax_loss = np.percentile(np.abs(loss), 98)
im1 = axes[1].pcolormesh(QG, OMG, loss,
                          norm=SymLogNorm(linthresh=1e-3, vmin=-vmax_loss, vmax=vmax_loss),
                          shading='auto', cmap='RdBu_r')
plt.colorbar(im1, ax=axes[1], label=r'$-\mathrm{Im}[1/\varepsilon]$')
axes[1].plot(q_overlay, om_high, 'k--', lw=1, alpha=0.5)
axes[1].plot(q_overlay, om_low, 'k:', lw=1, alpha=0.5)
axes[1].set_xlabel(r'$q/k_F$'); axes[1].set_ylabel(r'$\omega$')
axes[1].set_title(r'(b) Loss function (EELS): azul = atratividade')

fig.suptitle('Comparação A_col vs. loss function', fontsize=13)
plt.tight_layout()
plt.savefig(f'{FIGDIR}/fig15_A_col_vs_loss.png', dpi=140, bbox_inches='tight')
plt.close()
print(f"    salvo: fig15_A_col_vs_loss.png")


# =============================================================
# FIG 16: Quatro regimes em (alpha, g)
# =============================================================

print("[FIG 16] Modos coletivos em quatro regimes")

regimes = [
    ('Acopl. fraco', dict(alpha=0.05), dict(g=1.0)),
    ('alpha forte, g fraco', dict(alpha=1.0), dict(g=1.0)),
    ('alpha fraco, g forte', dict(alpha=0.1), dict(g=4.0)),
    ('Ambos fortes',  dict(alpha=1.0), dict(g=4.0)),
]

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.flatten()

for ax, (label, bk, ck) in zip(axes, regimes):
    bath_r = BathParams(m0=1.0, k0=2.0, **bk)
    coup_r = CouplingParams(**ck)
    A_r = collective_spectral_function(OMG, QG, bath_r, coup_r)
    A_safe = np.maximum(A_r, 1e-5)
    im = ax.pcolormesh(QG, OMG, A_safe,
                        norm=LogNorm(vmin=1e-3, vmax=max(A_safe.max(), 0.01)),
                        shading='auto', cmap='inferno')
    plt.colorbar(im, ax=ax, label=r'$A_{\rm col}$')

    ax.plot(q_overlay, om_high, 'w--', lw=1, alpha=0.6)
    ax.plot(q_overlay, om_low, 'w:', lw=1, alpha=0.6)
    ax.axvline(2.0, color='magenta', ls=':', alpha=0.5)
    ax.set_xlabel(r'$q/k_F$'); ax.set_ylabel(r'$\omega$')
    ax.set_title(rf'{label}: $\alpha={bath_r.alpha}$, $g={coup_r.g}$')

fig.suptitle('Modos coletivos em quatro regimes',
             fontsize=14)
plt.tight_layout()
plt.savefig(f'{FIGDIR}/fig16_modes_regimes.png', dpi=140, bbox_inches='tight')
plt.close()
print(f"    salvo: fig16_modes_regimes.png")


# =============================================================
# FIG 17: Dispersão omega_mode(q) — variação com alpha
# =============================================================

print("[FIG 17] Dispersão do modo: varredura alpha e g")

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# (a) Dispersão vs alpha
q_disp = np.linspace(0.1, 1.8, 35)
alphas = [0.05, 0.2, 0.5, 1.0]
for alpha, c in zip(alphas, plt.cm.viridis(np.linspace(0.15, 0.85, len(alphas)))):
    bath_d = BathParams(m0=1.0, k0=2.0, alpha=alpha)
    coup_d = CouplingParams(g=2.5)
    disp = collective_dispersion(q_disp, bath_d, coup_d, above_continuum_only=True)
    valid = ~np.isnan(disp['omega_mode'])
    axes[0].plot(q_disp[valid], disp['omega_mode'][valid], 'o-',
                  color=c, label=rf'$\alpha={alpha}$', markersize=4, lw=1.5)

# continuum band
inf_b, sup_b = continuum_boundaries(q_disp, kF=coup_d.kF, me=coup_d.me)
axes[0].fill_between(q_disp, inf_b, sup_b, alpha=0.2, color='gray',
                      label='continuum p-h')
axes[0].set_xlabel(r'$q/k_F$'); axes[0].set_ylabel(r'$\omega_{\rm modo}(q)$')
axes[0].set_title(r'(a) Dispersão: variando $\alpha$ ($g=2.5$)')
axes[0].legend(fontsize=9)

# (b) Largura do modo vs alpha
alphas_w = np.linspace(0.02, 1.5, 25)
g_widths = [1.5, 2.5, 4.0]
q_for_width = 0.6
for g_val, c in zip(g_widths, plt.cm.plasma(np.linspace(0.2, 0.8, len(g_widths)))):
    widths = []
    for alpha in alphas_w:
        b = BathParams(m0=1.0, k0=2.0, alpha=alpha)
        c2 = CouplingParams(g=g_val)
        mode = find_collective_mode(q_for_width, b, c2)
        widths.append(mode['fwhm'])
    widths = np.array(widths)
    valid = ~np.isnan(widths)
    axes[1].plot(alphas_w[valid], widths[valid], 'o-',
                  color=c, label=rf'$g={g_val}$', markersize=4)

axes[1].plot(alphas_w, alphas_w, 'k:', alpha=0.5, label=r'FWHM=$\alpha$')
axes[1].set_xlabel(r'$\alpha$'); axes[1].set_ylabel(r'FWHM em $q=0.6\,k_F$')
axes[1].set_title(r'(b) Largura do modo vs $\alpha$')
axes[1].legend(fontsize=9)

plt.tight_layout()
plt.savefig(f'{FIGDIR}/fig17_dispersion_width.png', dpi=140, bbox_inches='tight')
plt.close()
print(f"    salvo: fig17_dispersion_width.png")


# =============================================================
# FIG 18: Efeito de k_0 sobre os modos
# =============================================================

print("[FIG 18] Modos coletivos: variando k_0")

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

k0_vals = [0.8, 2.0, 5.0]
for ax, k0 in zip(axes, k0_vals):
    bath_k = BathParams(m0=1.0, k0=k0, alpha=0.3)
    coup_k = CouplingParams(g=2.5)
    A_k = collective_spectral_function(OMG, QG, bath_k, coup_k)
    A_safe = np.maximum(A_k, 1e-5)
    im = ax.pcolormesh(QG, OMG, A_safe,
                        norm=LogNorm(vmin=1e-3, vmax=max(A_safe.max(), 0.01)),
                        shading='auto', cmap='inferno')
    plt.colorbar(im, ax=ax, label=r'$A_{\rm col}$')

    ax.plot(q_overlay, om_high, 'w--', lw=1, alpha=0.6)
    ax.plot(q_overlay, om_low, 'w:', lw=1, alpha=0.6)
    ax.axvline(k0, color='magenta', ls=':', lw=2, alpha=0.8)
    ax.text(k0+0.05, 3.5, rf'$k_0={k0}$', color='magenta', fontsize=11)
    ax.set_xlabel(r'$q/k_F$'); ax.set_ylabel(r'$\omega$')
    ax.set_title(rf'$k_0={k0}\,k_F$, $E_0={bath_k.E0:.2g}$')

fig.suptitle(r'Efeito do suporte do banho $k_0$ sobre os modos coletivos',
             fontsize=13)
plt.tight_layout()
plt.savefig(f'{FIGDIR}/fig18_modes_vs_k0.png', dpi=140, bbox_inches='tight')
plt.close()
print(f"    salvo: fig18_modes_vs_k0.png")


# =============================================================
# FIG 19: Efeito de m_0 sobre os modos (banho leve vs íon)
# =============================================================

print("[FIG 19] Modos coletivos: variando m_0 (leve vs íon)")

# Para fazer comparação justa, devemos comparar em escalas de omega NORMALIZADAS por E_0
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

m0_vals = [1.0, 100.0, 1000.0]
for ax, m0 in zip(axes, m0_vals):
    bath_m = BathParams(m0=m0, k0=2.0, alpha=0.3)
    coup_m = CouplingParams(g=2.5)
    # adaptar grade omega à escala E_0 do banho
    omega_norm = np.linspace(0.01, 4.0, 400) * bath_m.E0 if bath_m.E0 < 1 else np.linspace(0.01, 4.0, 400)
    omega_use = np.linspace(0.01, 4.0 * bath_m.E0, 400)
    OMG_m, QG_m = np.meshgrid(omega_use, q_grid, indexing='ij')
    A_m = collective_spectral_function(OMG_m, QG_m, bath_m, coup_m)
    A_safe = np.maximum(A_m, 1e-7)
    im = ax.pcolormesh(QG_m, OMG_m/bath_m.E0, A_safe,
                        norm=LogNorm(vmin=1e-4, vmax=max(A_safe.max(), 1e-3)),
                        shading='auto', cmap='inferno')
    plt.colorbar(im, ax=ax, label=r'$A_{\rm col}$')

    # continuum em q
    inf_b_m, sup_b_m = continuum_boundaries(q_overlay, kF=coup_m.kF, me=coup_m.me)
    ax.plot(q_overlay, sup_b_m/bath_m.E0, 'w--', lw=1, alpha=0.6)

    ax.axvline(2.0, color='magenta', ls=':', alpha=0.8)
    ax.set_xlabel(r'$q/k_F$'); ax.set_ylabel(r'$\omega/E_0$')
    ax.set_title(rf'$m_0={m0:.0f}\,m_e$, $E_0={bath_m.E0:.2g}\,E_F$')

fig.suptitle(r'Modos coletivos: do banho leve ao banho iônico (escala $E_0$ comprime)',
             fontsize=13)
plt.tight_layout()
plt.savefig(f'{FIGDIR}/fig19_modes_vs_m0.png', dpi=140, bbox_inches='tight')
plt.close()
print(f"    salvo: fig19_modes_vs_m0.png")


print("\nAnálise dos modos coletivos concluída.")
