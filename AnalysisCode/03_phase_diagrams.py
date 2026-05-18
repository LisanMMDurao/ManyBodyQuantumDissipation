"""
analysis.03_phase_diagrams
==========================

Diagramas de fases do gás de elétrons acoplado ao banho de quasipartículas.

Eixos explorados:
    - Plano (alpha, g) com (m_0, k_0) fixos: regimes FL / MFL / incoerente
    - Plano (k_0/k_F, m_0/m_e) com (alpha, g) fixos: efeito da geometria do banho
    - Plano (omega, alpha): janela de coerência em função da escala

Critérios:
    lambda = pi alpha g^2 N(0) F(k_0/k_F)
    FL:          lambda < 0.1
    MFL:    0.1 <= lambda < 1.0
    incoerente:  lambda >= 1.0
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, ListedColormap
import sys, os
sys.path.insert(0, '/home/claude/qp_bath_project')

from core.bath import BathParams
from core.electron_gas import CouplingParams
from core.phase_analysis import (
    scan_alpha_g, scan_k0_m0, scan_omega_alpha,
    dimensionless_coupling, classify_phase
)

FIGDIR = '/home/lisan/Documentos/QuantumDissipation/qp_bath_project /qp_bath_project/'
os.makedirs(FIGDIR, exist_ok=True)

plt.rcParams.update({
    'font.size': 11, 'axes.labelsize': 12, 'legend.fontsize': 9,
    'lines.linewidth': 1.8, 'axes.grid': True, 'grid.alpha': 0.35,
})

phase_cmap = ListedColormap(['#3b73af', '#e8a13a', '#c34440'])


# =============================================================
# FIG 10: Diagrama de fases (alpha, g)
# =============================================================

print("[FIG 10] Diagrama de fases (alpha, g) — quatro configurações de banho")

alpha_grid = np.logspace(-2, 1, 30)
g_grid = np.linspace(0.1, 5.0, 30)

# Quatro configurações: combinações de (m_0, k_0) representativas
configs = [
    ('(a) $m_0=m_e$, $k_0=k_F$', 1.0, 1.0),
    ('(b) $m_0=m_e$, $k_0=3k_F$', 1.0, 3.0),
    ('(c) $m_0=100\\,m_e$, $k_0=k_F$', 100.0, 1.0),
    ('(d) $m_0=1000\\,m_e$, $k_0=k_F$', 1000.0, 1.0),
]

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
axes = axes.flatten()

for ax, (label, m0, k0r) in zip(axes, configs):
    result = scan_alpha_g(alpha_grid, g_grid, m0=m0, k0_over_kF=k0r)

    AG, GG = np.meshgrid(result['alpha_grid'], result['g_grid'], indexing='ij')
    phase_safe = np.clip(result['phase'], -0.5, 2.5)
    im = ax.pcolormesh(AG, GG, phase_safe, shading='auto',
                       cmap=phase_cmap, vmin=-0.5, vmax=2.5)

    # Contornos de lambda
    cs = ax.contour(AG, GG, result['lambda'], levels=[0.1, 1.0],
                    colors='white', linewidths=2, alpha=0.85)
    ax.clabel(cs, inline=True, fontsize=9, fmt={0.1: 'λ=0.1', 1.0: 'λ=1'})

    #ax.set_xscale('log')
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'$g$')
    p_ex = BathParams(m0=m0, k0=k0r, alpha=0.1)
    ax.set_title(f'{label} ($E_0={p_ex.E0:.3g}\\,E_F$)')

# Legenda comum
fig.subplots_adjust(top=0.92)
fig.suptitle('Diagrama de fases no plano $(\\alpha, g)$', fontsize=14, y=0.97)
import matplotlib.patches as mpatches
patches = [mpatches.Patch(color='#3b73af', label='FL coerente'),
           mpatches.Patch(color='#e8a13a', label='Marginal FL'),
           mpatches.Patch(color='#c34440', label='Incoerente')]
fig.legend(handles=patches, loc='lower center', ncol=3, fontsize=11,
           bbox_to_anchor=(0.5, -0.01))
plt.tight_layout(rect=[0, 0.02, 1, 0.95])
plt.show()
plt.savefig(f'{FIGDIR}/fig10_phase_alpha_g.png', dpi=140, bbox_inches='tight')
plt.close()
print(f"    salvo: fig10_phase_alpha_g.png")


# =============================================================
# FIG 11: Plano (k_0/k_F, m_0/m_e) — geometria do banho
# =============================================================

print("[FIG 11] Diagrama de fases (k_0/k_F, m_0/m_e)")

k0_grid = np.linspace(0.2, 5.0, 30)
m0_grid = np.logspace(0, 3.5, 30)  # 1 a ~3000

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

scan_configs = [
    (r'$\alpha=0.1$, $g=1$ (fraco)', 0.1, 1.0),
    (r'$\alpha=0.5$, $g=2$ (médio)', 0.5, 2.0),
    (r'$\alpha=1.0$, $g=3$ (forte)', 1.0, 3.0),
]

for ax, (label, alpha, g) in zip(axes, scan_configs):
    result = scan_k0_m0(k0_grid, m0_grid, alpha=alpha, g=g)

    KG, MG = np.meshgrid(result['k0_grid'], result['m0_grid'], indexing='ij')
    phase_safe = np.clip(result['phase'], -0.5, 2.5)
    im = ax.pcolormesh(KG, MG, phase_safe, shading='auto',
                        cmap=phase_cmap, vmin=-0.5, vmax=2.5)

    cs = ax.contour(KG, MG, result['lambda'], levels=[0.1, 1.0],
                     colors='white', linewidths=2, alpha=0.85)
    ax.clabel(cs, inline=True, fontsize=9, fmt={0.1:'λ=0.1', 1.0:'λ=1'})

    # linha k_0 = 2 k_F: transição de F < 1 para F = 1
    ax.axvline(2.0, color='cyan', ls=':', alpha=0.7, lw=2)
    ax.text(2.05, m0_grid[-1]*0.7, r'$k_0=2k_F$', color='cyan', fontsize=10)

    ax.set_yscale('log')
    ax.set_xlabel(r'$k_0 / k_F$')
    ax.set_ylabel(r'$m_0 / m_e$')
    ax.set_title(label)

fig.suptitle('Geometria do banho: efeito de $(k_0, m_0)$', fontsize=14)
plt.tight_layout()
plt.show()
plt.savefig(f'{FIGDIR}/fig11_phase_k0_m0.png', dpi=140, bbox_inches='tight')
plt.close()
print(f"    salvo: fig11_phase_k0_m0.png")


# =============================================================
# FIG 12: Métricas detalhadas (lambda, Z, E0)
# =============================================================

print("[FIG 12] Métricas detalhadas no plano (k0, m0)")

result = scan_k0_m0(k0_grid, m0_grid, alpha=0.5, g=2.0)
KG, MG = np.meshgrid(result['k0_grid'], result['m0_grid'], indexing='ij')

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

# (a) lambda
im0 = axes[0].pcolormesh(KG, MG, result['lambda'],
                          norm=LogNorm(vmin=0.01, vmax=10),
                          shading='auto', cmap='RdYlBu_r')
plt.colorbar(im0, ax=axes[0], label=r'$\lambda$')
axes[0].set_yscale('log')
axes[0].set_xlabel(r'$k_0/k_F$'); axes[0].set_ylabel(r'$m_0/m_e$')
axes[0].set_title(r'(a) Acopl. efetivo $\lambda$')

# (b) Z em omega = E_0/10
im1 = axes[1].pcolormesh(KG, MG, result['Z_omega'], shading='auto',
                          vmin=0.5, vmax=1.0, cmap='viridis')
plt.colorbar(im1, ax=axes[1], label=r'$Z(\omega=E_0/10)$')
axes[1].set_yscale('log')
axes[1].set_xlabel(r'$k_0/k_F$'); axes[1].set_ylabel(r'$m_0/m_e$')
axes[1].set_title(r'(b) Peso de quasipartícula em $\omega=E_0/10$')

# (c) E_0
im2 = axes[2].pcolormesh(KG, MG, result['E0'],
                          norm=LogNorm(vmin=1e-4, vmax=10),
                          shading='auto', cmap='inferno')
plt.colorbar(im2, ax=axes[2], label=r'$E_0/E_F$')
axes[2].set_yscale('log')
axes[2].set_xlabel(r'$k_0/k_F$'); axes[2].set_ylabel(r'$m_0/m_e$')
axes[2].set_title(r'(c) Cutoff $E_0 = \hbar^2 k_0^2/(2m_0)$')

fig.suptitle(r'Métricas em $(\alpha=0.5, g=2)$', fontsize=13)
plt.tight_layout()
plt.show()
plt.savefig(f'{FIGDIR}/fig12_metrics_k0_m0.png', dpi=140, bbox_inches='tight')
plt.close()
print(f"    salvo: fig12_metrics_k0_m0.png")


# =============================================================
# FIG 13: Janela de coerência no plano (omega, alpha)
# =============================================================

print("[FIG 13] Janela de coerência (omega, alpha)")

omega_grid = np.linspace(0.01, 2.0, 100)
alpha_grid_2 = np.logspace(-2, 1, 60)

scan_configs2 = [
    (r'$m_0=m_e$, $k_0=k_F$, $g=1$', dict(m0=1.0, k0=1.0), dict(g=1.0)),
    (r'$m_0=m_e$, $k_0=k_F$, $g=3$', dict(m0=1.0, k0=1.0), dict(g=3.0)),
    (r'$m_0=100$, $k_0=k_F$, $g=2$', dict(m0=100, k0=1.0), dict(g=2.0)),
]

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

for ax, (label, bk, ck) in zip(axes, scan_configs2):
    result = scan_omega_alpha(omega_grid, alpha_grid_2,
                               bath_kwargs=bk, coupling_kwargs=ck)
    OG, AG = np.meshgrid(result['omega_grid'], result['alpha_grid'], indexing='ij')

    eta_safe = np.clip(result['eta'], 1e-4, 100)
    im = ax.pcolormesh(AG, OG, eta_safe,
                        norm=LogNorm(vmin=1e-2, vmax=10),
                        shading='auto', cmap='RdYlBu_r')
    plt.colorbar(im, ax=ax, label=r'$\eta = \Gamma/(2\omega)$')

    # contornos críticos: eta = 0.1 (coerente), 1.0 (marginal), 10 (overdamped)
    cs = ax.contour(AG, OG, result['eta'], levels=[0.1, 1.0],
                    colors='white', linewidths=2, alpha=0.85)
    ax.clabel(cs, inline=True, fontsize=9, fmt={0.1:'η=0.1', 1.0:'η=1'})

    #ax.set_xscale('log')
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'$\omega$')
    ax.set_title(label)

fig.suptitle(r'Janela de coerência: $\eta = \Gamma/(2\omega) < 1$ é regime de QP bem definida',
             fontsize=13)
plt.tight_layout()
plt.show()
plt.savefig(f'{FIGDIR}/fig13_coherence_window.png', dpi=140, bbox_inches='tight')
plt.close()
print(f"    salvo: fig13_coherence_window.png")


# =============================================================
# RESUMO ESTATÍSTICO
# =============================================================

print("\n" + "="*70)
print("RESUMO ESTATÍSTICO")
print("="*70)

for label, m0, k0r in configs:
    result = scan_alpha_g(alpha_grid, g_grid, m0=m0, k0_over_kF=k0r)
    phase = result['phase']
    total = phase.size
    fl = np.sum(phase == 0); mfl = np.sum(phase == 1); inc = np.sum(phase == 2)
    print(f"\n{label}:")
    print(f"  FL: {100*fl/total:.1f}%, MFL: {100*mfl/total:.1f}%, Incoh: {100*inc/total:.1f}%")
    print(f"  lambda range: [{result['lambda'].min():.4f}, {result['lambda'].max():.4f}]")

print("\nDiagramas de fases concluídos.")
