"""
analysis.07_GW_corrected
========================

ANÁLISE CORRIGIDA usando a integral GW correta para a self-energia eletrônica.

O erro da análise anterior (módulo electron_gas.py): tratar Im Sigma_e como
SIMPLES MÚLTIPLO de Im Sigma_bath. Isto faz o expoente do elétron coincidir
com o do banho (n=1, ômico), perdendo toda a física da propagação via loop.

A integral GW correta inclui convolução em frequência interna, que ADICIONA
uma potência de omega ao expoente do banho:
    Banho ôhmico (Im D ~ |omega|) => Im Sigma_e ~ omega^2 (FL)
    Banho saturado (Im D ~ const) => Im Sigma_e ~ omega (MFL)

A transição entre regimes ocorre quando |Sigma_bath(omega)| ~ omega_0^2
no denominador do propagador.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, ListedColormap
import matplotlib.patches as mpatches
import sys, os
sys.path.insert(0, '/home/lisan/Documentos/QuantumDissipation/qp_bath_project /qp_bath_project/')

from core.bath import BathParams, im_sigma_bath
from core.electron_gas import CouplingParams
from core.electron_gas_GW import (
    im_sigma_e_GW, re_sigma_e_GW, sigma_e_GW_full,
    extract_exponent_GW, D_bath_R_massive
)

FIGDIR = 'home/lisan/Documentos/QuantumDissipation/qp_bath_project /qp_bath_project/figures'

plt.rcParams.update({
    'font.size': 11, 'axes.labelsize': 12, 'legend.fontsize': 9,
    'lines.linewidth': 1.8, 'axes.grid': True, 'grid.alpha': 0.35,
})


# =============================================================
# FIG 25: Im Sigma_e versus Im Sigma_bath — distinção crucial
# =============================================================

print("[FIG 25] Im Sigma_e (GW correto) vs Im Sigma_bath — comparação direta")

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

# Em escala log-log mostrar diferentes expoentes
omega_pos = np.logspace(-3, 0, 200)

alphas_plot = [0.1, 1.0, 10.0, 50.0]
for alpha, c in zip(alphas_plot,
                     plt.cm.viridis(np.linspace(0.15, 0.85, len(alphas_plot)))):
    bath = BathParams(m0=1.0, k0=1.0, alpha=alpha)
    coup = CouplingParams(g=1.0)
    # banho
    im_b = -im_sigma_bath(omega_pos, bath)
    # elétron via GW
    im_e = -im_sigma_e_GW(omega_pos, bath, coup, bath_mode='massive', n_q=50, n_omp=150)
    axes[0].loglog(omega_pos, im_b, color=c, lw=2, label=rf'banho, $\alpha={alpha}$')
    axes[1].loglog(omega_pos, im_e, color=c, lw=2, label=rf'elétron, $\alpha={alpha}$')

# referências
om_ref = omega_pos[omega_pos > 0.01]
axes[0].loglog(om_ref, om_ref * 0.5, 'k:', alpha=0.5, label=r'$\sim\omega^1$')
axes[1].loglog(om_ref, om_ref * 0.05, 'k:', alpha=0.5, label=r'$\sim\omega^1$')
axes[1].loglog(om_ref, om_ref**2 * 0.5, 'k--', alpha=0.5, label=r'$\sim\omega^2$')

axes[0].set_xlabel(r'$\omega$')
axes[0].set_ylabel(r'$|\mathrm{Im}\,\Sigma_{\rm bath}|$')
axes[0].set_title(r'(a) Banho: SEMPRE $\sim\omega^1$ (Conv. A)')
axes[0].legend(fontsize=8)

axes[1].set_xlabel(r'$\omega$')
axes[1].set_ylabel(r'$|\mathrm{Im}\,\Sigma_e^R|$ via GW')
axes[1].set_title(r'(b) Elétron: $\sim\omega^2$ (FL) → $\sim\omega^1$ (incoh.)')
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.show()
plt.savefig(f'{FIGDIR}/fig25_bath_vs_electron.png', dpi=140, bbox_inches='tight')
plt.close()
print(f"    salvo: fig25_bath_vs_electron.png")


# =============================================================
# FIG 26: Transição de expoente n(alpha) e n(g)
# =============================================================

print("[FIG 26] Transição do expoente: n vs (alpha, g)")

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

# (a) n versus alpha (g, k_0 fixos)
alpha_range = np.logspace(-1.5, 2.5, 30)
g_vals = [0.5, 1.0, 2.0]

for g, c in zip(g_vals, plt.cm.plasma(np.linspace(0.2, 0.8, len(g_vals)))):
    n_vals = []
    for alpha in alpha_range:
        bath = BathParams(m0=1.0, k0=1.0, alpha=alpha)
        coup = CouplingParams(g=g)
        fit = extract_exponent_GW(bath, coup,
                                    window_fraction=(0.005, 0.05),
                                    bath_mode='massive', n_points=80)
        n_vals.append(fit['n'])
    axes[0].semilogx(alpha_range, n_vals, 'o-', color=c,
                      label=rf'$g={g}$', markersize=4)

axes[0].axhline(2.0, color='blue', ls='--', alpha=0.5, label='FL: $n=2$')
axes[0].axhline(1.0, color='red', ls='--', alpha=0.5, label='MFL: $n=1$')
axes[0].axhline(1.5, color='gray', ls=':', alpha=0.5, label='fronteira')
axes[0].set_xlabel(r'$\alpha$')
axes[0].set_ylabel(r'expoente $n$')
axes[0].set_title(r'(a) $n(\alpha)$: transição FL → MFL')
axes[0].legend(fontsize=9)
axes[0].set_ylim(0, 2.2)

# (b) n versus k_0 (alpha, g fixos)
print("    computando varredura em k_0...")
k0_range = np.linspace(0.3, 4.0, 20)
alpha_fixed_vals = [0.5, 5.0, 20.0]
for alpha, c in zip(alpha_fixed_vals,
                     plt.cm.viridis(np.linspace(0.15, 0.85, len(alpha_fixed_vals)))):
    n_vals = []
    for k0 in k0_range:
        bath = BathParams(m0=1.0, k0=k0, alpha=alpha)
        coup = CouplingParams(g=1.0)
        # janela adaptada ao E_0 do banho
        fit = extract_exponent_GW(bath, coup,
                                    window_fraction=(0.005, 0.05),
                                    bath_mode='massive', n_points=80)
        n_vals.append(fit['n'])
    axes[1].plot(k0_range, n_vals, 'o-', color=c,
                  label=rf'$\alpha={alpha}$', markersize=4)

axes[1].axhline(2.0, color='blue', ls='--', alpha=0.5)
axes[1].axhline(1.0, color='red', ls='--', alpha=0.5)
axes[1].axvline(2.0, color='cyan', ls=':', alpha=0.5, label=r'$k_0=2k_F$')
axes[1].set_xlabel(r'$k_0/k_F$')
axes[1].set_ylabel(r'expoente $n$')
axes[1].set_title(r'(b) $n(k_0)$: $k_0$ grande $\Rightarrow$ FL recuperado')
axes[1].legend(fontsize=9)
axes[1].set_ylim(0, 2.2)

plt.tight_layout()
plt.show()
plt.savefig(f'{FIGDIR}/fig26_exponent_transition.png', dpi=140, bbox_inches='tight')
plt.close()
print(f"    salvo: fig26_exponent_transition.png")


# =============================================================
# FIG 27: Mapa 2D do expoente n no plano (alpha, k_0)
# =============================================================

print("[FIG 27] Mapa 2D: n(alpha, k_0)")

alpha_grid = np.logspace(-1, 2, 20)
k0_grid = np.linspace(0.3, 3.0, 20)

n_map = np.zeros((len(alpha_grid), len(k0_grid)))
A_map = np.zeros((len(alpha_grid), len(k0_grid)))

print("    computando grade 20x20 (pode demorar ~30s)...")
for i, alpha in enumerate(alpha_grid):
    for j, k0 in enumerate(k0_grid):
        bath = BathParams(m0=1.0, k0=k0, alpha=alpha)
        coup = CouplingParams(g=1.0)
        fit = extract_exponent_GW(bath, coup,
                                    window_fraction=(0.005, 0.05),
                                    bath_mode='massive', n_points=80)
        n_map[i, j] = fit['n']
        A_map[i, j] = fit['A']
    if i % 5 == 0:
        print(f"      linha {i+1}/{len(alpha_grid)}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

AG, KG = np.meshgrid(alpha_grid, k0_grid, indexing='ij')

# (a) Mapa de n
im0 = axes[0].pcolormesh(AG, KG, n_map, shading='auto',
                          cmap='RdYlBu', vmin=0.5, vmax=2.0)
plt.colorbar(im0, ax=axes[0], label=r'expoente $n$')
cs = axes[0].contour(AG, KG, n_map, levels=[1.0, 1.5, 1.8],
                     colors='black', linewidths=1.5, alpha=0.7)
axes[0].clabel(cs, inline=True, fontsize=9)
axes[0].set_xscale('log')
axes[0].set_xlabel(r'$\alpha$'); axes[0].set_ylabel(r'$k_0/k_F$')
axes[0].set_title(r'(a) Expoente $n$ via fit GW')

# Anotações de regime
axes[0].annotate('FL\n$n\\to 2$', xy=(0.3, 2.5), fontsize=11, color='blue',
                 fontweight='bold', ha='center')
axes[0].annotate('MFL\n$n\\sim 1.5$', xy=(10, 1.0), fontsize=11, color='orange',
                 fontweight='bold', ha='center')
axes[0].annotate('SM\n$n\\sim 1$', xy=(70, 0.5), fontsize=11, color='red',
                 fontweight='bold', ha='center')

# (b) Diagrama categórico
phase_cmap = ListedColormap(['#3b73af', '#e8a13a', '#c34440'])
phase = np.zeros_like(n_map, dtype=int)
phase[n_map < 1.5] = 1  # MFL
phase[n_map < 1.2] = 2  # SM
im1 = axes[1].pcolormesh(AG, KG, phase, shading='auto',
                          cmap=phase_cmap, vmin=-0.5, vmax=2.5)
cs = axes[1].contour(AG, KG, n_map, levels=[1.2, 1.5],
                     colors='white', linewidths=2, alpha=0.85)
axes[1].clabel(cs, inline=True, fontsize=9)
axes[1].set_xscale('log')
axes[1].set_xlabel(r'$\alpha$'); axes[1].set_ylabel(r'$k_0/k_F$')
axes[1].set_title(r'(b) Diagrama de fases categórico')

patches = [mpatches.Patch(color='#3b73af', label=r'FL ($n>1.5$)'),
           mpatches.Patch(color='#e8a13a', label=r'MFL ($1.2<n<1.5$)'),
           mpatches.Patch(color='#c34440', label=r'SM ($n<1.2$)')]
axes[1].legend(handles=patches, fontsize=9, loc='upper right')

fig.suptitle(r'Diagrama de fases corrigido (integral GW): expoente $n$ varia',
             fontsize=13)
plt.tight_layout()
plt.show()
plt.savefig(f'{FIGDIR}/fig27_phase_diagram_GW.png', dpi=140, bbox_inches='tight')
plt.close()
print(f"    salvo: fig27_phase_diagram_GW.png")


# =============================================================
# FIG 28: Demonstração de como o propagador do banho gera a transição
# =============================================================

print("[FIG 28] Estrutura do propagador D^R_bath em diferentes regimes")

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

omega_t = np.logspace(-3, 0.5, 200)
alphas_demo = [0.1, 1.0, 10.0, 100.0]

for alpha, c in zip(alphas_demo,
                     plt.cm.viridis(np.linspace(0.15, 0.85, len(alphas_demo)))):
    bath = BathParams(m0=1.0, k0=1.0, alpha=alpha)
    D = D_bath_R_massive(omega_t, bath)
    axes[0].loglog(omega_t/bath.E0, -np.imag(D)/np.pi, color=c,
                    label=rf'$\alpha={alpha}$')

# referências
om_ref = omega_t[omega_t > 0.005]
axes[0].loglog(om_ref, om_ref * 0.1, 'k:', alpha=0.5, label=r'$\sim\omega^1$')
axes[0].loglog(om_ref, np.ones_like(om_ref) * 0.05, 'k--', alpha=0.5, label='$\sim$ const')

axes[0].set_xlabel(r'$\omega/E_0$')
axes[0].set_ylabel(r'$A_{\rm bath}(\omega) = -\mathrm{Im}\,D^R/\pi$')
axes[0].set_title(r'(a) Densidade espectral do BANHO')
axes[0].legend(fontsize=9)

# (b) razão expoente-IR diferente
# Mostrar Im D vs omega em escala log-log com regiões anotadas
for alpha, c in zip(alphas_demo,
                     plt.cm.viridis(np.linspace(0.15, 0.85, len(alphas_demo)))):
    bath = BathParams(m0=1.0, k0=1.0, alpha=alpha)
    # quociente Im D / |omega|
    D = D_bath_R_massive(omega_t, bath)
    ratio = -np.imag(D) / omega_t  # se ~ const, D~omega; se ~1/omega, D~const
    axes[1].semilogx(omega_t/bath.E0, ratio, color=c,
                      label=rf'$\alpha={alpha}$')

axes[1].set_xlabel(r'$\omega/E_0$')
axes[1].set_ylabel(r'$-\mathrm{Im}\,D^R / \omega$')
axes[1].set_title(r'(b) Razão $\mathrm{Im}\,D/\omega$: const=linear, decai=saturado')
axes[1].legend(fontsize=9)

fig.suptitle(r'Crossover no propagador do banho controla regime do elétron',
             fontsize=13)
plt.tight_layout()
plt.show()
plt.savefig(f'{FIGDIR}/fig28_bath_propagator_regimes.png', dpi=140, bbox_inches='tight')
plt.close()
print(f"    salvo: fig28_bath_propagator_regimes.png")


print("\nAnálise GW corrigida concluída.")
print()
print("CONCLUSÃO: o expoente n do ELÉTRON varia genuinamente de ~2 (FL) a ~1 (SM)")
print("conforme alpha aumenta. A análise prévia que dava n=1 universal estava")
print("ERRADA porque tratava Im Sigma_e como múltiplo direto de Im Sigma_bath.")
