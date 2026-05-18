"""
analysis.08_rpa_plasmons_lifetimes
==================================

ANÁLISE RPA COMPLETA:
    - Função dielétrica eps(omega, q)
    - Função resposta chi(omega, q)
    - Densidade espectral A_col = -Im chi / pi
    - Loss function -Im[1/eps]
    - Dispersão de plasmons omega_p(q) e tempo de vida Gamma_p(q)
    - Variação dos modos coletivos com os regimes (alpha, g, k_0)
    - TEMPO DE VIDA exibindo as diferentes fases

A interação é MEDIADA UNICAMENTE PELO BANHO:
    V_eff(omega) = g^2 * D_bath^R(omega)

A resposta RPA com esta interação produz modos coletivos que herdam:
    - frequência ~ sqrt(E_0^2 + correções RPA)
    - largura ~ Im V_eff * Re chi_0 (proporcional à dissipação do banho)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm, ListedColormap
import matplotlib.patches as mpatches
import sys, os
sys.path.insert(0, '/home/lisan/Documentos/QuantumDissipation/qp_bath_project /qp_bath_project/')

from core.bath import BathParams
from core.electron_gas import CouplingParams
from core.electron_gas_GW import im_sigma_e_GW
from core.rpa_response import (
    lindhard_3d, V_bath_mediated, dielectric_RPA, response_RPA,
    spectral_density, loss_function_EELS,
    plasmon_dispersion, continuum_boundaries
)

FIGDIR = 'home/lisan/Documentos/QuantumDissipation/qp_bath_project /qp_bath_project/figures'

plt.rcParams.update({
    'font.size': 11, 'axes.labelsize': 12, 'legend.fontsize': 9,
    'lines.linewidth': 1.8, 'axes.grid': True, 'grid.alpha': 0.35,
})


# =============================================================
# CASO DE REFERÊNCIA
# =============================================================
bath_ref = BathParams(m0=1.0, k0=2.0, alpha=0.1)
coup_ref = CouplingParams(g=10.0)


# =============================================================
# FIG 29: Função dielétrica eps(omega, q) — Re e Im
# =============================================================

print("[FIG 29] Função dielétrica RPA")

omega = np.linspace(0.01, 5.0, 400)
q_grid = np.linspace(0.05, 2.5, 150)
OMG, QG = np.meshgrid(omega, q_grid, indexing='ij')

eps_grid = dielectric_RPA(OMG, QG, bath_ref, coup_ref)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Re eps
re_eps = np.real(eps_grid)
vmax_re = np.percentile(np.abs(re_eps), 95)
im0 = axes[0].pcolormesh(QG, OMG, re_eps,
                          vmin=-vmax_re, vmax=vmax_re,
                          shading='auto', cmap='RdBu_r')
plt.colorbar(im0, ax=axes[0], label=r'$\mathrm{Re}\,\varepsilon$')

# linha de zero de Re eps
cs = axes[0].contour(QG, OMG, re_eps, levels=[0], colors='black',
                     linewidths=2, alpha=0.9)
axes[0].clabel(cs, inline=True, fontsize=10, fmt='Re ε = 0')

# Continuum
q_overlay = np.linspace(0.05, 2.5, 200)
inf_b, sup_b = continuum_boundaries(q_overlay)
axes[0].plot(q_overlay, sup_b, 'w--', lw=1.5, alpha=0.7, label='borda sup. p-h')
axes[0].plot(q_overlay, inf_b, 'w:', lw=1.5, alpha=0.7)
axes[0].set_xlabel(r'$q/k_F$'); axes[0].set_ylabel(r'$\omega/E_F$')
axes[0].set_title(r'(a) $\mathrm{Re}\,\varepsilon(\omega, q)$: zero = plasmon')
axes[0].legend(loc='upper left', fontsize=9)

# Im eps
im_eps = np.imag(eps_grid)
im1 = axes[1].pcolormesh(QG, OMG, im_eps,
                          norm=SymLogNorm(linthresh=1e-3, vmin=-1, vmax=1),
                          shading='auto', cmap='RdBu_r')
plt.colorbar(im1, ax=axes[1], label=r'$\mathrm{Im}\,\varepsilon$')
axes[1].plot(q_overlay, sup_b, 'k--', lw=1, alpha=0.6)
axes[1].plot(q_overlay, inf_b, 'k:', lw=1, alpha=0.6)
axes[1].set_xlabel(r'$q/k_F$'); axes[1].set_ylabel(r'$\omega/E_F$')
axes[1].set_title(r'(b) $\mathrm{Im}\,\varepsilon$: dissipação')

fig.suptitle(rf'Função dielétrica RPA: $\alpha={bath_ref.alpha}$, $g={coup_ref.g}$, '
             rf'$k_0={bath_ref.k0}\,k_F$',
             fontsize=13)
plt.tight_layout()
plt.show()
plt.savefig(f'{FIGDIR}/fig29_dielectric.png', dpi=140, bbox_inches='tight')
plt.close()
print(f"    salvo: fig29_dielectric.png")


# =============================================================
# FIG 30: Densidade espectral A_col com dispersão de plasmon
# =============================================================

print("[FIG 30] Densidade espectral A_col + dispersão do plasmon")

A_grid = spectral_density(OMG, QG, bath_ref, coup_ref)
loss_grid = loss_function_EELS(OMG, QG, bath_ref, coup_ref)

fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))

# A_col
A_safe = np.maximum(A_grid, 1e-4)
im0 = axes[0].pcolormesh(QG, OMG, A_safe,
                          norm=LogNorm(vmin=1e-3, vmax=A_safe.max()),
                          shading='auto', cmap='inferno')
plt.colorbar(im0, ax=axes[0], label=r'$A_{\rm col} = -\mathrm{Im}\,\chi/\pi$')

# Dispersão por busca de pico
q_disp = np.linspace(0.1, 2.0, 40)
disp = plasmon_dispersion(q_disp, bath_ref, coup_ref, method='peak')
valid = disp['found']
axes[0].plot(q_disp[valid], disp['omega_p'][valid], 'cyan', lw=2.5,
              marker='o', markersize=3, label='plasmon (pico A_col)')

axes[0].plot(q_overlay, sup_b, 'w--', lw=1.5, alpha=0.6, label='borda p-h')
axes[0].plot(q_overlay, inf_b, 'w:', lw=1.5, alpha=0.6)
axes[0].set_xlabel(r'$q/k_F$'); axes[0].set_ylabel(r'$\omega/E_F$')
axes[0].set_title(r'(a) $A_{\rm col}(\omega, q)$ + dispersão do plasmon')
axes[0].legend(loc='upper left', fontsize=9)
axes[0].set_ylim(0, 5)

# Loss function
vmax_loss = np.percentile(np.abs(loss_grid), 98)
im1 = axes[1].pcolormesh(QG, OMG, loss_grid,
                          norm=SymLogNorm(linthresh=1e-3,
                                          vmin=-vmax_loss, vmax=vmax_loss),
                          shading='auto', cmap='RdBu_r')
plt.colorbar(im1, ax=axes[1], label=r'$-\mathrm{Im}[1/\varepsilon]$')
axes[1].plot(q_overlay, sup_b, 'k--', lw=1, alpha=0.6)
axes[1].plot(q_overlay, inf_b, 'k:', lw=1, alpha=0.6)
axes[1].set_xlabel(r'$q/k_F$'); axes[1].set_ylabel(r'$\omega/E_F$')
axes[1].set_title(r'(b) Loss function (EELS)')
axes[1].set_ylim(0, 5)

fig.suptitle(r'Resposta RPA: modos coletivos com interação mediada pelo banho',
             fontsize=13)
plt.tight_layout()
plt.show()
plt.savefig(f'{FIGDIR}/fig30_spectral_density.png', dpi=140, bbox_inches='tight')
plt.close()
print(f"    salvo: fig30_spectral_density.png")


# =============================================================
# FIG 31: Plasmon em 4 regimes do banho
# =============================================================

print("[FIG 31] Plasmon em 4 regimes")

regimes_pl = [
    ('Banho fraco $\\alpha=0.05$', dict(alpha=0.05), dict(g=10.0)),
    ('Banho médio $\\alpha=0.5$',  dict(alpha=0.5),  dict(g=10.0)),
    ('Banho forte $\\alpha=2.0$',  dict(alpha=2.0),  dict(g=10.0)),
    ('Banho muito forte $\\alpha=10$', dict(alpha=10.0), dict(g=10.0)),
]

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
axes = axes.flatten()

for ax, (label, bk, ck) in zip(axes, regimes_pl):
    bath_r = BathParams(m0=1.0, k0=2.0, **bk)
    coup_r = CouplingParams(**ck)
    A_r = spectral_density(OMG, QG, bath_r, coup_r)
    A_safe = np.maximum(A_r, 1e-4)
    im = ax.pcolormesh(QG, OMG, A_safe,
                        norm=LogNorm(vmin=1e-3, vmax=max(A_safe.max(), 0.01)),
                        shading='auto', cmap='inferno')
    plt.colorbar(im, ax=ax, label=r'$A_{\rm col}$')

    # Dispersão
    disp = plasmon_dispersion(q_disp, bath_r, coup_r, method='peak')
    valid = disp['found']
    if valid.sum() > 0:
        ax.plot(q_disp[valid], disp['omega_p'][valid], 'cyan', lw=2, marker='o',
                markersize=2.5)

    ax.plot(q_overlay, sup_b, 'w--', lw=1, alpha=0.5)
    ax.plot(q_overlay, inf_b, 'w:', lw=1, alpha=0.5)
    ax.set_xlabel(r'$q/k_F$'); ax.set_ylabel(r'$\omega$')
    ax.set_title(label)
    ax.set_ylim(0, 5)

fig.suptitle(r'Modos coletivos: degradação com banho mais dissipativo',
             fontsize=13)
plt.tight_layout()
plt.show()
plt.savefig(f'{FIGDIR}/fig31_modes_alpha_scan.png', dpi=140, bbox_inches='tight')
plt.close()
print(f"    salvo: fig31_modes_alpha_scan.png")


# =============================================================
# FIG 32: Dispersão omega_p(q) e largura Gamma_p(q) vs alpha
# =============================================================

print("[FIG 32] Dispersão e largura do plasmon vs alpha")

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

q_fine = np.linspace(0.1, 2.0, 35)
alphas_disp = [0.05, 0.2, 0.5, 1.0, 2.0]

for alpha, c in zip(alphas_disp,
                     plt.cm.viridis(np.linspace(0.15, 0.85, len(alphas_disp)))):
    bath_a = BathParams(m0=1.0, k0=2.0, alpha=alpha)
    coup_a = CouplingParams(g=10.0)
    disp = plasmon_dispersion(q_fine, bath_a, coup_a, method='peak')
    valid = disp['found'] & ~np.isnan(disp['omega_p'])
    axes[0].plot(q_fine[valid], disp['omega_p'][valid], 'o-',
                  color=c, markersize=3, label=rf'$\alpha={alpha}$')

inf_qf, sup_qf = continuum_boundaries(q_fine)
axes[0].fill_between(q_fine, inf_qf, sup_qf, alpha=0.2, color='gray',
                      label='continuum p-h')
axes[0].set_xlabel(r'$q/k_F$'); axes[0].set_ylabel(r'$\omega_p(q)$')
axes[0].set_title(r'(a) Dispersão do plasmon')
axes[0].legend(fontsize=9)
axes[0].set_ylim(0, 5)

# (b) Largura
for alpha, c in zip(alphas_disp,
                     plt.cm.viridis(np.linspace(0.15, 0.85, len(alphas_disp)))):
    bath_a = BathParams(m0=1.0, k0=2.0, alpha=alpha)
    coup_a = CouplingParams(g=10.0)
    disp = plasmon_dispersion(q_fine, bath_a, coup_a, method='peak')
    valid = disp['found'] & ~np.isnan(disp['gamma_p'])
    if valid.sum() > 2:
        axes[1].plot(q_fine[valid], disp['gamma_p'][valid], 'o-',
                      color=c, markersize=3, label=rf'$\alpha={alpha}$')

axes[1].set_xlabel(r'$q/k_F$'); axes[1].set_ylabel(r'$\Gamma_p(q)$')
axes[1].set_title(r'(b) Largura do plasmon (formal via $d\,\mathrm{Re}\,\varepsilon$)')
axes[1].legend(fontsize=9)
axes[1].set_yscale('log')

plt.tight_layout()
plt.show()
plt.savefig(f'{FIGDIR}/fig32_dispersion_widths.png', dpi=140, bbox_inches='tight')
plt.close()
print(f"    salvo: fig32_dispersion_widths.png")


# =============================================================
# FIG 33: Tempo de vida exibindo as diferentes fases
# =============================================================

print("[FIG 33] Tempo de vida vs fases: razão Gamma/omega")

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

# (a) Tempo de vida do PLASMON vs alpha em q fixo
alpha_grid_lt = np.logspace(-2, 2, 30)
q_values = [0.3, 0.6, 1.0]

for q_val, c in zip(q_values, plt.cm.plasma(np.linspace(0.2, 0.8, len(q_values)))):
    eta_vals = []
    omega_p_vals = []
    for alpha in alpha_grid_lt:
        bath_a = BathParams(m0=1.0, k0=2.0, alpha=alpha)
        coup_a = CouplingParams(g=10.0)
        disp = plasmon_dispersion(np.array([q_val]), bath_a, coup_a, method='peak')
        if disp['found'][0] and not np.isnan(disp['gamma_p'][0]):
            eta_vals.append(disp['gamma_p'][0] / (2 * disp['omega_p'][0]))
            omega_p_vals.append(disp['omega_p'][0])
        else:
            eta_vals.append(np.nan)
            omega_p_vals.append(np.nan)
    eta_vals = np.array(eta_vals)
    valid = ~np.isnan(eta_vals)
    axes[0].loglog(alpha_grid_lt[valid], eta_vals[valid], 'o-',
                    color=c, markersize=4, label=rf'$q={q_val}\,k_F$')

axes[0].axhline(0.1, color='blue', ls='--', alpha=0.5, label=r'coerente: $\eta<0.1$')
axes[0].axhline(1.0, color='red', ls='--', alpha=0.5, label=r'incoerente: $\eta>1$')
axes[0].set_xlabel(r'$\alpha$')
axes[0].set_ylabel(r'$\eta_p = \Gamma_p / (2 \omega_p)$')
axes[0].set_title(r'(a) Razão tempo de vida do PLASMON')
axes[0].legend(fontsize=9)

# (b) Tempo de vida da QUASIPARTÍCULA (elétron via GW)
# eta_e(omega) = Gamma_e(omega) / (2 omega)
omega_grid_qp = np.linspace(0.005, 0.3, 50)
for alpha, c in zip([0.1, 1.0, 10.0, 50.0],
                     plt.cm.viridis(np.linspace(0.15, 0.85, 4))):
    bath_a = BathParams(m0=1.0, k0=2.0, alpha=alpha)
    coup_a = CouplingParams(g=10.0)
    im_S_e = -im_sigma_e_GW(omega_grid_qp, bath_a, coup_a,
                             bath_mode='massive', n_q=50, n_omp=120)
    gamma_e = 2 * im_S_e
    eta_e = gamma_e / (2 * omega_grid_qp)
    mask = eta_e > 0
    axes[1].loglog(omega_grid_qp[mask], eta_e[mask], color=c,
                    label=rf'$\alpha={alpha}$', lw=2)

axes[1].axhline(0.1, color='blue', ls='--', alpha=0.5)
axes[1].axhline(1.0, color='red', ls='--', alpha=0.5)
axes[1].set_xlabel(r'$\omega$')
axes[1].set_ylabel(r'$\eta_e(\omega) = \Gamma_e / (2\omega)$')
axes[1].set_title(r'(b) Razão tempo de vida da QUASIPARTÍCULA')
axes[1].legend(fontsize=9)

fig.suptitle(r'Tempo de vida exibe as fases: coerente ($\eta<0.1$) → incoerente ($\eta>1$)',
             fontsize=13)
plt.tight_layout()
plt.show()
plt.savefig(f'{FIGDIR}/fig33_lifetime_phases.png', dpi=140, bbox_inches='tight')
plt.close()
print(f"    salvo: fig33_lifetime_phases.png")


# =============================================================
# FIG 34: Diagrama unificado — plasmon e quasipartícula
# =============================================================

print("[FIG 34] Diagrama unificado: regimes de plasmon e QP no plano (alpha, g)")

alpha_2d = np.logspace(-1.5, 2, 15)
g_2d = np.linspace(2, 20, 12)

# Plasmon eta_p em q fixo
eta_p_map = np.full((len(alpha_2d), len(g_2d)), np.nan)
# Expoente n do elétron via GW
n_e_map = np.full((len(alpha_2d), len(g_2d)), np.nan)

q_eval = 0.5  # ponto de avaliação do plasmon
print("    computando grade (pode demorar ~1min)...")

from core.electron_gas_GW import extract_exponent_GW

for i, alpha in enumerate(alpha_2d):
    for j, g in enumerate(g_2d):
        bath_ij = BathParams(m0=1.0, k0=2.0, alpha=alpha)
        coup_ij = CouplingParams(g=g)

        # Plasmon
        disp_ij = plasmon_dispersion(np.array([q_eval]), bath_ij, coup_ij,
                                       method='peak', n_omega=400)
        if disp_ij['found'][0] and not np.isnan(disp_ij['gamma_p'][0]):
            eta_p_map[i, j] = disp_ij['gamma_p'][0] / (2 * disp_ij['omega_p'][0])

        # Expoente eletrônico
        fit = extract_exponent_GW(bath_ij, coup_ij,
                                   window_fraction=(0.005, 0.05),
                                   bath_mode='massive', n_points=60)
        n_e_map[i, j] = fit['n']

    if i % 3 == 0:
        print(f"      linha {i+1}/{len(alpha_2d)}")

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

AG_2d, GG_2d = np.meshgrid(alpha_2d, g_2d, indexing='ij')

# (a) eta_p (plasmon)
im0 = axes[0].pcolormesh(AG_2d, GG_2d, eta_p_map,
                          norm=LogNorm(vmin=0.01, vmax=10),
                          shading='auto', cmap='RdYlBu_r')
plt.colorbar(im0, ax=axes[0], label=r'$\eta_p$ em $q=0.5\,k_F$')
cs = axes[0].contour(AG_2d, GG_2d, eta_p_map, levels=[0.1, 1.0],
                     colors='black', linewidths=1.5)
axes[0].clabel(cs, inline=True, fontsize=9, fmt={0.1:'η=0.1', 1.0:'η=1'})
axes[0].set_xscale('log')
axes[0].set_xlabel(r'$\alpha$'); axes[0].set_ylabel(r'$g$')
axes[0].set_title(r'(a) Plasmon: $\eta_p = \Gamma_p/(2\omega_p)$')

# (b) n_e (quasipartícula)
im1 = axes[1].pcolormesh(AG_2d, GG_2d, n_e_map, shading='auto',
                          cmap='RdYlBu', vmin=0.5, vmax=2.0)
plt.colorbar(im1, ax=axes[1], label=r'expoente $n$ em $|\mathrm{Im}\,\Sigma_e|\sim\omega^n$')
cs1 = axes[1].contour(AG_2d, GG_2d, n_e_map, levels=[1.2, 1.5, 1.8],
                       colors='black', linewidths=1.5)
axes[1].clabel(cs1, inline=True, fontsize=9)
axes[1].set_xscale('log')
axes[1].set_xlabel(r'$\alpha$'); axes[1].set_ylabel(r'$g$')
axes[1].set_title(r'(b) Quasipartícula: expoente $n$ via GW')

fig.suptitle(r'Correspondência: regime do plasmon ↔ regime da quasipartícula',
             fontsize=13)
plt.tight_layout()
plt.show()
plt.savefig(f'{FIGDIR}/fig34_unified_diagram.png', dpi=140, bbox_inches='tight')
plt.close()
print(f"    salvo: fig34_unified_diagram.png")


print("\nAnálise RPA completa concluída.")
print()
print("RESUMO:")
print("- Plasmon = pico de A_col acima do continuum p-h")
print("- Tempo de vida do plasmon: eta_p = Gamma_p / (2 omega_p)")
print("- Tempo de vida da QP: eta_e = Gamma_e / (2 omega) = 2|Im Sigma_e|/omega")
print("- Ambos transitam de < 0.1 (FL) a > 1 (incoerente)")
print("- Plasmon e QP exibem MESMA estrutura de fases (Fig. 34)")
