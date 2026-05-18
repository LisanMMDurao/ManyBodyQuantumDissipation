"""
analysis.08_plasmon_dispersion
==============================

Dispersão de plasmons via RPA com interação Coulomb (long-range).

A função dielétrica RPA é
    epsilon(omega, q) = 1 - V_q * chi_0(omega, q)

onde V_q é a interação bare. Para Coulomb 3D, V_q = 4*pi*e^2/q^2.
Plasmons são ZEROS de Re epsilon (ou polos de chi = chi_0/eps).

Em sistema livre, dispersão analítica:
    omega_p^2(q) = omega_p^2(0) + (3/5) v_F^2 q^2 + O(q^4)
com omega_p(0) = sqrt(4*pi*e^2*n/m) (frequência de plasma).

Em sistema acoplado ao banho (renormalização de Sigma_e),
o plasmon herda damping do banho:
    Gamma_plasmon(q) ~ -Im[1/eps] FWHM

Esta análise gera:
    1. Mapa A_col(omega, q) com plasmon visível acima do continuum
    2. Dispersão omega_p(q) em vários regimes do banho
    3. Largura do plasmon (tempo de vida do modo coletivo) vs alpha
    4. Verificação da relação canônica omega_p^2 = omega_p(0)^2 + (3/5) v_F^2 q^2
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import sys, os
sys.path.insert(0, '/home/lisan/Documentos/QuantumDissipation/qp_bath_project /qp_bath_project/')

from core.bath import BathParams
from core.electron_gas import CouplingParams
from core.collective_modes import (
    lindhard_3d, V_effective, dielectric_function,
    response_function, collective_spectral_function, loss_function,
    find_collective_mode, collective_dispersion, continuum_boundaries
)

FIGDIR = 'home/lisan/Documentos/QuantumDissipation/qp_bath_project /qp_bath_project/figures'

plt.rcParams.update({
    'font.size': 11, 'axes.labelsize': 12, 'legend.fontsize': 9,
    'lines.linewidth': 1.8, 'axes.grid': True, 'grid.alpha': 0.35,
})


# =============================================================
# FIG 29: Plasmon Coulomb canônico (referência sem banho)
# =============================================================

print("[FIG 29] Plasmon Coulomb canônico — referência")

# Configuração: usar form_factor 'long_range' (Coulomb)
# Sem mediação do banho (alpha bem pequeno), o sistema é gás livre + Coulomb
bath_ref = BathParams(m0=1.0, k0=5.0, alpha=0.001)  # banho quase inerte
coupling = CouplingParams(g=1.0)

omega = np.linspace(0.01, 3.0, 400)
q_grid = np.linspace(0.05, 1.5, 150)
OMG, QG = np.meshgrid(omega, q_grid, indexing='ij')

# Plasmon clássico: precisa Coulomb V_q = 4 pi/q^2
# Aqui usamos coupling.g como amplitude do Coulomb
A_col = collective_spectral_function(OMG, QG, bath_ref, coupling,
                                       form_factor='long_range')
loss = loss_function(OMG, QG, bath_ref, coupling, form_factor='long_range')

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# (a) A_col
A_safe = np.maximum(A_col, 1e-5)
im0 = axes[0].pcolormesh(QG, OMG, A_safe,
                          norm=LogNorm(vmin=1e-3, vmax=max(A_safe.max(), 0.01)),
                          shading='auto', cmap='inferno')
plt.colorbar(im0, ax=axes[0], label=r'$A_{\rm col}(\omega,q)$')

q_overlay = np.linspace(0.05, 1.5, 200)
om_low, om_high = continuum_boundaries(q_overlay, kF=coupling.kF, me=coupling.me)
axes[0].plot(q_overlay, om_high, 'w--', lw=1.5, alpha=0.7, label='borda sup. cont.')
axes[0].plot(q_overlay, om_low, 'w:', lw=1.5, alpha=0.7, label='borda inf. cont.')

# Dispersão analítica esperada: omega_p^2 = omega_p0^2 + (3/5) v_F^2 q^2
# onde omega_p0^2 = 4 pi e^2 n / m, com n = k_F^3 / (3 pi^2)
n = coupling.kF**3 / (3 * np.pi**2)
omega_p0_sq = 4 * np.pi * coupling.g**2 * n / coupling.me
vF = coupling.kF / coupling.me
omega_p_pred = np.sqrt(omega_p0_sq + (3/5) * vF**2 * q_overlay**2)
axes[0].plot(q_overlay, omega_p_pred, 'cyan', lw=2.5, alpha=0.8,
              label=r'$\omega_p^{\rm RPA}(q)$')

# Buscar plasmon numericamente
disp_data = collective_dispersion(q_overlay[::8], bath_ref, coupling,
                                    omega_max=3.0, form_factor='long_range')
valid = ~np.isnan(disp_data['omega_mode'])
axes[0].plot(disp_data['q_grid'][valid], disp_data['omega_mode'][valid],
              'o', color='yellow', markersize=4, label='pico (numérico)')

axes[0].set_xlabel(r'$q/k_F$'); axes[0].set_ylabel(r'$\omega$')
axes[0].set_title(r'(a) $A_{\rm col}$ com Coulomb: plasmon dispersa')
axes[0].legend(loc='upper left', fontsize=8)

# (b) -Im[1/eps] (loss function EELS — observable canônico para plasmons)
from matplotlib.colors import SymLogNorm
vmax = np.percentile(np.abs(loss), 99)
im1 = axes[1].pcolormesh(QG, OMG, loss,
                          norm=SymLogNorm(linthresh=1e-3, vmin=-vmax, vmax=vmax),
                          shading='auto', cmap='RdBu_r')
plt.colorbar(im1, ax=axes[1], label=r'$-\mathrm{Im}[1/\varepsilon]$')
axes[1].plot(q_overlay, omega_p_pred, 'k-', lw=2, alpha=0.7,
              label=r'$\omega_p^{\rm RPA}$')
axes[1].plot(q_overlay, om_high, 'k--', lw=1, alpha=0.5)
axes[1].set_xlabel(r'$q/k_F$'); axes[1].set_ylabel(r'$\omega$')
axes[1].set_title(r'(b) Loss function: pico = plasmon (EELS)')
axes[1].legend(loc='upper left', fontsize=9)

fig.suptitle(r'Plasmon RPA com banho quase inerte ($\alpha=10^{-3}$)', fontsize=13)
plt.tight_layout()
plt.show()
plt.savefig(f'{FIGDIR}/fig29_plasmon_canonical.png', dpi=140, bbox_inches='tight')
plt.close()
print(f"    salvo: fig29_plasmon_canonical.png")


# =============================================================
# FIG 30: Verificação da dispersão omega_p(q) — comparação numérico vs analítico
# =============================================================

print("[FIG 30] Verificação da dispersão do plasmon")

q_disp = np.linspace(0.05, 1.0, 25)
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

# (a) omega_p(q) — varredura em g (intensidade Coulomb)
g_vals = [0.5, 1.0, 1.5, 2.0]
for g, c in zip(g_vals, plt.cm.viridis(np.linspace(0.15, 0.85, len(g_vals)))):
    coup = CouplingParams(g=g)
    omega_p0_sq = 4 * np.pi * g**2 * n / coup.me
    omega_p_pred = np.sqrt(omega_p0_sq + (3/5) * vF**2 * q_disp**2)
    axes[0].plot(q_disp, omega_p_pred, color=c, ls='--', lw=1.2, alpha=0.6,
                  label=rf'RPA analítico, $g={g}$')

    # numérico
    disp = collective_dispersion(q_disp, bath_ref, coup,
                                  omega_max=4.0, form_factor='long_range')
    valid = ~np.isnan(disp['omega_mode'])
    axes[0].plot(disp['q_grid'][valid], disp['omega_mode'][valid], 'o-',
                  color=c, markersize=4, lw=1.5, label=rf'numérico, $g={g}$')

axes[0].set_xlabel(r'$q/k_F$'); axes[0].set_ylabel(r'$\omega_p(q)$')
axes[0].set_title(r'(a) Dispersão: analítico vs numérico')
axes[0].legend(fontsize=7, ncol=2)

# (b) omega_p^2 versus q^2 — esperado linear com inclinação (3/5) v_F^2
g = 1.0
coup = CouplingParams(g=g)
disp = collective_dispersion(q_disp, bath_ref, coup,
                              omega_max=4.0, form_factor='long_range')
valid = ~np.isnan(disp['omega_mode'])
qq = disp['q_grid'][valid]
oo = disp['omega_mode'][valid]
axes[1].plot(qq**2, oo**2, 'o', color='tab:blue', markersize=5,
              label='numérico')

# fit linear
coef = np.polyfit(qq**2, oo**2, 1)
slope_num, intercept_num = coef
omega_p0_sq_pred = 4 * np.pi * g**2 * n / coup.me
slope_pred = (3/5) * vF**2

q2_line = np.linspace(0, qq[-1]**2 * 1.1, 100)
axes[1].plot(q2_line, slope_pred * q2_line + omega_p0_sq_pred, 'r--', lw=2,
              label=rf'analítico: $\omega_p^2={omega_p0_sq_pred:.3f}+{slope_pred:.3f}q^2$')
axes[1].plot(q2_line, slope_num * q2_line + intercept_num, 'g:', lw=2,
              label=rf'fit: $\omega_p^2={intercept_num:.3f}+{slope_num:.3f}q^2$')

axes[1].set_xlabel(r'$q^2/k_F^2$'); axes[1].set_ylabel(r'$\omega_p^2(q)$')
axes[1].set_title(r'(b) Linearidade em $q^2$: confirma RPA')
axes[1].legend(fontsize=9)

plt.tight_layout()
plt.show()
plt.savefig(f'{FIGDIR}/fig30_dispersion_verification.png', dpi=140, bbox_inches='tight')
plt.close()
print(f"    salvo: fig30_dispersion_verification.png")
print(f"    omega_p^2(0): analítico = {omega_p0_sq_pred:.4f}, numérico = {intercept_num:.4f}")
print(f"    slope: analítico = {slope_pred:.4f}, numérico = {slope_num:.4f}")


# =============================================================
# FIG 31: Damping do plasmon por banho — degradação com alpha
# =============================================================

print("[FIG 31] Damping do plasmon vs alpha do banho")

q_fixed = 0.3  # momentum pequeno (plasmon bem definido)
alpha_grid = np.logspace(-3, 1, 25)

fwhm_list = []
omega_p_list = []
amp_list = []

for alpha in alpha_grid:
    bath_a = BathParams(m0=1.0, k0=5.0, alpha=alpha)
    coup_a = CouplingParams(g=1.0)
    mode = find_collective_mode(q_fixed, bath_a, coup_a,
                                  omega_range=(0.05, 4.0),
                                  form_factor='long_range')
    fwhm_list.append(mode['fwhm'])
    omega_p_list.append(mode['omega_mode'])
    amp_list.append(mode['amplitude'])

fwhm_arr = np.array(fwhm_list)
omega_p_arr = np.array(omega_p_list)
amp_arr = np.array(amp_list)

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

# (a) Frequência e largura vs alpha
ax1 = axes[0]
ax2 = ax1.twinx()
valid = ~np.isnan(omega_p_arr)
ax1.semilogx(alpha_grid[valid], omega_p_arr[valid], 'b-o', markersize=4,
              label=r'$\omega_p(q=0.3)$')
valid_fw = ~np.isnan(fwhm_arr)
ax2.semilogx(alpha_grid[valid_fw], fwhm_arr[valid_fw], 'r-s', markersize=4,
              label=r'FWHM')

ax1.set_xlabel(r'$\alpha$ (banho)')
ax1.set_ylabel(r'$\omega_p$', color='b')
ax2.set_ylabel(r'FWHM (largura)', color='r')
ax1.tick_params(axis='y', labelcolor='b')
ax2.tick_params(axis='y', labelcolor='r')
ax1.set_title(rf'(a) Plasmon em $q={q_fixed}\,k_F$: $\omega_p$ e largura vs $\alpha$')
ax1.legend(loc='upper left'); ax2.legend(loc='upper right')

# (b) Q-factor: omega_p / FWHM (qualidade do modo)
ax_b = axes[1]
Q_factor = omega_p_arr / fwhm_arr
valid_Q = ~np.isnan(Q_factor) & (Q_factor > 0)
ax_b.loglog(alpha_grid[valid_Q], Q_factor[valid_Q], 'g-o', markersize=4)
ax_b.axhline(1, color='k', ls='--', alpha=0.5, label=r'$Q=1$ (fronteira)')
ax_b.axhline(10, color='gray', ls=':', alpha=0.5, label=r'$Q=10$')
ax_b.set_xlabel(r'$\alpha$')
ax_b.set_ylabel(r'$Q = \omega_p/\Gamma$ (qualidade do plasmon)')
ax_b.set_title(r'(b) Q-factor: $\alpha$ pequeno = plasmon de alta qualidade')
ax_b.legend()

plt.tight_layout()
plt.show()
plt.savefig(f'{FIGDIR}/fig31_plasmon_damping.png', dpi=140, bbox_inches='tight')
plt.close()
print(f"    salvo: fig31_plasmon_damping.png")


# =============================================================
# FIG 32: Modos coletivos em 3 regimes do banho — comparação
# =============================================================

print("[FIG 32] Modos coletivos em 3 regimes")

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

regimes = [
    (r'(a) Banho fraco $\alpha=0.01$', 0.01, '$Q\\gg 1$, plasmon bem definido'),
    (r'(b) Banho médio $\alpha=0.5$',  0.5,  '$Q\\sim$ moderado, plasmon largo'),
    (r'(c) Banho forte $\alpha=5.0$',  5.0,  '$Q<1$, plasmon dissolvido'),
]

omega_3 = np.linspace(0.01, 3.0, 400)
q_3 = np.linspace(0.05, 1.5, 100)
OMG_3, QG_3 = np.meshgrid(omega_3, q_3, indexing='ij')

for ax, (label, alpha, ann) in zip(axes, regimes):
    bath_r = BathParams(m0=1.0, k0=5.0, alpha=alpha)
    coup_r = CouplingParams(g=1.0)
    A_r = collective_spectral_function(OMG_3, QG_3, bath_r, coup_r,
                                        form_factor='long_range')
    A_safe = np.maximum(A_r, 1e-5)
    im = ax.pcolormesh(QG_3, OMG_3, A_safe,
                        norm=LogNorm(vmin=1e-3, vmax=max(A_safe.max(), 1e-2)),
                        shading='auto', cmap='inferno')
    plt.colorbar(im, ax=ax, label=r'$A_{\rm col}$')

    q_o = np.linspace(0.05, 1.5, 200)
    o_low_r, o_high_r = continuum_boundaries(q_o)
    ax.plot(q_o, o_high_r, 'w--', lw=1, alpha=0.6)
    ax.plot(q_o, o_low_r, 'w:', lw=1, alpha=0.6)

    # plasmon analítico (referência)
    om_p_ref = np.sqrt(omega_p0_sq + (3/5) * vF**2 * q_o**2)
    ax.plot(q_o, om_p_ref, 'cyan', lw=1.5, alpha=0.7,
             label=r'$\omega_p$ (sem dressing)')

    ax.set_xlabel(r'$q/k_F$'); ax.set_ylabel(r'$\omega$')
    ax.set_title(label)
    ax.text(0.05, 0.95, ann, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', color='white',
            bbox=dict(facecolor='black', alpha=0.6))
    ax.legend(loc='lower right', fontsize=8)

fig.suptitle(r'Crossover do plasmon nos três regimes do banho (Coulomb $g=1$)',
             fontsize=13)
plt.tight_layout()
plt.show()
plt.savefig(f'{FIGDIR}/fig32_plasmon_regimes.png', dpi=140, bbox_inches='tight')
plt.close()
print(f"    salvo: fig32_plasmon_regimes.png")


print("\nAnálise da dispersão de plasmons concluída.")
