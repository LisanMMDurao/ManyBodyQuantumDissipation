"""
analysis.06_fit_and_correspondence
==================================

Duas análises complementares:

A) FIT DA LEI DE POTÊNCIA (analogia ao código original):
   Extrai o expoente n em |Im Sigma_e^R(omega)| ~ omega^n por ajuste log-log.
   Na Convenção A, n = 1 universalmente — o que distingue regimes é o
   PRÉ-FATOR A do fit (que coincide com lambda).

B) CORRESPONDÊNCIA BANHO <-> GÁS DE ELÉTRONS:
   Identifica como os regimes do banho (caracterizados por alpha, E_0)
   se mapeiam nos regimes do gás (caracterizados por lambda, omega_coh).

   A relação chave é:
        Banho fraco (alpha pequeno) AND/OR g pequeno AND/OR F pequeno
        => lambda pequeno => FL coerente

        Banho forte AND g forte AND F ~ 1
        => lambda grande => incoerente

   O fato de F entrar SOMENTE no gás (não no banho) introduz uma
   ASSIMETRIA: pode-se ter banho fortemente dissipativo isoladamente
   (alpha=1) mas gás de elétrons FL (g pequeno ou F pequeno).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, ListedColormap
import matplotlib.patches as mpatches
import sys, os
sys.path.insert(0, '/home/lisan/Documentos/QuantumDissipation/qp_bath_project /qp_bath_project/')

from core.bath import BathParams, im_sigma_bath
from core.electron_gas import (
    CouplingParams, im_sigma_e, momentum_cutoff_factor
)
from core.fit_analysis import (
    power_law_fit, extract_n_for_params, classify_by_exponent,
    classify_by_amplitude, scan_exponent_alpha_g,
    temperature_proxy_resistivity
)
from core.phase_analysis import dimensionless_coupling, classify_phase

FIGDIR = 'home/lisan/Documentos/QuantumDissipation/qp_bath_project /qp_bath_project/figures'
os.makedirs(FIGDIR, exist_ok=True)

plt.rcParams.update({
    'font.size': 11, 'axes.labelsize': 12, 'legend.fontsize': 9,
    'lines.linewidth': 1.8, 'axes.grid': True, 'grid.alpha': 0.35,
})


# =============================================================
# PARTE A: FIT DA LEI DE POTÊNCIA
# =============================================================

# FIG 20: Demonstração do fit
print("[FIG 20] Demonstração do ajuste de lei de potência")

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

cases = [
    ('FL: $\\lambda \\approx 0.02$', BathParams(m0=1, k0=1, alpha=0.05), CouplingParams(g=1.0), 'tab:blue'),
    ('MFL: $\\lambda \\approx 0.4$', BathParams(m0=1, k0=1, alpha=0.5), CouplingParams(g=2.5), 'tab:orange'),
    ('Incoh.: $\\lambda \\approx 1.6$', BathParams(m0=1, k0=1, alpha=2.0), CouplingParams(g=3.5), 'tab:red'),
]

omega_full = np.logspace(-4, 1, 300)
omega_window = (0.01, 0.15)  # janela de fit (em unidades de E_0 = 0.5)

for label, bath, coup, color in cases:
    im_S = -im_sigma_e(omega_full, bath, coup)
    axes[0].loglog(omega_full/bath.E0, im_S, color=color, label=label, lw=2)

    # fit
    fit = power_law_fit(omega_full, im_S, omega_window)
    lam = dimensionless_coupling(bath, coup)
    om_fit = np.linspace(omega_window[0], omega_window[1], 20)
    fit_curve = fit['A'] * om_fit**fit['n']
    axes[0].loglog(om_fit/bath.E0, fit_curve, color=color, ls='--', lw=1.5, alpha=0.7,
                   label=rf'fit: $n={fit["n"]:.2f}$, $A={fit["A"]:.3f}$ ($\lambda={lam:.3f}$)')

axes[0].axvspan(omega_window[0]/0.5, omega_window[1]/0.5, alpha=0.15, color='yellow',
                label='janela de fit')
axes[0].set_xlabel(r'$\omega/E_0$')
axes[0].set_ylabel(r'$|\mathrm{Im}\,\Sigma_e^R|$')
axes[0].set_title(r'Ajuste log-log: $|\mathrm{Im}\,\Sigma| \sim A\omega^n$')
axes[0].legend(fontsize=7, loc='lower right')

# (b) Pré-fator A vs lambda — correlação direta
print("[FIG 20b] Correlação A_fit vs lambda")

# Varredura ampla
A_vals = []
lambda_vals = []
n_vals = []
for alpha in np.logspace(-2, 1, 20):
    for g in np.linspace(0.5, 5, 10):
        bath_s = BathParams(m0=1, k0=1, alpha=alpha)
        coup_s = CouplingParams(g=g)
        lam = dimensionless_coupling(bath_s, coup_s)
        omega_t = np.linspace(0.005, 0.15, 200)
        im_S = -im_sigma_e(omega_t, bath_s, coup_s)
        fit = power_law_fit(omega_t, im_S, (0.01, 0.12))
        if not np.isnan(fit['n']):
            A_vals.append(fit['A'])
            lambda_vals.append(lam)
            n_vals.append(fit['n'])

A_vals = np.array(A_vals)
lambda_vals = np.array(lambda_vals)
n_vals = np.array(n_vals)

sc = axes[1].scatter(lambda_vals, A_vals, c=n_vals, cmap='RdYlBu',
                      vmin=0.7, vmax=1.1, s=15, alpha=0.7)
axes[1].plot([1e-4, 100], [1e-4, 100], 'k--', alpha=0.5, label=r'$A = \lambda$')
plt.colorbar(sc, ax=axes[1], label=r'$n$ (expoente)')
axes[1].set_xscale('log'); axes[1].set_yscale('log')
axes[1].set_xlabel(r'$\lambda$ (analítico)')
axes[1].set_ylabel(r'$A_{\rm fit}$ (numérico)')
axes[1].set_title(r'Pré-fator $A_{\rm fit}$ vs. $\lambda$ analítico')
axes[1].legend()

plt.tight_layout()
plt.show()
plt.savefig(f'{FIGDIR}/fig20_power_law_fit.png', dpi=140, bbox_inches='tight')
plt.close()
print(f"    salvo: fig20_power_law_fit.png")


# =============================================================
# FIG 21: Mapa do expoente n no plano (alpha, g)
# =============================================================

print("[FIG 21] Mapa do expoente n e pré-fator A no plano (alpha, g)")

alpha_grid = np.logspace(-2, 1, 25)
g_grid = np.linspace(0.3, 5.0, 25)

# Computar n_map e A_map manualmente para ter A também
n_map = np.zeros((len(alpha_grid), len(g_grid)))
A_map = np.zeros((len(alpha_grid), len(g_grid)))
lam_map = np.zeros((len(alpha_grid), len(g_grid)))

for i, alpha in enumerate(alpha_grid):
    bath_i = BathParams(m0=1, k0=1, alpha=alpha)
    for j, g in enumerate(g_grid):
        coup_j = CouplingParams(g=g)
        lam = dimensionless_coupling(bath_i, coup_j)
        omega_t = np.linspace(0.005, 0.15, 150)
        im_S = -im_sigma_e(omega_t, bath_i, coup_j)
        fit = power_law_fit(omega_t, im_S, (0.01, 0.12))
        n_map[i, j] = fit['n']
        A_map[i, j] = fit['A']
        lam_map[i, j] = lam

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

AG, GG = np.meshgrid(alpha_grid, g_grid, indexing='ij')

# (a) n_map — quase uniforme
im0 = axes[0].pcolormesh(AG, GG, n_map, shading='auto',
                          cmap='RdYlBu', vmin=0.7, vmax=1.2)
plt.colorbar(im0, ax=axes[0], label=r'$n$ (expoente)')
axes[0].set_xscale('log')
axes[0].set_xlabel(r'$\alpha$'); axes[0].set_ylabel(r'$g$')
axes[0].set_title(r'(a) Expoente $n$: quase uniforme $\sim 1$')

# (b) A_map — varia muito
im1 = axes[1].pcolormesh(AG, GG, A_map, shading='auto',
                          norm=LogNorm(vmin=1e-3, vmax=10), cmap='viridis')
plt.colorbar(im1, ax=axes[1], label=r'$A_{\rm fit}$')
axes[1].set_xscale('log')
axes[1].set_xlabel(r'$\alpha$'); axes[1].set_ylabel(r'$g$')
axes[1].set_title(r'(b) Pré-fator $A_{\rm fit}$ (cobre 4 ordens)')

# Contornos de A = lambda crítico
cs = axes[1].contour(AG, GG, A_map, levels=[0.1, 1.0],
                     colors='white', linewidths=2, alpha=0.85)
axes[1].clabel(cs, inline=True, fontsize=9)

# (c) Razão A/lambda — verificação
ratio = A_map / lam_map
im2 = axes[2].pcolormesh(AG, GG, ratio, shading='auto',
                          vmin=0.5, vmax=1.5, cmap='RdBu_r')
plt.colorbar(im2, ax=axes[2], label=r'$A_{\rm fit}/\lambda$')
axes[2].set_xscale('log')
axes[2].set_xlabel(r'$\alpha$'); axes[2].set_ylabel(r'$g$')
axes[2].set_title(r'(c) Razão $A_{\rm fit}/\lambda$ (deve ser $\sim 1$)')

fig.suptitle(r'Fit log-log no plano $(\alpha, g)$: $n$ uniforme, $A$ rastreia $\lambda$',
             fontsize=13)
plt.tight_layout()
plt.show()
plt.savefig(f'{FIGDIR}/fig21_n_A_maps.png', dpi=140, bbox_inches='tight')
plt.close()
print(f"    salvo: fig21_n_A_maps.png")


# =============================================================
# FIG 22: Resistividade proxy rho(T)
# =============================================================

print("[FIG 22] Proxy térmico: rho(T) ~ |Im Sigma(omega = T)|")

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

T_grid = np.linspace(0.01, 0.3, 30)

# (a) rho vs T para 3 regimes
cases_rho = [
    ('FL',     BathParams(m0=1, k0=1, alpha=0.05), CouplingParams(g=1.0), 'tab:blue'),
    ('MFL',    BathParams(m0=1, k0=1, alpha=0.5),  CouplingParams(g=2.5), 'tab:orange'),
    ('Incoh.', BathParams(m0=1, k0=1, alpha=2.0),  CouplingParams(g=3.5), 'tab:red'),
]

for label, bath, coup, color in cases_rho:
    res = temperature_proxy_resistivity(T_grid, bath, coup)
    axes[0].plot(T_grid, res['rho_T'], 'o-', color=color, markersize=4,
                  label=rf'{label}: $p={res["exponent_p"]:.2f}$')

axes[0].set_xlabel(r'$T$ (proxy: $\omega = T$)')
axes[0].set_ylabel(r'$\rho \propto |\mathrm{Im}\,\Sigma_e^R(T)|$')
axes[0].set_title(r'(a) Resistividade proxy: $\rho \sim T^p$')
axes[0].legend(fontsize=9)

# (b) log-log
for label, bath, coup, color in cases_rho:
    res = temperature_proxy_resistivity(T_grid, bath, coup)
    mask = res['rho_T'] > 0
    axes[1].loglog(T_grid[mask], res['rho_T'][mask], 'o-',
                   color=color, markersize=4,
                   label=rf'{label}: $p={res["exponent_p"]:.2f}$')

# referências
T_ref = T_grid[T_grid > 0.02]
axes[1].loglog(T_ref, T_ref * 0.5, 'k:', alpha=0.5, label=r'$\sim T$')
axes[1].loglog(T_ref, T_ref**2 * 5, 'k--', alpha=0.5, label=r'$\sim T^2$')

axes[1].set_xlabel(r'$T$')
axes[1].set_ylabel(r'$\rho(T)$')
axes[1].set_title(r'(b) Log-log: na Convenção A, todos $p \sim 1$')
axes[1].legend(fontsize=9)

plt.tight_layout()
plt.show()
plt.savefig(f'{FIGDIR}/fig22_resistivity_proxy.png', dpi=140, bbox_inches='tight')
plt.close()
print(f"    salvo: fig22_resistivity_proxy.png")


# =============================================================
# PARTE B: CORRESPONDÊNCIA BANHO <-> GÁS DE ELÉTRONS
# =============================================================

# FIG 23: Mapa de correspondência (regime do banho vs regime do gás)
print("[FIG 23] Correspondência banho <-> gás")

fig = plt.figure(figsize=(15, 11))
gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.35)

# (a) Função espectral DO BANHO em três regimes de alpha
ax_a = fig.add_subplot(gs[0, 0])
omega_b = np.linspace(-3, 3, 400)
for alpha, c in zip([0.05, 0.5, 2.0],
                     plt.cm.viridis(np.linspace(0.15, 0.85, 3))):
    p = BathParams(m0=1, k0=1, alpha=alpha)
    ax_a.plot(omega_b, -im_sigma_bath(omega_b, p), color=c,
              label=rf'$\alpha={alpha}$')
ax_a.set_xlabel(r'$\omega$')
ax_a.set_ylabel(r'$|\mathrm{Im}\,\Sigma_{\rm bath}|$')
ax_a.set_title(r'(a) Banho isolado: regime de $\alpha$')
ax_a.legend(fontsize=9)

# (b) Função espectral DO GÁS em mesmos regimes, mas com g=1
ax_b = fig.add_subplot(gs[0, 1])
for alpha, c in zip([0.05, 0.5, 2.0],
                     plt.cm.viridis(np.linspace(0.15, 0.85, 3))):
    bath_b = BathParams(m0=1, k0=1, alpha=alpha)
    coup_b = CouplingParams(g=1.0)
    lam = dimensionless_coupling(bath_b, coup_b)
    ax_b.plot(omega_b, -im_sigma_e(omega_b, bath_b, coup_b), color=c,
              label=rf'$\alpha={alpha}$ ($\lambda={lam:.3f}$)')
ax_b.set_xlabel(r'$\omega$')
ax_b.set_ylabel(r'$|\mathrm{Im}\,\Sigma_e|$')
ax_b.set_title(r'(b) Gás com $g=1$: regime no GÁS depende de $\lambda$')
ax_b.legend(fontsize=8)

# (c) Função espectral DO GÁS com g grande - "amplifica" o banho
ax_c = fig.add_subplot(gs[0, 2])
for alpha, c in zip([0.05, 0.5, 2.0],
                     plt.cm.viridis(np.linspace(0.15, 0.85, 3))):
    bath_b = BathParams(m0=1, k0=1, alpha=alpha)
    coup_b = CouplingParams(g=4.0)
    lam = dimensionless_coupling(bath_b, coup_b)
    ax_c.plot(omega_b, -im_sigma_e(omega_b, bath_b, coup_b), color=c,
              label=rf'$\alpha={alpha}$ ($\lambda={lam:.2f}$)')
ax_c.set_xlabel(r'$\omega$')
ax_c.set_ylabel(r'$|\mathrm{Im}\,\Sigma_e|$')
ax_c.set_title(r'(c) Gás com $g=4$: amplificação por $g^2 N(0) F$')
ax_c.legend(fontsize=8)

# (d, e, f) Mapas: alpha (banho) vs g (gás)
# (d) alpha do banho
ax_d = fig.add_subplot(gs[1, 0])
alpha_b_grid = np.logspace(-2, 1, 100)
g_b_grid = np.linspace(0.1, 5, 100)
AG_b, GG_b = np.meshgrid(alpha_b_grid, g_b_grid, indexing='ij')

# Im Sigma_bath em omega=0.1 -- escala do banho isolado
im_bath_at_01 = np.pi * AG_b * 0.1 * np.exp(-0.1/0.5)
im_b_plot = im_bath_at_01 * np.ones_like(GG_b)  # não depende de g
imd = ax_d.pcolormesh(AG_b, GG_b, im_b_plot, norm=LogNorm(vmin=1e-3, vmax=10),
                       shading='auto', cmap='Blues')
plt.colorbar(imd, ax=ax_d, label=r'$|\mathrm{Im}\,\Sigma_{\rm bath}(0.1)|$')
ax_d.set_xscale('log')
ax_d.set_xlabel(r'$\alpha$'); ax_d.set_ylabel(r'$g$')
ax_d.set_title(r'(d) Banho isolado: depende SÓ de $\alpha$')

# (e) Im Sigma_e em omega = 0.1 (gás dependente de ambos)
ax_e = fig.add_subplot(gs[1, 1])
# Pré-fator: g^2 N(0) F = g^2 * 0.05066 * 0.25 = 0.01266 g^2 (com m0=1, k0=1)
prefactor = GG_b**2 * (1.0/(2*np.pi**2)) * 0.25
im_e_full = prefactor * im_bath_at_01
ime = ax_e.pcolormesh(AG_b, GG_b, im_e_full, norm=LogNorm(vmin=1e-5, vmax=1),
                       shading='auto', cmap='Reds')
plt.colorbar(ime, ax=ax_e, label=r'$|\mathrm{Im}\,\Sigma_e(0.1)|$')
ax_e.set_xscale('log')
ax_e.set_xlabel(r'$\alpha$'); ax_e.set_ylabel(r'$g$')
ax_e.set_title(r'(e) Gás: $\propto g^2 \cdot \alpha$ (separável)')

# Contornos lambda
import matplotlib.patches as mpatches
lam_full = np.pi * AG_b * GG_b**2 * (1.0/(2*np.pi**2)) * 0.25
cs = ax_e.contour(AG_b, GG_b, lam_full, levels=[0.1, 1.0],
                   colors='white', linewidths=2, alpha=0.85)
ax_e.clabel(cs, inline=True, fontsize=9, fmt={0.1:'λ=0.1', 1.0:'λ=1'})

# (f) Diagrama esquemático
ax_f = fig.add_subplot(gs[1, 2])
ax_f.axis('off')

text = (
    r'$\mathbf{Correspondência\ banho \leftrightarrow gás:}$' + '\n\n'
    r'Banho isolado controlado por: $\alpha$' + '\n'
    r'Im $\Sigma_{\rm bath} = -\pi\alpha|\omega|e^{-|\omega|/E_0}$' + '\n\n'
    r'Gás de elétrons controlado por: $\lambda$' + '\n'
    r'$\lambda = \pi\alpha\,g^2\,N(0)\,F(k_0/k_F)$' + '\n'
    r'Im $\Sigma_e = -\lambda|\omega|e^{-|\omega|/E_0}$' + '\n\n'
    r'$\mathbf{Pontes:}$' + '\n'
    r'• $g$: vértice elétron-mediador' + '\n'
    r'• $N(0)$: dispersão fermi' + '\n'
    r'• $F(k_0/k_F)$: espaço de fase' + '\n\n'
    r'$\mathbf{Cenários:}$' + '\n'
    r'• $\alpha$ grande, $g$ pequeno → banho dissipa' + '\n'
    r'  isoladamente, mas gás é FL' + '\n'
    r'• $\alpha$ pequeno, $g$ grande → banho calmo' + '\n'
    r'  mas gás é amplificador → MFL/incoh.'
)
ax_f.text(0.02, 0.97, text, transform=ax_f.transAxes,
          verticalalignment='top', fontsize=10,
          bbox=dict(boxstyle='round', facecolor='#f5f5f5', alpha=0.9))

fig.suptitle(r'Correspondência: regimes do banho vs regimes do gás',
             fontsize=14, fontweight='bold')
plt.show()
plt.savefig(f'{FIGDIR}/fig23_correspondence.png', dpi=140, bbox_inches='tight')
plt.close()
print(f"    salvo: fig23_correspondence.png")


# =============================================================
# FIG 24: Tabela visual de fases nos planos
# =============================================================

print("[FIG 24] Tabela visual: alpha do banho vs lambda do gás")

# Plano alpha (banho) vs g (gás) - mostrando regime do GÁS, contornos do BANHO

fig, ax = plt.subplots(figsize=(10, 7))

phase_cmap = ListedColormap(['#3b73af', '#e8a13a', '#c34440'])

# Categorizar gás
phase_gas = np.zeros_like(lam_full, dtype=int)
phase_gas[lam_full >= 0.1] = 1
phase_gas[lam_full >= 1.0] = 2

im = ax.pcolormesh(AG_b, GG_b, phase_gas, shading='auto',
                    cmap=phase_cmap, vmin=-0.5, vmax=2.5)

# Contornos do BANHO: alpha thresholds
# "Banho fraco": alpha < 0.1; "Banho forte": alpha > 1
ax.axvline(0.1, color='cyan', ls='--', lw=2, label=r'banho fraco | médio')
ax.axvline(1.0, color='lime', ls='--', lw=2, label=r'banho médio | forte')

# Anotações nos quadrantes
ax.annotate('Banho FRACO,\nGás FL', xy=(0.03, 4),
            fontsize=11, color='white', ha='center', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
ax.annotate('Banho FORTE,\nmas Gás FL\n(g pequeno)', xy=(3.0, 0.7),
            fontsize=11, color='white', ha='center', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
ax.annotate('Banho FRACO,\nGás INCOH\n(g grande)', xy=(0.03, 0.7),
            fontsize=10, color='white', ha='center', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
ax.annotate('Ambos FORTES:\nGás claramente\nincoerente', xy=(3.0, 4),
            fontsize=11, color='white', ha='center', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))

ax.set_xscale('log')
ax.set_xlabel(r'$\alpha$ (regime do banho)', fontsize=13)
ax.set_ylabel(r'$g$ (acoplamento sistema-banho)', fontsize=13)
ax.set_title(r'Fases do GÁS no plano $(\alpha, g)$ — fronteiras do BANHO em ciano/lima',
             fontsize=12)

patches = [mpatches.Patch(color='#3b73af', label='FL (gás)'),
           mpatches.Patch(color='#e8a13a', label='MFL (gás)'),
           mpatches.Patch(color='#c34440', label='Incoh. (gás)')]
ax.legend(handles=patches, loc='lower right')
plt.tight_layout()
plt.show()
plt.savefig(f'{FIGDIR}/fig24_phase_table.png', dpi=140, bbox_inches='tight')
plt.close()
print(f"    salvo: fig24_phase_table.png")


print("\nAnálise de fit e correspondência concluída.")
