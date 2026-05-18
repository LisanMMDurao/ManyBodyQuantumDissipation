"""
analysis.02_electron_gas
========================

Estudo do gás de elétrons não-interagentes acoplado ao banho de quasipartículas.

Investigamos como a estrutura espectral do elétron muda em função de:
    - alpha: acoplamento sistema-banho
    - g: acoplamento elétron-mediador
    - k_0/k_F: cutoff em momentum (forma de F)
    - m_0/m_e: massa do banho (afeta E_0)
"""

import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, '/home/lisan/Documentos/QuantumDissipation/qp_bath_project /qp_bath_project/')

from core.bath import BathParams
from core.electron_gas import (
    CouplingParams, im_sigma_e, re_sigma_e, sigma_e,
    spectral_function, dos, momentum_cutoff_factor,
    coherence_window
)

FIGDIR = 'home/lisan/Documentos/QuantumDissipation/qp_bath_project /qp_bath_project/figures'
os.makedirs(FIGDIR, exist_ok=True)

plt.rcParams.update({
    'font.size': 11, 'axes.labelsize': 12, 'legend.fontsize': 9,
    'lines.linewidth': 1.8, 'axes.grid': True, 'grid.alpha': 0.35,
})


# =============================================================
# FIG 5: Função espectral A(k=k_F, omega) para várias intensidades
# =============================================================

print("[FIG 5] Função espectral na superfície de Fermi")

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
omega = np.linspace(-2, 2, 800)

# (a) variando alpha com g, k0, m0 fixos
alphas = [0.05, 0.2, 0.5, 1.5, 3.0]
for alpha, c in zip(alphas, plt.cm.viridis(np.linspace(0.15, 0.85, len(alphas)))):
    bath = BathParams(m0=1.0, k0=1.0, alpha=alpha)
    coupling = CouplingParams(g=2.0)
    A = spectral_function(omega, 0.0, bath, coupling, broadening=5e-3)
    axes[0].plot(omega, A, color=c, label=rf'$\alpha={alpha}$')

axes[0].set_xlabel(r'$\omega$ (rel. a $\mu$)')
axes[0].set_ylabel(r'$A(k_F, \omega)$')
axes[0].set_title(r'(a) Variando $\alpha$: $g=2$, $m_0=m_e$, $k_0=k_F$')
axes[0].legend()
axes[0].axvline(0, color='k', lw=0.5)

# (b) variando g
gs = [0.5, 1.0, 2.0, 3.0, 5.0]
for g, c in zip(gs, plt.cm.plasma(np.linspace(0.15, 0.85, len(gs)))):
    bath = BathParams(m0=1.0, k0=1.0, alpha=0.5)
    coupling = CouplingParams(g=g)
    A = spectral_function(omega, 0.0, bath, coupling, broadening=5e-3)
    axes[1].plot(omega, A, color=c, label=rf'$g={g}$')

axes[1].set_xlabel(r'$\omega$ (rel. a $\mu$)')
axes[1].set_ylabel(r'$A(k_F, \omega)$')
axes[1].set_title(r'(b) Variando $g$: $\alpha=0.5$, $m_0=m_e$, $k_0=k_F$')
axes[1].legend()
axes[1].axvline(0, color='k', lw=0.5)

fig.suptitle('Função espectral eletrônica na superfície de Fermi', fontsize=13)
plt.tight_layout()
plt.show()
plt.savefig(f'{FIGDIR}/fig05_spectral_function_FS.png', dpi=140, bbox_inches='tight')
plt.close()
print(f"    salvo: fig05_spectral_function_FS.png")


# =============================================================
# FIG 6: Mapa A(k, omega) — dispersão renormalizada
# =============================================================

print("[FIG 6] Mapa A(k, omega) — espectro completo")

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

cases = [
    ('Fraco: $\\alpha=0.1$, $g=1$', BathParams(m0=1, k0=1, alpha=0.1), CouplingParams(g=1.0)),
    ('Intermediário: $\\alpha=0.5$, $g=2$', BathParams(m0=1, k0=1, alpha=0.5), CouplingParams(g=2.0)),
    ('Forte: $\\alpha=1.0$, $g=3$', BathParams(m0=1, k0=1, alpha=1.0), CouplingParams(g=3.0)),
]

k_vals = np.linspace(0.1, 1.8, 80)
omega_grid = np.linspace(-1.5, 2.5, 200)

for ax, (label, bath, coupling) in zip(axes, cases):
    A_map = np.zeros((len(omega_grid), len(k_vals)))
    for i, k in enumerate(k_vals):
        eps_k = k**2/2 - 0.5  # epsilon_k = k^2/(2m) - mu, mu = EF = 1/2
        A_map[:, i] = spectral_function(omega_grid, eps_k, bath, coupling, broadening=2e-2)

    from matplotlib.colors import LogNorm
    A_safe = np.maximum(A_map, 1e-3)
    vmax_val = max(A_safe.max(), 0.1)
    im = ax.pcolormesh(k_vals, omega_grid, A_safe,
                        norm=LogNorm(vmin=1e-2, vmax=vmax_val),
                        shading='auto', cmap='inferno')
    plt.colorbar(im, ax=ax, label=r'$A(k,\omega)$')

    # dispersão livre como referência
    k_ref = np.linspace(0.1, 1.8, 100)
    eps_ref = k_ref**2/2 - 0.5
    ax.plot(k_ref, eps_ref, 'w--', lw=1.5, alpha=0.7, label='livre')

    ax.axhline(0, color='cyan', ls=':', alpha=0.7, label=r'$\mu$')
    ax.axvline(1, color='cyan', ls=':', alpha=0.7, label=r'$k_F$')
    ax.set_xlabel(r'$k/k_F$'); ax.set_ylabel(r'$\omega$')
    ax.set_title(label)
    ax.legend(loc='upper left', fontsize=8)

plt.tight_layout()
plt.show()
plt.savefig(f'{FIGDIR}/fig06_dispersion_map.png', dpi=140, bbox_inches='tight')
plt.close()
print(f"    salvo: fig06_dispersion_map.png")


# =============================================================
# FIG 7: Efeito de k_0 (escala do banho)
# =============================================================

print("[FIG 7] Efeito da escala do banho k_0")

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

# (a) Função F(k_0/k_F) — peso do espaço de fase de acoplamento
k0_range = np.linspace(0.2, 4.0, 100)
F_vals = []
bath_ref = BathParams(m0=1.0, k0=1.0, alpha=0.3)
coupling_ref = CouplingParams(g=1.0)
for k0 in k0_range:
    b = BathParams(m0=1.0, k0=k0, alpha=0.3)
    F_vals.append(momentum_cutoff_factor(b, coupling_ref))

axes[0].plot(k0_range, F_vals, 'b-', lw=2)
axes[0].axvline(2.0, color='k', ls='--', alpha=0.5, label=r'$k_0 = 2k_F$')
axes[0].set_xlabel(r'$k_0/k_F$')
axes[0].set_ylabel(r'$F(k_0/k_F)$ — peso de fase')
axes[0].set_title(r'Espaço de fase de back-scattering')
axes[0].legend()

# (b) função espectral para diferentes k_0
k0_vals = [0.5, 1.0, 2.0, 5.0]
omega = np.linspace(-1.5, 1.5, 600)
for k0, c in zip(k0_vals, plt.cm.viridis(np.linspace(0.15, 0.85, len(k0_vals)))):
    bath = BathParams(m0=1.0, k0=k0, alpha=0.5)
    coupling = CouplingParams(g=2.0)
    A = spectral_function(omega, 0.0, bath, coupling, broadening=8e-3)
    label = rf'$k_0={k0}\,k_F$ ($E_0={bath.E0:.2g}$, $F={momentum_cutoff_factor(bath,coupling):.2f}$)'
    axes[1].plot(omega, A, color=c, label=label)

axes[1].set_xlabel(r'$\omega$')
axes[1].set_ylabel(r'$A(k_F, \omega)$')
axes[1].set_title(r'(b) Espectros para diferentes $k_0$ ($\alpha=0.5$, $g=2$)')
axes[1].legend(fontsize=8)
axes[1].axvline(0, color='k', lw=0.5)

plt.tight_layout()
plt.show()
plt.savefig(f'{FIGDIR}/fig07_k0_dependence.png', dpi=140, bbox_inches='tight')
plt.close()
print(f"    salvo: fig07_k0_dependence.png")


# =============================================================
# FIG 8: Efeito de m_0 (massa do banho)
# =============================================================

print("[FIG 8] Efeito da massa do banho m_0")

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

m0_vals = [1, 10, 100, 1000]  # elétron até massa de íon
omega = np.linspace(-1.5, 1.5, 600)

for m0, c in zip(m0_vals, plt.cm.plasma(np.linspace(0.15, 0.85, len(m0_vals)))):
    bath = BathParams(m0=m0, k0=1.0, alpha=0.5)
    coupling = CouplingParams(g=2.0)
    A = spectral_function(omega, 0.0, bath, coupling, broadening=8e-3)
    label = rf'$m_0/m_e={m0}$ ($E_0={bath.E0:.2g}$)'
    axes[0].plot(omega, A, color=c, label=label)

axes[0].set_xlabel(r'$\omega$ (em $E_F$)')
axes[0].set_ylabel(r'$A(k_F, \omega)$')
axes[0].set_title(r'(a) Função espectral vs. massa do banho')
axes[0].legend(fontsize=8); axes[0].axvline(0, color='k', lw=0.5)

# (b) zoom em escala log mostrando estrutura adiabática
for m0, c in zip(m0_vals, plt.cm.plasma(np.linspace(0.15, 0.85, len(m0_vals)))):
    bath = BathParams(m0=m0, k0=1.0, alpha=0.5)
    coupling = CouplingParams(g=2.0)
    # Mostrar |Im Sigma| em log-log
    om_pos = np.logspace(-5, 0.5, 200)
    im_S = -im_sigma_e(om_pos, bath, coupling)
    axes[1].loglog(om_pos, im_S, color=c, label=rf'$m_0/m_e={m0}$')

axes[1].set_xlabel(r'$\omega$ (em $E_F$)')
axes[1].set_ylabel(r'$|\mathrm{Im}\,\Sigma_e^R|$')
axes[1].set_title(r'(b) $|\mathrm{Im}\,\Sigma_e|$ — cutoff adiabático em $E_0(m_0)$')
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.show()
plt.savefig(f'{FIGDIR}/fig08_m0_dependence.png', dpi=140, bbox_inches='tight')
plt.close()
print(f"    salvo: fig08_m0_dependence.png")


# =============================================================
# FIG 9: Densidade de estados (DOS)
# =============================================================

print("[FIG 9] Densidade de estados")

fig, ax = plt.subplots(figsize=(8, 5.5))
omega_DOS = np.linspace(-1.0, 1.0, 200)

configs = [
    ('livre', None, None, 'k', '--'),
    (r'$\alpha=0.2$', BathParams(m0=1, k0=1, alpha=0.2), CouplingParams(g=2.0), 'tab:blue', '-'),
    (r'$\alpha=0.5$', BathParams(m0=1, k0=1, alpha=0.5), CouplingParams(g=2.0), 'tab:orange', '-'),
    (r'$\alpha=1.0$', BathParams(m0=1, k0=1, alpha=1.0), CouplingParams(g=2.0), 'tab:red', '-'),
]

# free DOS de referência
EF = 0.5
free_dos = np.where(omega_DOS + EF > 0,
                    np.sqrt(np.maximum(omega_DOS + EF, 0)/EF),
                    0.0) * (1.0/(2*np.pi**2))
ax.plot(omega_DOS, free_dos, 'k--', label='livre')

for label, bath, coupling, color, ls in configs[1:]:
    N = dos(omega_DOS, bath, coupling, n_k=200, broadening=2e-2)
    ax.plot(omega_DOS, N, color=color, ls=ls, label=label)

ax.set_xlabel(r'$\omega$'); ax.set_ylabel(r'$N(\omega)$')
ax.set_title(r'Densidade de estados ($g=2$, $m_0=m_e$, $k_0=k_F$)')
ax.legend(); ax.axvline(0, color='gray', lw=0.5)
plt.tight_layout()
plt.show()
plt.savefig(f'{FIGDIR}/fig09_dos.png', dpi=140, bbox_inches='tight')
plt.close()
print(f"    salvo: fig09_dos.png")

print("\nAnálise do gás de elétrons concluída.")
