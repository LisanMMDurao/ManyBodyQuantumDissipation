"""
CUDA-Accelerated Phase Diagram: Resistivity Exponent α(T, control parameter)
Uses PyTorch with custom Simpson integration (GPU) or Monte Carlo (GPU).
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print(f"Using device: {DEVICE}")

# Integration settings
INTEGRATOR = 'simpson'      # 'simpson' or 'mc'
MC_N_SAMPLES = 500_000      # Monte Carlo samples per integration

# Simpson settings – enforce odd numbers of points
N_Q_BASE = 1000              # will be made odd
N_TH_BASE = 1000             # will be made odd
if INTEGRATOR == 'simpson':
    N_Q = N_Q_BASE if (N_Q_BASE % 2 == 1) else N_Q_BASE + 1
    N_TH = N_TH_BASE if (N_TH_BASE % 2 == 1) else N_TH_BASE + 1
else:
    N_Q = N_Q_BASE
    N_TH = N_TH_BASE

Q_MAX_FACTOR = 4.0         # q range = [1e-16, Q_MAX_FACTOR * k0]

# Phase diagram scanning settings (adjust for speed vs. resolution)
N_T = 180                    # number of temperature points
N_PARAM =  180             # number of parameter points
T_MIN, T_MAX = 0.001, .125   # temperature range (in units of E_F)
LOG_PARAM = False           # linear spacing by default

# -----------------------------------------------------------------------------
# Physical constants (natural units: hbar = kB = 1)
# -----------------------------------------------------------------------------
E_F = 1.0
k_F = 1.0
m = 0.5                     # E_F = k_F^2/(2m)
v_F = k_F / m               # Fermi velocity = 2.0
N_0 = m * k_F / (2.0 * np.pi**2)       # DOS at Fermi level
ABS_PI_0 = N_0 / 2.0                    # |Π₀(2k_F)|

# -----------------------------------------------------------------------------
# Helper: Simpson's rule on a uniform grid (torch implementation)
# -----------------------------------------------------------------------------
def simpson_torch(y, x, axis=-1):
    """
    Simpson's rule integration for evenly spaced points.
    y : tensor of values at points x
    x : 1D tensor of coordinates (must be evenly spaced)
    axis : axis along which to integrate
    Returns integral (scalar or tensor with other dimensions reduced).
    """
    n = y.shape[axis]
    assert n % 2 == 1, "Number of points must be odd for Simpson's rule"
    dx = (x[-1] - x[0]) / (n - 1)
    # Simpson weights: 1,4,2,4,...,2,4,1
    weights = torch.ones_like(y)
    if axis == -1:
        weights[..., 1:-1:2] = 4.0
        weights[..., 2:-2:2] = 2.0
    else:
        # handle generic axis - not needed for our 1D/2D use, but kept simple
        idx = [slice(None)] * y.ndim
        idx[axis] = slice(1, -1, 2)
        weights[tuple(idx)] = 4.0
        idx[axis] = slice(2, -2, 2)
        weights[tuple(idx)] = 2.0
    integral = torch.sum(weights * y, dim=axis) * dx / 3.0
    return integral

# -----------------------------------------------------------------------------
# Physics functions (all tensor operations)
# -----------------------------------------------------------------------------
def get_dispersion(q, cfg):
    """Return boson dispersion Ω_q (without the polarization shift)."""
    if cfg['dispersion'] == 'constant':
        return torch.full_like(q, cfg['k0']**2 / (2.0 * cfg['M']))
    elif cfg['dispersion'] == 'linear':
        return cfg['c'] * q / cfg['k0']
    elif cfg['dispersion'] == 'quadratic':
        return torch.sqrt(q**2 / (2.0 * cfg['M']) + cfg['k0']**2 / (2.0 * cfg['M']))
    else:
        return torch.ones_like(q)

def get_damping(nu, cfg):
    """Return damping rate Γ(ν) (could be complex for Drude)."""
    if cfg['damping'] == 'ohmic':
        return cfg['gamma0'] * nu
    elif cfg['damping'] == 'drude':
        return cfg['GammaD'] / (1.0 + 1j * (nu / cfg['omegaD']))
    else:
        return torch.zeros_like(nu)

# -----------------------------------------------------------------------------
# Self-energy integration on GPU
# -----------------------------------------------------------------------------
def im_sigma_simpson(omega, T, cfg):
    """ImΣ(ω,T) using Simpson integration on GPU."""
    # Create q and theta grids
    q_max = Q_MAX_FACTOR * cfg['k0']
    q = torch.linspace(1e-12, q_max, N_Q, device=DEVICE)
    th = torch.linspace(0.0, np.pi, N_TH, device=DEVICE)
    Q, TH = torch.meshgrid(q, th, indexing='ij')
    # q^2 * sinθ * dq dθ measure (Jacobian)
    measure = (Q**2 * torch.sin(TH)) / (4.0 * np.pi**2)

    # Fermion energy after scattering (on Fermi surface)
    xi = Q**2 / (2*m) + v_F * Q * torch.cos(TH)
    nu = omega - xi

    # Boson dispersion (with polarization shift)
    Omega_q = get_dispersion(Q, cfg) - ABS_PI_0 * cfg['g']**2
    gamma_nu = get_damping(nu, cfg)

    # Boson spectral function A_D = -Im[ 1/(Ω² - ν² - i ν Γ) ]
    denom = Omega_q**2 - nu**2 - 1j * nu * gamma_nu + 1e-12
    A_D = -torch.imag(1.0 / denom)

    # Thermal factors (clipped for stability)
    n_F = 0.5-0.5*torch.tanh(0.5 * nu / T).clamp(-50, 50)
    n_B = 0.5*1.0 / (torch.tanh(0.5 * xi / T).clamp(-50, 50) + 1e-12)-0.5

    integrand = A_D * (n_F + n_B) * measure

    # Integrate over theta, then over q
    int_th = simpson_torch(integrand, th, axis=1)  # shape (N_Q,)
    integral = simpson_torch(int_th, q, axis=0)    # scalar
    return -np.pi * cfg['g']**2 * integral.item()

def im_sigma_mc(omega, T, cfg):
    """ImΣ(ω,T) using GPU Monte Carlo."""
    q = torch.rand(MC_N_SAMPLES, device=DEVICE) * (Q_MAX_FACTOR * cfg['k0'])
    th = torch.rand(MC_N_SAMPLES, device=DEVICE) * np.pi
    xi = q**2 / (2*m) + v_F * q * torch.cos(th)
    nu = omega - xi
    Omega_q = get_dispersion(q, cfg) - ABS_PI_0 * cfg['g']**2
    gamma_nu = get_damping(nu, cfg)
    denom = Omega_q**2 - nu**2 - 1j * nu * gamma_nu + 1e-12
    A_D = -torch.imag(1.0 / denom)
    n_F = torch.tanh(0.5 * nu / T).clamp(-30, 30)
    n_B = 1.0 / (torch.tanh(0.5 * xi / T).clamp(-30, 30) + 1e-12)
    integrand = A_D * (n_F + n_B) * q**2 * torch.sin(th) / (4.0 * np.pi**2)

    # Integration volume: 2π from phi, full q & theta range
    vol = (cfg['k0']**3 / 3.0) * 2.0 / (4.0 * np.pi**2) * np.pi
    mean_val = torch.mean(integrand) * vol * (4.0 * np.pi**2)
    return -np.pi * cfg['g']**2 * mean_val.item()

# Select integrator
if INTEGRATOR == 'simpson':
    im_sigma_func = im_sigma_simpson
else:
    im_sigma_func = im_sigma_mc

# -----------------------------------------------------------------------------
# Resistivity proxy and exponent extraction
# -----------------------------------------------------------------------------
def compute_resistivity_proxy(T, cfg, omega_min=1e-4):
    """ρ ∝ |ImΣ(ω→0, T)| with Drude prefactor."""
    im_S = np.abs(im_sigma_func(omega_min, T, cfg))
    return im_S / v_F**2 * (cfg['k0']**2 / (6 * k_F**2))

def extract_local_alpha(T, cfg, delta_frac=0.25):
    """α = d lnρ / d lnT via centered finite difference."""
    dT = max(T * delta_frac, 1e-4)
    T_pts = np.array([max(T - dT, 1e-6), T, min(T + dT, 2.0)])
    rho_pts = np.array([compute_resistivity_proxy(t, cfg) for t in T_pts])
    mask = rho_pts > 1e-15
    if mask.sum() < 2:
        return np.nan
    alpha = np.polyfit(np.log(T_pts[mask]), np.log(rho_pts[mask]), 1)[0]
    return alpha# np.clip(alpha, 0.0, 2.5)

# -----------------------------------------------------------------------------
# Phase diagram plotting
# -----------------------------------------------------------------------------
def plot_T_vs_param_phase_diagram(param_vals, T_vals, param_name, cfg, title, xlabel):
    """2D scan over parameter and temperature; compute α and plot."""
    n_T, n_p = len(T_vals), len(param_vals)
    alpha_mat = np.full((n_T, n_p), np.nan)

    print(f"  Scanning {n_T} T × {n_p} {param_name} ...")
    for i, T in enumerate(tqdm(T_vals, desc=f'T-scan ({param_name})')):
        for j, p in enumerate(param_vals):
            cfg_scan = cfg.copy()
            cfg_scan[param_name] = p
            try:
                alpha_mat[i, j] = extract_local_alpha(T, cfg_scan)
            except Exception as e:
                alpha_mat[i, j] = np.nan

    # Fallback: plot resistivity if all α are NaN
    if np.all(np.isnan(alpha_mat)):
        print(f"  Warning: all α NaN – plotting resistivity fallback.")
        rho_mat = np.zeros((n_T, n_p))
        for i, T in enumerate(tqdm(T_vals, desc='Fallback')):
            for j, p in enumerate(param_vals):
                cfg_scan = cfg.copy()
                cfg_scan[param_name] = p
                rho_mat[i, j] = compute_resistivity_proxy(T, cfg_scan)
        fig, ax = plt.subplots(figsize=(7,5))
        log_rho = np.log10(np.clip(rho_mat, 1e-12, None))
        pcm = ax.pcolormesh(param_vals, T_vals, log_rho, shading='auto', cmap='plasma')
        fig.colorbar(pcm, ax=ax, label='log₁₀[ρ]')
        ax.set_xlabel(xlabel); ax.set_ylabel('T')
        ax.set_title(f'{title} [fallback]')
        plt.tight_layout(); plt.show()
        return

    # Main plot
    fig, ax = plt.subplots(figsize=(7,5))
    X, Y = np.meshgrid(param_vals, T_vals)
    valid = alpha_mat[~np.isnan(alpha_mat)]
    vmin, vmax = (np.percentile(valid, 5), np.percentile(valid, 95)) if len(valid) else (0,2)
    pcm = ax.pcolormesh(X, Y, alpha_mat, shading='auto', cmap='RdYlGn_r',
                        vmin=0.0, vmax=2.0)
    cbar = fig.colorbar(pcm, ax=ax, label=r'$\alpha$  ($\rho \sim T^\alpha$)')
    # Contours
    for lvl, col, ls in zip([1.0,1.5,1.8], ['red','black','blue'], ['-','--',':']):
        try:
            ax.contour(X, Y, alpha_mat, levels=[lvl], colors=[col],
                       linewidths=1.2, linestyles=[ls], alpha=0.8)
        except:
            pass
    ax.set_xlabel(xlabel); ax.set_ylabel('Temperature T')
    ax.set_title(title)
    ax.text(0.02, 0.98, 'α≈2: FL\nα≈1: Planckian\nα<1: Bad Metal',
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.grid(True, alpha=0.2)
    plt.tight_layout(); plt.show()

# -----------------------------------------------------------------------------
# Main execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Define model configuration (only one active for demonstration)
    cfg = {
        'name': 'Ohmic + Constant',
        'damping': 'ohmic',
        'dispersion': 'constant',
        'gamma0': 5.0,
        'M': 0.5,
        'k0': 1.0,
        'g': 2.0,
    }

    # Scan ranges (adjust N_T, N_PARAM above for speed/resolution)
    T_vals = np.linspace(T_MIN, T_MAX, N_T)
    k0_vals = np.linspace(1e-20, 0.5, N_PARAM)
    g_vals = np.linspace(0.3, 20.0, N_PARAM)
    gamma0_vals = np.linspace(5.0, 15.0, N_PARAM)

    print("="*70)
    print("GPU‑Accelerated Phase Diagrams")
    print(f"Integrator: {INTEGRATOR} | Device: {DEVICE}")
    print("="*70)

    # 1. T vs k0
    plot_T_vs_param_phase_diagram(k0_vals, T_vals, 'k0', cfg,
                                  f'{cfg["name"]} (T vs k₀)', 'Momentum scale k₀')
    # 2. T vs g
    #plot_T_vs_param_phase_diagram(g_vals, T_vals, 'g', cfg,
    #                              f'{cfg["name"]} (T vs g)', 'Coupling g')
    # 3. T vs damping (ohmic gamma0)
    #plot_T_vs_param_phase_diagram(gamma0_vals, T_vals, 'gamma0', cfg,
     #                             f'{cfg["name"]} (T vs γ₀)', 'Damping γ₀')

    print("\n✅ Phase diagram generation complete.")
