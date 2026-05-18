import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.integrate import simpson

# =============================================================================
# PHYSICAL CONSTANTS (Natural Units: hbar = kB = 1)
# =============================================================================
E_F = 1.0
k_F = 1.0
m = 0.5          # E_F = k_F^2/(2m) = 1.0
v_F = k_F / m    # Fermi velocity = 2.0
ETA = 1e-4 * E_F # Causal broadening

# =============================================================================
# CORE PHYSICS FUNCTIONS (Syntax & Kinematics Corrected)
# =============================================================================
def get_dispersion(q, cfg):
    if cfg['dispersion'] == 'constant': 
        return cfg['k0']**2 / (2.0 * cfg['M'])
    elif cfg['dispersion'] == 'linear': 
        return cfg['c'] * q / cfg['k0']
    elif cfg['dispersion'] == 'quadratic': 
        return np.sqrt(q**2 / (2.0 * cfg['M']) + cfg['k0']**2 / (2.0 * cfg['M']))

def get_damping(nu, cfg):
    if cfg['damping'] == 'ohmic': 
        return nu * cfg['gamma0']
    elif cfg['damping'] == 'drude': 
        return cfg['GammaD'] / (1.0 + 1j * (nu / cfg['omegaD']))

# =============================================================================
# SELF-ENERGY & PHYSICALLY-SCALED RESISTIVITY
# =============================================================================
def im_sigma_3d_simpson(omega, T, cfg):
    """ImΣ with corrected kinematics: xi = ε_{k+q} - E_F (k on Fermi surface)"""
    q_pts = np.linspace(1e-6, cfg['k0'], 61)
    th_pts = np.linspace(0, np.pi, 31)
    Q, TH = np.meshgrid(q_pts, th_pts, indexing='ij')
    
    # CORRECTED: fermion energy after scattering (k on FS)
    xi = Q**2 / (2*m) + v_F * Q * np.cos(TH)
    nu = omega - xi

    Omega_q = get_dispersion(Q, cfg)
    gamma_nu = get_damping(nu, cfg)

    response = 1.0 / (Omega_q**2 - nu**2 - 1j * nu * gamma_nu)
    A_D = np.imag(response)

    n_F = np.tanh(0.5 * nu / T)
    n_B = 1.0 / (np.tanh(0.5 * xi / T) + 1e-12)

    measure = (Q**2 * np.sin(TH)) / (4.0 * np.pi**2)
    integrand = A_D * (n_F + n_B) * measure

    int_th = simpson(integrand, th_pts, axis=1)
    return -np.pi * cfg['g']**2 * simpson(int_th, q_pts, axis=0)

def compute_resistivity_scaled(T, cfg, omega_max_factor=5.0, N_omega=40):
    """
    DC resistivity with physical scaling:
    ρ(T) ∝ (1/v_F²) ∫ dω [-∂f/∂ω] ω |ImΣ(ω,T)|
    """
    omega_max = max(omega_max_factor * T, 1e-3 * E_F)
    omega_pts = np.linspace(1e-6 * E_F, omega_max, N_omega)
    
    # Thermal window: -∂f/∂ω
    thermal_weight = 1.0 / (4.0 * T * np.cosh(omega_pts / (2.0 * T))**2 + 1e-12)
    
    # Transport vertex approximation <1-cosθ> ≈ <q²>/(2k_F²)
    q_avg_sq = cfg['k0']**2 / 3.0
    transport_vertex = q_avg_sq / (2.0 * k_F**2)
    
    # Integrate
    im_S_vals = np.array([im_sigma_3d_simpson(w, T, cfg) for w in omega_pts])
    integrand = thermal_weight * omega_pts * np.abs(im_S_vals)
    integral = simpson(integrand, omega_pts)
    
    # Drude prefactor scaling
    rho = (transport_vertex / v_F**2) * integral
    return np.clip(rho, 1e-12, 1e6)

def extract_local_alpha(T, cfg, delta_frac=0.2):
    """Extract α = d lnρ / d lnT via centered log-log difference"""
    dT = max(T * delta_frac, 1e-3)
    T_pts = np.array([max(T - dT, 1e-4), T, min(T + dT, 2.0)])
    rho_pts = np.array([compute_resistivity_scaled(t, cfg) for t in T_pts])
    
    mask = rho_pts > 1e-15
    if mask.sum() < 2: return np.nan
    
    # Linear fit in log-log space for robustness
    alpha = np.polyfit(np.log(T_pts[mask]), np.log(rho_pts[mask]), 1)[0]
    return alpha #np.clip(alpha, -0.5, 3.0)

# =============================================================================
# ANALYTICAL QCP LOCUS (T=0 Prediction)
# =============================================================================
def get_analytical_k0_c(g, cfg):
    """k₀^c(g) from Ω_q² = g²|Π₀(2k_F)| at T=0"""
    N0 = m * k_F / (2 * np.pi**2)
    Pi_2kF = 0.5 * N0  # |Π₀(2k_F)|
    
    if cfg['dispersion'] == 'constant':
        return (4 * cfg['M']**2 * g**2 * Pi_2kF)**0.25
    elif cfg['dispersion'] == 'linear':
        return cfg['c'] * 2*k_F / np.sqrt(g**2 * Pi_2kF)
    elif cfg['dispersion'] == 'quadratic':
        rhs = 2 * cfg['M'] * g**2 * Pi_2kF - (2*k_F)**2
        return np.sqrt(rhs) if rhs > 0 else np.nan
    return np.nan

def get_analytical_g_c(k0, cfg):
    """Inverse: g^c(k₀)"""
    if cfg['dispersion'] == 'constant':
        return np.sqrt((k0**4 / (4*cfg['M']**2)) / (0.5 * m * k_F / (2*np.pi**2)))
    elif cfg['dispersion'] == 'linear':
        Pi = 0.5 * m * k_F / (2*np.pi**2)
        return cfg['c'] * 2*k_F / (k0 * np.sqrt(Pi))
    elif cfg['dispersion'] == 'quadratic':
        Pi = 0.5 * m * k_F / (2*np.pi**2)
        return np.sqrt(((k0**2 + (2*k_F)**2) / (2*cfg['M'])) / Pi)
    return np.nan

# =============================================================================
# PLOTTING: T vs Control Parameter (Resistivity Exponent α)
# =============================================================================
plt.rcParams.update({
    'font.size': 10, 'text.usetex': False, 'font.family': 'serif',
    'axes.linewidth': 1.0, 'lines.linewidth': 1.5, 'figure.dpi': 150,
    'axes.titlesize': 11, 'axes.labelsize': 10, 'legend.fontsize': 9
})

def plot_T_vs_param_alpha(param_vals, T_vals, param_name, cfg, title, xlabel):
    n_p, n_T = len(param_vals), len(T_vals)
    X, Y = np.meshgrid(param_vals, T_vals)
    alpha_mat = np.zeros((n_T, n_p))

    for i, T in enumerate(tqdm(T_vals, desc=f'T-scan ({param_name})')):
        for j, p in enumerate(param_vals):
            cfg_scan = cfg.copy()
            cfg_scan[param_name] = p
            alpha_mat[i, j] = extract_local_alpha(T, cfg_scan)

    fig, ax = plt.subplots(figsize=(7, 5))
    pcm = ax.pcolormesh(X, Y, alpha_mat, shading='auto', cmap='RdYlGn_r', vmin=0.0, vmax=2.5)
    cbar = fig.colorbar(pcm, ax=ax, label='Resistivity Exponent α (ρ ~ T^α)')
    
    # Contours for key regimes
    ax.contour(X, Y, alpha_mat, levels=[1.0, 1.5, 2.0], colors=['red', 'black', 'blue'], 
               linewidths=1.2, linestyles=['-', '--', ':'])
    
    # Analytical QCP overlay
    if param_name == 'k0':
        k0_c = get_analytical_k0_c(cfg['g'], cfg)
        if np.isfinite(k0_c) and 0 < k0_c < param_vals.max():
            ax.axvline(k0_c, color='white', linestyle='--', linewidth=2.0, zorder=5)
            ax.text(k0_c, T_vals.min() + 0.02, f'QCP: $k_0^c$={k0_c/k_F:.2f}$k_F$', 
                    color='white', fontsize=9, ha='center', va='bottom', rotation=90,
                    bbox=dict(boxstyle='round', facecolor='black', alpha=0.6))
    elif param_name == 'g':
        g_c = get_analytical_g_c(cfg['k0'], cfg)
        if np.isfinite(g_c) and 0 < g_c < param_vals.max():
            ax.axhline(g_c, color='white', linestyle='--', linewidth=2.0, zorder=5)
            ax.text(param_vals.min() + 0.02, g_c, f'QCP: $g^c$={g_c:.2f}', 
                    color='white', fontsize=9, ha='left', va='center', rotation=0,
                    bbox=dict(boxstyle='round', facecolor='black', alpha=0.6))

    ax.set_xlabel(xlabel)
    ax.set_ylabel('Temperature T')
    ax.set_title(f'{title}\nρ(T) scaling exponent')
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    plt.show()

# =============================================================================
# MAIN EXECUTION
# =============================================================================
configs = [
    {'name': 'Ohmic + Constant',    'damping': 'ohmic',   'dispersion': 'constant',    'gamma0': 0.5, 'M': 0.5, 'k0': 1.0, 'g': 1.0},
    #{'name': 'Ohmic + Linear',      'damping': 'ohmic',   'dispersion': 'linear',      'gamma0': 0.5, 'c': 1.0, 'k0': 1.0, 'g': 1.0},
    #{'name': 'Ohmic + Quadratic',   'damping': 'ohmic',   'dispersion': 'quadratic',   'gamma0': 0.5, 'M': 0.5, 'k0': 1.0, 'g': 1.0},
    #{'name': 'Drude + Constant',    'damping': 'drude',   'dispersion': 'constant',    'GammaD': 0.5, 'M': 0.5, 'omegaD': 1.0, 'k0': 1.0, 'g': 1.0},
    #{'name': 'Drude + Linear',      'damping': 'drude',   'dispersion': 'linear',      'GammaD': 0.5, 'omegaD': 1.0, 'c': 1.0, 'k0': 1.0, 'g': 1.0},
    #{'name': 'Drude + Quadratic',   'damping': 'drude',   'dispersion': 'quadratic',   'GammaD': 0.5, 'omegaD': 1.0, 'M': 0.5, 'k0': 1.0, 'g': 1.0},
]

# Scan ranges
T_range = np.linspace(0.005 * E_F, 1.0 * E_F, 50)
k0_range = np.linspace(0.05 * k_F, 0.7* k_F, 55)
g_range = np.linspace(0.3*E_F, 20.0*E_F, 55)

for cfg in configs:
    print(f"\n{'='*50}")
    print(f"Processing: {cfg['name']}")
    print(f"{'='*50}")
    
    # 1. T vs k0 (at fixed g)
    plot_T_vs_param_alpha(k0_range, T_range, 'k0', cfg, 
                         f'{cfg["name"]} (T vs k₀)', 'Momentum scale k₀')
                         
    # 2. T vs g (at fixed k₀)
    plot_T_vs_param_alpha(g_range, T_range, 'g', cfg, 
                         f'{cfg["name"]} (T vs g)', 'Coupling strength g')

print("\n✅ All resistivity exponent phase diagrams generated.")
print("📖 Guide: α≈2 (blue) → Fermi Liquid | α≈1 (red) → Planckian/QCP | α<1 → Bad Metal")
print("   White dashed line = analytical T=0 QCP prediction")