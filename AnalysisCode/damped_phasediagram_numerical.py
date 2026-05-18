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
# Static Lindhard polarization magnitude at q_c = 2k_F
N_0 = m * k_F / (2.0 * np.pi**2)  # DOS at Fermi level
ABS_PI_0 = N_0 / 2.0  # |Π₀(2k_F)| = (m k_F)/(4π²)
# =============================================================================
# CORE PHYSICS FUNCTIONS (SYNTAX FIXED)
# =============================================================================
def get_dispersion(q, cfg):
    if cfg['dispersion'] == 'constant': 
        return cfg['k0']**2 / (2.0 * cfg['M'])
    elif cfg['dispersion'] == 'linear': 
        return cfg['c'] * q / cfg['k0']
    elif cfg['dispersion'] == 'quadratic': 
        return np.sqrt(q**2 / (2.0 * cfg['M']) + cfg['k0']**2 / (2.0 * cfg['M']))
    return 1.0

def get_damping(nu, cfg):
    if cfg['damping'] == 'ohmic': 
        return nu * cfg['gamma0']
    elif cfg['damping'] == 'drude': 
        return cfg['GammaD'] / (1.0 + 1j * (nu / cfg['omegaD']))
    return 0.0

# =============================================================================
# SELF-ENERGY INTEGRATION (SYNTAX FIXED + STABILIZED)
# =============================================================================
def im_sigma_3d_simpson(omega, T, cfg):
    """Deterministic 2D Simpson integration for ImΣ(ω,T)."""
    q_pts = np.linspace(1e-16, 16*cfg['k0'], 1000)
    th_pts = np.linspace(0, np.pi, 31)
    Q, TH = np.meshgrid(q_pts, th_pts, indexing='ij')
    
    # Fermion energy after scattering (k on Fermi surface)
    xi = Q**2 / (2*m) + v_F * Q * np.cos(TH)
    nu = omega - xi

    Omega_q = get_dispersion(Q, cfg)  - ABS_PI_0*cfg['g']**2
    gamma_nu = get_damping(nu, cfg)

    # Boson spectral function
    denom = Omega_q**2 - nu**2 - 1j * nu * gamma_nu + 1e-12
    response = 1.0 / denom
    A_D = np.imag(response)

    # Thermal factors with overflow protection
    n_F = np.tanh(0.5 * nu / T)
    n_B = 1.0 / (np.tanh(0.5 * xi / T) + 1e-12)

    measure = (Q**2 * np.sin(TH)) / (4.0 * np.pi**2)
    integrand = A_D * (n_F + n_B) * measure

    int_th = simpson(integrand, th_pts, axis=1)
    return -np.pi * cfg['g']**2 * simpson(int_th, q_pts, axis=0)

def im_sigma_3d_mc(omega, T, cfg, N=200000):
    """Monte Carlo integration fallback for faster scans."""
    q = np.random.uniform(1e-6, 2*cfg['k0'], N)
    th = np.random.uniform(0, np.pi, N)
    xi = q**2 / (2*m) + v_F * q * np.cos(th)
    nu = omega - xi

    Omega_q = get_dispersion(q, cfg)
    gamma_nu = get_damping(nu, cfg)

    denom = Omega_q**2 - nu**2 - 1j * nu * gamma_nu + 1e-12
    A_D = np.imag(1.0 / denom)

    n_F = np.tanh(np.clip(0.5 * nu / T, -30, 30))
    n_B = 1.0 / (np.tanh(np.clip(0.5 * xi / T, -30, 30)) + 1e-12)

    integrand = A_D * (n_F + n_B) * q**2 * np.sin(th) / (4.0 * np.pi**2)
    vol = (cfg['k0']**3 / 3.0) * 2.0 / (4.0 * np.pi**2) * np.pi
    return -np.pi * cfg['g']**2 * np.mean(integrand) * vol * (4.0 * np.pi**2)

# Select integrator
INTEGRATOR = 'simpson'
im_sigma_func = im_sigma_3d_simpson if INTEGRATOR == 'simpson' else im_sigma_3d_mc

# =============================================================================
# RESISTIVITY & EXPONENT EXTRACTION (ROBUST)
# =============================================================================
def compute_resistivity_proxy(T, cfg, omega_min=1e-4):
    """DC resistivity proxy: ρ ∝ |ImΣ(ω→0, T)| with physical scaling."""
    im_S = np.abs(im_sigma_func(omega_min, T, cfg))
    # Drude prefactor and transport vertex approximation
    return np.clip(im_S / v_F**2 * (cfg['k0']**2 / (6 * k_F**2)), 1e-12, 1e6)

def extract_local_alpha(T, cfg, delta_frac=0.25):
    """Extract α = d lnρ / d lnT via centered log-log difference."""
    dT = max(T * delta_frac, 1e-3)
    T_pts = np.array([max(T - dT, 1e-4), T, min(T + dT, 2.0)])
    rho_pts = np.array([compute_resistivity_proxy(t, cfg) for t in T_pts])
    
    mask = rho_pts > 1e-15
    if mask.sum() < 2:
        return np.nan
    # Linear fit in log-log space
    alpha = np.polyfit(np.log(T_pts[mask]), np.log(rho_pts[mask]), 1)[0]
    return np.clip(alpha, 0.0, 2.0)

# =============================================================================
# PLOTTING (FIXED + ADAPTIVE SCALING)
# =============================================================================
plt.rcParams.update({
    'font.size': 11, 'text.usetex': False, 'font.family': 'serif',
    'axes.linewidth': 1.0, 'lines.linewidth': 1.8, 'figure.dpi': 150,
    'axes.titlesize': 12, 'axes.labelsize': 11, 'legend.fontsize': 9
})

def plot_T_vs_param_phase_diagram(param_vals, T_vals, param_name, cfg, title, xlabel):
    """Plot T vs control parameter phase diagram for resistivity exponent α."""
    n_p, n_T = len(param_vals), len(T_vals)
    X, Y = np.meshgrid(param_vals, T_vals)
    alpha_mat = np.full((n_T, n_p), np.nan)

    print(f"  Scanning {n_T} temperatures × {n_p} {param_name} values...")
    for i, T in enumerate(tqdm(T_vals, desc=f'T-scan ({param_name})')):
        for j, p in enumerate(param_vals):
            cfg_scan = cfg.copy()
            cfg_scan[param_name] = p
            try:
                alpha_mat[i, j] = extract_local_alpha(T, cfg_scan)
            except Exception as e:
                alpha_mat[i, j] = np.nan

    # Check for all-NaN result
    if np.all(np.isnan(alpha_mat)):
        print(f"  ⚠️  Warning: All α values are NaN. Check parameters or integration.")
        # Fallback: plot raw resistivity instead
        rho_mat = np.zeros((n_T, n_p))
        for i, T in enumerate(tqdm(T_vals, desc='Fallback scan')):
            for j, p in enumerate(param_vals):
                cfg_scan = cfg.copy()
                cfg_scan[param_name] = p
                rho_mat[i, j] = compute_resistivity_proxy(T, cfg_scan)
        
        fig, ax = plt.subplots(figsize=(7, 5))
        log_rho = np.log10(np.clip(rho_mat, 1e-12, None))
        pcm = ax.pcolormesh(X, Y, log_rho, shading='auto', cmap='plasma')
        fig.colorbar(pcm, ax=ax, label='log₁₀[ρ] (arb. units)')
        ax.set_xlabel(xlabel); ax.set_ylabel('Temperature T')
        ax.set_title(f'{title}\n[Fallback: Resistivity]')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        return

    # Normal plotting with adaptive color scaling
    fig, ax = plt.subplots(figsize=(7, 5))
    
    # Adaptive vmin/vmax to prevent flat/blank panels
    valid_data = alpha_mat[~np.isnan(alpha_mat)]
    if len(valid_data) > 0:
        vmin = np.percentile(valid_data, 5)
        vmax = np.percentile(valid_data, 95)
    else:
        vmin, vmax = 0.0, 2.0
    
    pcm = ax.pcolormesh(X, Y, alpha_mat, shading='auto', cmap='RdYlGn_r', 
                        vmin=max(0.0, vmin), vmax=min(3.0, vmax))
    cbar = fig.colorbar(pcm, ax=ax, label='Resistivity Exponent α (ρ ~ T^α)')
    
    # Contours for key regimes
    levels = [1.0, 1.5, 2.0]
    colors = ['red', 'black', 'blue']
    linestyles = ['-', '--', ':']
    for lvl, col, ls in zip(levels, colors, linestyles):
        try:
            ax.contour(X, Y, alpha_mat, levels=[lvl], colors=[col], 
                      linewidths=1.2, linestyles=[ls], alpha=0.8)
        except:
            pass
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Temperature T')
    ax.set_title(f'{title}\nρ(T) scaling exponent')
    ax.grid(True, alpha=0.2)
    
    # Annotation for regimes
    ax.text(0.02, 0.98, 'α≈2: Fermi Liquid\nα≈1: Planckian/QCP\nα<1: Bad Metal', 
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    fig.tight_layout()
    plt.show()

# =============================================================================
# CONFIGURATIONS & MAIN EXECUTION
# =============================================================================
configs = [
    {'name': 'Ohmic + Constant',    'damping': 'ohmic',   'dispersion': 'constant',    'gamma0': 0.5, 'M': 0.5, 'k0': 1.0, 'g': 10.0},
    #{'name': 'Ohmic + Linear',      'damping': 'ohmic',   'dispersion': 'linear',      'gamma0': 0.5, 'c': 1.0, 'k0': 1.0, 'g': 1.0},
    #{'name': 'Ohmic + Quadratic',   'damping': 'ohmic',   'dispersion': 'quadratic',   'gamma0': 0.5, 'M': 0.5, 'k0': 1.0, 'g': 1.0},
    #{'name': 'Drude + Constant',    'damping': 'drude',   'dispersion': 'constant',    'GammaD': 0.5, 'M': 0.5, 'omegaD': 1.0, 'k0': 1.0, 'g': 1.0},
    #{'name': 'Drude + Linear',      'damping': 'drude',   'dispersion': 'linear',      'GammaD': 0.5, 'omegaD': 1.0, 'c': 1.0, 'k0': 1.0, 'g': 1.0},
    #{'name': 'Drude + Quadratic',   'damping': 'drude',   'dispersion': 'quadratic',   'GammaD': 0.5, 'omegaD': 1.0, 'M': 0.5, 'k0': 1.0, 'g': 1.0},
]

# Scan ranges (low-T focus for QCP)
T_range = np.linspace(0.005 * E_F, 1.0 * E_F, 1000)
k0_range = np.linspace(0.05 * k_F, 2.0 * k_F, 1000)
g_range = np.linspace(0.3, 20.0, 1000)
damp_range = np.linspace(0.001 * E_F, 3.0 * E_F, 1000)

print("="*70)
print("PHASE DIAGRAMS: Resistivity Exponent α(T, control parameter)")
print("Fixed syntax errors + adaptive scaling to prevent blank plots")
print("="*70)

for cfg in configs:
    print(f"\n{'#'*70}")
    print(f"# Configuration: {cfg['name']}")
    print(f"#'*70")
    
    # 1. T vs k₀ (at fixed g)
    plot_T_vs_param_phase_diagram(k0_range, T_range, 'k0', cfg, 
                                 f'{cfg["name"]} (T vs k₀)', 'Momentum scale k₀')
    
    # 2. T vs g (at fixed k₀)
    plot_T_vs_param_phase_diagram(g_range, T_range, 'g', cfg, 
                                 f'{cfg["name"]} (T vs g)', 'Coupling strength g')
    
    # 3. T vs damping (Ohmic: γ₀, Drude: Γ_D)
    d_key = 'gamma0' if cfg['damping'] == 'ohmic' else 'GammaD'
    plot_T_vs_param_phase_diagram(damp_range, T_range, d_key, cfg, 
                                 f'{cfg["name"]} (T vs {d_key})', f'Damping {d_key}')

print("\n" + "="*70)
print("✅ PHASE DIAGRAMS COMPLETE")
print("="*70)
print("\n📖 Interpretation:")
print("   • Red regions (α≈1): Planckian/Quantum Critical regime")
print("   • Blue regions (α≈2): Fermi Liquid regime")
print("   • Contours: α=1 (solid red), α=1.5 (dashed black), α=2 (dotted blue)")
print("\n🔧 If plots still appear blank:")
print("   1. Try INTEGRATOR = 'mc' for faster but noisier results")
print("   2. Increase N_k, N_cos in im_sigma_3d_simpson for better convergence")
print("   3. Adjust g or k₀ ranges to be closer to expected QCP values")

