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

# =============================================================================
# CORE PHYSICS FUNCTIONS
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
# FAST NUMERICAL INTEGRATION (Simpson + Monte Carlo)
# =============================================================================
def im_sigma_3d_simpson(omega, T, cfg):
    """Deterministic 2D Simpson integration."""
    q_pts = np.linspace(1e-6, cfg['k0'], 61)  # Odd for Simpson
    th_pts = np.linspace(0, np.pi, 31)
    Q, TH = np.meshgrid(q_pts, th_pts, indexing='ij')
    xi = Q**2 / (2*m) + v_F * Q * np.cos(TH)
    nu = omega - xi

    Omega_q = get_dispersion(Q, cfg)
    gamma_nu = get_damping(nu, cfg)

    # Bosonic spectral function proxy
    response = 1.0 / (Omega_q**2 - nu**2 - 1j * nu * gamma_nu)
    A_D = np.imag(response)

    # Thermal factors
    n_F = np.tanh(0.5 * nu / T)
    n_B = 1.0 / (np.tanh(0.5 * xi / T) + 1e-12)

    measure = (Q**2 * np.sin(TH)) / (4.0 * np.pi**2)
    integrand = A_D * (n_F + n_B) * measure

    int_th = simpson(integrand, th_pts, axis=1)
    return -np.pi * cfg['g']**2 * simpson(int_th, q_pts, axis=0)

def im_sigma_3d_mc(omega, T, cfg, N=1500000):
    """Vectorized Monte Carlo integration."""
    q = np.random.uniform(0, 2*cfg['k0'], N)
    th = np.random.uniform(0, np.pi, N)
    xi = q**2 / (2*m) - v_F * q * np.cos(th) + E_F
    nu = omega - xi

    Omega_q = get_dispersion(q, cfg)
    gamma_nu = get_damping(nu, cfg)

    response = 1.0 / (Omega_q**2 - nu**2 - 1j * nu * gamma_nu)
    A_D = np.imag(response)

    n_F = np.tanh(0.5 * xi / T)
    n_B = 1.0 / (np.tanh(0.5 * nu / T) + 1e-22)

    integrand = A_D * (n_F + n_B) * q**2 * np.sin(th) / (4.0 * np.pi**2)
    vol = (cfg['k0']**3 / 3.0) * 2.0 / (4.0 * np.pi**2) * np.pi
    return -np.pi * cfg['g']**2 * np.mean(integrand) * vol * (4.0 * np.pi**2)

# Select integrator: 'simpson' (accurate) or 'mc' (faster for scans)
INTEGRATOR = 'simpson'
im_sigma_func = im_sigma_3d_simpson if INTEGRATOR == 'simpson' else im_sigma_3d_mc

# =============================================================================
# RESISTIVITY & EXPONENT EXTRACTION
# =============================================================================
#def compute_resistivity(T, cfg, omega_min=5e-10):
#    """DC resistivity proxy: rho(T) ~ |Im Sigma(omega->0, T)|
#    In electron-boson models, this captures the dominant T-dependence 
#    of the transport scattering rate."""
#    return np.abs(im_sigma_func(omega_min, T, cfg))
def compute_resistivity_scaled(T, cfg, omega_max_factor=5.0, N_omega=40):
    """
    DC resistivity with proper physical scaling:
    ρ(T) ∝ (1/v_F²) × ∫ dω [-∂f/∂ω] × ⟨(1-cosθ) ImΣ(ω,T)⟩
    
    Parameters:
    -----------
    T : float
        Temperature
    cfg : dict
        Model parameters
    omega_max_factor : float
        Integrate up to ω_max = omega_max_factor * T (min 1e-3*E_F)
    N_omega : int
        Number of frequency points for integration
    """
    # Thermal window: -df/dω peaks at ω=0 with width ~4T
    omega_max = max(omega_max_factor * T, 1e-3 * E_F)
    omega_pts = np.linspace(1e-6 * E_F, omega_max, N_omega)
    
    # Thermal weighting: -∂f/∂ω = 1/(4T cosh²(ω/2T))
    thermal_weight = 1.0 / (4.0 * T * np.cosh(omega_pts / (2.0 * T))**2 + 1e-12)
    
    # Transport vertex approximation: <1-cosθ> ≈ <q²>/(2k_F²)
    # For isotropic scattering, use average q² from boson dispersion
    q_avg_sq = cfg['k0']**2 / 3.0  # Rough average for q ∈ [0, k0]
    transport_vertex = q_avg_sq / (2.0 * k_F**2)
    
    # Compute ImΣ at each ω with transport weighting
    im_S_vals = np.array([im_sigma_func(w, T, cfg) for w in omega_pts])
    
    # Integrate: ρ ∝ (1/v_F²) × transport_vertex × ∫ dω [-∂f/∂ω] × ω × |ImΣ|
    # Note: Extra ω factor from phase space in electron-boson scattering
    integrand = thermal_weight * omega_pts * np.abs(im_S_vals)
    integral = simpson(integrand, omega_pts)
    
    # Final scaling with Drude prefactor
    rho = (transport_vertex / v_F**2) * integral
    
    return np.clip(rho, 1e-12, 1e6)  # Prevent numerical overflow/underflow
#def extract_local_alpha(T, cfg, delta_frac=0.002):
#    """Extract local exponent alpha where rho(T) ~ T^alpha.
#    Uses a small logarithmic window around T to compute d ln(rho)/d ln T."""
#    dT = max(T * delta_frac, 1e-3)
#    T_pts = np.array([max(T - dT, 1e-4), T, min(T + dT, 2.0)])
#    rho_pts = np.array([compute_resistivity(t, cfg) for t in T_pts])
#    
#    mask = rho_pts > 1e-15
#    if mask.sum() < 2: return np.nan
#    
#    # Linear fit in log-log space
#    alpha = np.polyfit(np.log(T_pts[mask]), np.log(rho_pts[mask]), 1)[0]
#    return alpha
def extract_local_alpha(T, cfg, delta_frac=0.2, rho_func=compute_resistivity_scaled):
    """Extract local exponent α where ρ(T) ~ T^α, using physically-scaled resistivity."""
    dT = max(T * delta_frac, 1e-3)
    T_pts = np.array([max(T - dT, 1e-4), T, min(T + dT, 2.0)])
    rho_pts = np.array([rho_func(t, cfg) for t in T_pts])
    
    mask = rho_pts > 1e-15
    if mask.sum() < 2: return np.nan
    
    # Linear fit in log-log space
    alpha = np.polyfit(np.log(T_pts[mask]), np.log(rho_pts[mask]), 1)[0]
    return np.clip(alpha, -1.0, 4.0)  # Physical bounds
# =============================================================================
# PLOTTING & DIAGRAM GENERATION
# =============================================================================
plt.rcParams.update({
    'font.size': 10, 'text.usetex': False, 'font.family': 'serif',
    'axes.linewidth': 1.0, 'lines.linewidth': 1.5, 'figure.dpi': 150,
    'axes.titlesize': 11, 'axes.labelsize': 10, 'legend.fontsize': 9
})

#def plot_resistivity_phase_diagram(x_vals, T_vals, x_param_name, cfg, title, xlabel):
#    """Generates 1x2 figure: Resistivity Map + Local Exponent Map"""
#    n_x, n_T = len(x_vals), len(T_vals)
#    X, Y = np.meshgrid(x_vals, T_vals)
#    
#    rho_mat = np.zeros((n_T, n_x))
#    alpha_mat = np.zeros((n_T, n_x))#
#
#    for i, T in enumerate(tqdm(T_vals, desc=f'T-scan ({x_param_name})')):
#        for j, x in enumerate(x_vals):
#            cfg_scan = cfg.copy()
 #           cfg_scan[x_param_name] = x
#            rho_mat[i, j] = compute_resistivity(T, cfg_scan)
#            alpha_mat[i, j] = extract_local_alpha(T, cfg_scan)
#
#    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
#    fig.suptitle(f'QCP Resistivity Analysis: {title}', fontsize=12, y=0.95)
#
#    # Panel 1: Raw Resistivity
#    ax1, ax2 = axes
#    vmin1 = np.nanpercentile(rho_mat, 1)
#    vmax1 = np.nanpercentile(rho_mat, 99.9)
#    pcm1 = ax1.pcolormesh(X, Y, rho_mat, shading='auto', cmap='plasma', vmin=vmin1, vmax=vmax1)
#    fig.colorbar(pcm1, ax=ax1, label='Resistivity ρ(T)', fraction=0.046, pad=0.04)
#    ax1.contour(X, Y, rho_mat, levels=6, colors='white', linewidths=0.7, linestyles=':', alpha=0.5)
#    #ax1.set_xscale('log')
#    #ax1.set_yscale('log')
#    ax1.set_xlabel(xlabel)
#    ax1.set_ylabel('Temperature T')
#    ax1.set_title('ρ(T, Control Parameter)')
#    ax1.grid(True, alpha=0.2)
#
#    # Panel 2: Local Exponent α (ρ ~ T^α)
#    pcm2 = ax2.pcolormesh(X, Y, alpha_mat, shading='auto', cmap='RdYlGn_r', vmin=0.0, vmax=2.5)
#    fig.colorbar(pcm2, ax=ax2, label='Exponent α (ρ ~ T^α)', fraction=0.046, pad=0.04)
#    ax2.contour(X, Y, alpha_mat, levels=[1.0, 1.5, 2.0], colors=['blue', 'black', 'red'], 
#                linewidths=1.2, linestyles=['--', '-', '-.'])
#    #ax2.set_xscale('log')
#    #ax2.set_yscale('log')
#    ax2.set_xlabel(xlabel)
#    ax2.set_ylabel('Temperature T')
#    ax2.set_title('Local Scaling Exponent α')
#    ax2.grid(True, alpha=0.2)#
#
#    fig.tight_layout()
#    plt.show()
def plot_resistivity_phase_diagram_scaled(x_vals, T_vals, x_param_name, cfg, title, xlabel):
    """Generates 1x2 figure: Scaled Resistivity Map + Local Exponent Map"""
    n_x, n_T = len(x_vals), len(T_vals)
    X, Y = np.meshgrid(x_vals, T_vals)
    
    rho_mat = np.zeros((n_T, n_x))
    alpha_mat = np.zeros((n_T, n_x))

    for i, T in enumerate(tqdm(T_vals, desc=f'T-scan ({x_param_name})')):
        for j, x in enumerate(x_vals):
            cfg_scan = cfg.copy()
            cfg_scan[x_param_name] = x
            rho_mat[i, j] = compute_resistivity_scaled(T, cfg_scan)
            alpha_mat[i, j] = extract_local_alpha(T, cfg_scan, rho_func=compute_resistivity_scaled)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle(f'QCP Resistivity Analysis (Physical Scaling): {title}', fontsize=12, y=0.95)

    # Panel 1: Scaled Resistivity
    ax1, ax2 = axes
    # Use log scale for resistivity to capture wide dynamic range
    log_rho = np.log10(np.clip(rho_mat, 1e-12, None))
    vmin1, vmax1 = np.nanpercentile(log_rho, 2), np.nanpercentile(log_rho, 98)
    pcm1 = ax1.pcolormesh(X, Y, log_rho, shading='auto', cmap='plasma', vmin=vmin1, vmax=vmax1)
    cbar1 = fig.colorbar(pcm1, ax=ax1, label='log₁₀[ρ(T)] (arb. units)', fraction=0.046, pad=0.04)
    ax1.contour(X, Y, rho_mat, levels=6, colors='white', linewidths=0.7, linestyles=':', alpha=0.5)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel('Temperature T')
    ax1.set_title('Scaled Resistivity ρ(T)')
    ax1.grid(True, alpha=0.2)

    # Panel 2: Local Exponent α (ρ ~ T^α)
    pcm2 = ax2.pcolormesh(X, Y, alpha_mat, shading='auto', cmap='RdYlGn_r', vmin=0.0, vmax=2.5)
    fig.colorbar(pcm2, ax=ax2, label='Exponent α (ρ ~ T^α)', fraction=0.046, pad=0.04)
    ax2.contour(X, Y, alpha_mat, levels=[1.0, 1.5, 2.0], colors=['blue', 'black', 'red'], 
                linewidths=1.2, linestyles=['--', '-', '-.'])
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel('Temperature T')
    ax2.set_title('Local Scaling Exponent α')
    ax2.grid(True, alpha=0.2)

    fig.tight_layout()
    plt.show()
# =============================================================================
# CONFIGURATIONS & MAIN EXECUTION
# =============================================================================
configs = [
    {'name': 'Ohmic + Constant',    'damping': 'ohmic',   'dispersion': 'constant',    'gamma0': 0.5, 'M': 0.5, 'Omega0': 0.3, 'k0': 1.0, 'g': 1.0},
    {'name': 'Ohmic + Linear',      'damping': 'ohmic',   'dispersion': 'linear',      'gamma0': 0.5, 'c': 1.0, 'k0': 1.0, 'g': 1.0},
    {'name': 'Ohmic + Quadratic',   'damping': 'ohmic',   'dispersion': 'quadratic',   'gamma0': 0.5, 'M': 0.5, 'k0': 1.0, 'g': 1.0},
    {'name': 'Drude + Constant',    'damping': 'drude',   'dispersion': 'constant',    'GammaD': 0.5, 'M': 0.5, 'omegaD': 1.0, 'Omega0': 0.3, 'k0': 1.0, 'g': 1.0},
    {'name': 'Drude + Linear',      'damping': 'drude',   'dispersion': 'linear',      'GammaD': 0.5, 'omegaD': 10.0, 'c': 1.0, 'k0': 1.0, 'g': 1.0},
    {'name': 'Drude + Quadratic',   'damping': 'drude',   'dispersion': 'quadratic',   'GammaD': 0.5, 'omegaD': 10.0, 'M': 0.5, 'k0': 1.0, 'g': 1.0},
]

# Scan Ranges (Low-T focus for QCP)
T_range = np.linspace(0.005 * E_F, 1.0 * E_F, 100)
damp_range = np.linspace(0.001 * E_F, 5 * E_F, 100)
k0_range = np.linspace(0.01 * k_F, 2.0 * k_F, 100)
k0_range = np.linspace(0.01*np.sqrt(2*0.5), 2.0*np.sqrt(2*0.5), 100)
omegaD_range = np.linspace(5.0 * E_F, 15.0 * E_F, 100)

for cfg in configs:
    print(f"\n{'='*50}")
    print(f"Processing: {cfg['name']}")
    print(f"{'='*50}")
    
    d_key = 'gamma0' if cfg['damping'] == 'ohmic' else 'GammaD'

    # 1. T vs Damping
    plot_resistivity_phase_diagram_scaled(damp_range, T_range, d_key, cfg, 
                         f'{cfg["name"]} (T vs Damping)', f'Damping strength {d_key}')

    # 2. T vs k0
    plot_resistivity_phase_diagram_scaled(k0_range, T_range, 'k0', cfg, 
                         f'{cfg["name"]} (T vs k0)', 'Momentum scale k0')

    # 3. T vs omegaD (Drude only)
    if cfg['damping'] == 'drude':
        plot_resistivity_phase_diagram_scaled(omegaD_range, T_range, 'omegaD', cfg, 
                             f'{cfg["name"]} (T vs omegaD)', 'Drude frequency omegaD')

print("\n✅ All resistivity phase diagrams generated successfully.")
print("\n📖 Interpretation Guide:")
print("   α = 2.0  -> Fermi Liquid (ρ ~ T²)")
print("   α = 1.0  -> Planckian/MFL (ρ ~ T)")
print("   α < 1.0  -> Sub-linear / Bad Metal")
print("   QCP Signature: A fan-shaped region where α ≈ 1.0 emerges from T→0 at a critical control parameter.")

