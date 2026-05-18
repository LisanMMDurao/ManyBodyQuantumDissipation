import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.integrate import simpson

# ---------- GPU ACCELERATION ----------
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("✓ CuPy found – using GPU for Monte Carlo integration")
except ImportError:
    cp = np
    GPU_AVAILABLE = False
    print("⚠ CuPy not found – falling back to CPU (slower)")

# =============================================================================
# PHYSICAL CONSTANTS (Natural Units: hbar = kB = 1)
# =============================================================================
E_F = 1.0
k_F = 1.0
m = 0.5          # E_F = k_F^2/(2m) = 1.0
v_F = k_F / m    # Fermi velocity = 2.0

# =============================================================================
# CORE PHYSICS FUNCTIONS (GPU‑capable versions using cupy)
# =============================================================================
def get_dispersion(q, cfg, xp):
    """Returns boson dispersion Omega_q. xp is either numpy or cupy."""
    if cfg['dispersion'] == 'constant':
        return xp.full_like(q, cfg['k0']**2 / (2.0 * cfg['M']))
    elif cfg['dispersion'] == 'linear':
        return cfg['c'] * q / cfg['k0']
    elif cfg['dispersion'] == 'quadratic':
        return xp.sqrt(q**2 / (2.0 * cfg['M']) + cfg['k0']**2 / (2.0 * cfg['M']))

def get_damping(nu, cfg, xp):
    """Damping rate gamma(nu). Returns a complex array if nu is complex."""
    if cfg['damping'] == 'ohmic':
        return nu * cfg['gamma0']
    elif cfg['damping'] == 'drude':
        return cfg['GammaD'] / (1.0 + 1j * (nu / cfg['omegaD']))

# -----------------------------------------------------------------------------
# GPU MONTE CARLO INTEGRATOR (single (omega, T) point)
# -----------------------------------------------------------------------------
def im_sigma_3d_mc_gpu(omega, T, cfg, N=500000):
    """
    Compute Im[Sigma(omega, T)] using Monte Carlo on GPU.
    Returns a float (CPU value).
    """
    xp = cp if GPU_AVAILABLE else np
    # Move constants to GPU
    omega_g = xp.asarray(omega, dtype=xp.float64)
    T_g = xp.asarray(T, dtype=xp.float64)
    
    # Random samples
    q = xp.random.uniform(0, 2 * cfg['k0'], N)
    th = xp.random.uniform(0, np.pi, N)
    
    # Compute xi = ε_{k+q} - ε_k
    xi = q**2 / (2*m) + v_F * q * xp.cos(th) #+ E_F
    nu = omega_g - xi
    
    Omega_q = get_dispersion(q, cfg, xp)
    gamma_nu = get_damping(nu, cfg, xp)
    
    # Spectral function A_D = Im[ 1/( Omega_q^2 - nu^2 - i nu gamma_nu ) ]
    denom = Omega_q**2 - nu**2 - 1j * nu * gamma_nu
    A_D = xp.imag(1.0 / denom)
    
    # Thermal factors
    n_F = xp.tanh(0.5 * xi / T_g)
    n_B = 1.0 / (xp.tanh(0.5 * nu / T_g) + 1e-22)
    
    # Integrand and Monte Carlo average
    measure = q**2 * xp.sin(th) / (4.0 * np.pi**2)
    integrand = A_D * (n_F + n_B) * measure
    
    # Volume factor: ∫ dq dθ q² sinθ over q∈[0,k0], θ∈[0,π] = (k0³/3)*2
    vol = (cfg['k0']**3 / 3.0) * 2.0
    integral = xp.mean(integrand) * vol
    
    # Return to CPU as scalar
    result = -np.pi * cfg['g']**2 * float(cp.asnumpy(integral) if GPU_AVAILABLE else integral)
    return result

# -----------------------------------------------------------------------------
# CPU FALLBACK (original NumPy version)
# -----------------------------------------------------------------------------
def im_sigma_3d_mc_cpu(omega, T, cfg, N=500000):
    """CPU Monte Carlo – used when GPU is not available."""
    q = np.random.uniform(0, 2*cfg['k0'], N)
    th = np.random.uniform(0, np.pi, N)
    xi = q**2 / (2*m) - v_F * q * np.cos(th) + E_F
    nu = omega - xi
    Omega_q = get_dispersion(q, cfg, np)
    gamma_nu = get_damping(nu, cfg, np)
    denom = Omega_q**2 - nu**2 - 1j * nu * gamma_nu
    A_D = np.imag(1.0 / denom)
    n_F = np.tanh(0.5 * xi / T)
    n_B = 1.0 / (np.tanh(0.5 * nu / T) + 1e-22)
    integrand = A_D * (n_F + n_B) * q**2 * np.sin(th) / (4.0 * np.pi**2)
    vol = (cfg['k0']**3 / 3.0) * 2.0
    integral = np.mean(integrand) * vol
    return -np.pi * cfg['g']**2 * integral

# -----------------------------------------------------------------------------
# Integrator selector
# -----------------------------------------------------------------------------
if GPU_AVAILABLE:
    im_sigma_mc = im_sigma_3d_mc_gpu
else:
    im_sigma_mc = im_sigma_3d_mc_cpu

# Note: Simpson integrator (CPU only) – we keep it for completeness but Monte Carlo is used by default.
def im_sigma_3d_simpson(omega, T, cfg):
    """Deterministic 2D Simpson integration (CPU only)."""
    q_pts = np.linspace(1e-6, cfg['k0'], 61)
    th_pts = np.linspace(0, np.pi, 31)
    Q, TH = np.meshgrid(q_pts, th_pts, indexing='ij')
    xi = Q**2 / (2*m) + v_F * Q * np.cos(TH)
    nu = omega - xi
    Omega_q = get_dispersion(Q, cfg, np)
    gamma_nu = get_damping(nu, cfg, np)
    response = 1.0 / (Omega_q**2 - nu**2 - 1j * nu * gamma_nu)
    A_D = np.imag(response)
    n_F = np.tanh(0.5 * nu / T)
    n_B = 1.0 / (np.tanh(0.5 * xi / T) + 1e-12)
    measure = (Q**2 * np.sin(TH)) / (4.0 * np.pi**2)
    integrand = A_D * (n_F + n_B) * measure
    int_th = simpson(integrand, th_pts, axis=1)
    return -np.pi * cfg['g']**2 * simpson(int_th, q_pts, axis=0)

# We'll use Monte Carlo for GPU – Simpson is too heavy to port.
im_sigma_func = im_sigma_mc

# =============================================================================
# RESISTIVITY & EXPONENT EXTRACTION (unchanged logic)
# =============================================================================
def compute_resistivity(T, cfg, omega_min=5e-10):
    """DC resistivity proxy: rho(T) ~ |Im Sigma(omega->0, T)|"""
    return np.abs(im_sigma_func(omega_min, T, cfg))

def extract_local_alpha(T, cfg, delta_frac=0.002):
    """Local exponent alpha where rho(T) ~ T^alpha."""
    dT = max(T * delta_frac, 1e-3)
    T_pts = np.array([max(T - dT, 1e-4), T, min(T + dT, 2.0)])
    rho_pts = np.array([compute_resistivity(t, cfg) for t in T_pts])
    mask = rho_pts > 1e-15
    if mask.sum() < 2:
        return np.nan
    alpha = np.polyfit(np.log(T_pts[mask]), np.log(rho_pts[mask]), 1)[0]
    return alpha

# =============================================================================
# PLOTTING (unchanged)
# =============================================================================
plt.rcParams.update({
    'font.size': 10, 'text.usetex': False, 'font.family': 'serif',
    'axes.linewidth': 1.0, 'lines.linewidth': 1.5, 'figure.dpi': 150,
    'axes.titlesize': 11, 'axes.labelsize': 10, 'legend.fontsize': 9
})

def plot_resistivity_phase_diagram(x_vals, T_vals, x_param_name, cfg, title, xlabel):
    """Generates 1x2 figure: Resistivity Map + Local Exponent Map"""
    n_x, n_T = len(x_vals), len(T_vals)
    X, Y = np.meshgrid(x_vals, T_vals)
    
    rho_mat = np.zeros((n_T, n_x))
    alpha_mat = np.zeros((n_T, n_x))

    for i, T in enumerate(tqdm(T_vals, desc=f'T-scan ({x_param_name})')):
        for j, x in enumerate(x_vals):
            cfg_scan = cfg.copy()
            cfg_scan[x_param_name] = x
            rho_mat[i, j] = compute_resistivity(T, cfg_scan)
            alpha_mat[i, j] = extract_local_alpha(T, cfg_scan)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle(f'QCP Resistivity Analysis: {title}', fontsize=12, y=0.95)

    # Panel 1: Raw Resistivity
    ax1, ax2 = axes
    vmin1 = np.nanpercentile(rho_mat, 1)
    vmax1 = np.nanpercentile(rho_mat, 99.9)
    pcm1 = ax1.pcolormesh(X, Y, rho_mat, shading='auto', cmap='plasma', vmin=vmin1, vmax=vmax1)
    fig.colorbar(pcm1, ax=ax1, label='Resistivity ρ(T)', fraction=0.046, pad=0.04)
    ax1.contour(X, Y, rho_mat, levels=6, colors='white', linewidths=0.7, linestyles=':', alpha=0.5)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel('Temperature T')
    ax1.set_title('ρ(T, Control Parameter)')
    ax1.grid(True, alpha=0.2)

    # Panel 2: Local Exponent α
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
T_range = np.linspace(0.005 * E_F, 1.0 * E_F, 40)      # reduced for faster runtime; adjust as needed
damp_range = np.linspace(0.001 * E_F, 5 * E_F, 40)
k0_range = np.linspace(0.01 * k_F, 2.0 * k_F, 40)
omegaD_range = np.linspace(5.0 * E_F, 15.0 * E_F, 40)

for cfg in configs:
    print(f"\n{'='*50}")
    print(f"Processing: {cfg['name']} (GPU: {GPU_AVAILABLE})")
    print(f"{'='*50}")
    
    d_key = 'gamma0' if cfg['damping'] == 'ohmic' else 'GammaD'

    # 1. T vs Damping
    plot_resistivity_phase_diagram(damp_range, T_range, d_key, cfg, 
                         f'{cfg["name"]} (T vs Damping)', f'Damping strength {d_key}')

    # 2. T vs k0
    plot_resistivity_phase_diagram(k0_range, T_range, 'k0', cfg, 
                         f'{cfg["name"]} (T vs k0)', 'Momentum scale k0')

    # 3. T vs omegaD (Drude only)
    if cfg['damping'] == 'drude':
        plot_resistivity_phase_diagram(omegaD_range, T_range, 'omegaD', cfg, 
                             f'{cfg["name"]} (T vs omegaD)', 'Drude frequency omegaD')

print("\n✅ All resistivity phase diagrams generated successfully.")
print("\n📖 Interpretation Guide:")
print("   α = 2.0  -> Fermi Liquid (ρ ~ T²)")
print("   α = 1.0  -> Planckian/MFL (ρ ~ T)")
print("   α < 1.0  -> Sub-linear / Bad Metal")
print("   QCP Signature: A fan-shaped region where α ≈ 1.0 emerges from T→0 at a critical control parameter.")
