import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.integrate import simpson

=============================================================================
PHYSICAL CONSTANTS (Natural Units: hbar = kB = 1)
=============================================================================
E_F = 1.0
k_F = 1.0
m = 0.5          # E_F = k_F^2/(2m) = 1.0
v_F = k_F / m    # Fermi velocity = 2.0

=============================================================================
CORE PHYSICS FUNCTIONS
=============================================================================
def get_dispersion(q, cfg):
    if cfg['dispersion'] == 'constant': 
        return cfg['k0']**2 / (2.0 * cfg['M'])
    elif cfg['dispersion'] == 'linear': 
        return cfg['c'] * (q / cfg['k0'])
    elif cfg['dispersion'] == 'quadratic': 
        return np.sqrt(q**2 / (2.0 * cfg['M']) + cfg['k0']**2 / (2.0 * cfg['M']))

def get_damping(nu, cfg):
    if cfg['damping'] == 'ohmic': 
        return nu * cfg['gamma0']
    elif cfg['damping'] == 'drude': 
        return cfg['GammaD'] / (1.0 + 1j * (nu / cfg['omegaD']))

=============================================================================
FAST NUMERICAL INTEGRATION (Simpson + Monte Carlo)
=============================================================================
def im_sigma_3d_simpson(omega, T, cfg):
    """Deterministic 2D Simpson integration. Accurate and robust."""
    q_pts = np.linspace(1e-6, cfg['k0'], 61)  # Odd for Simpson
    th_pts = np.linspace(0, np.pi, 31)
    Q, TH = np.meshgrid(q_pts, th_pts, indexing='ij')
    #xi = Q**2 / (2*m) - v_F * Q * np.cos(TH) + E_F
    xi = Q**2 / (2*m) + v_F * Q # + E_F
    nu = omega - xi

    Omega_q = get_dispersion(Q, cfg)
    gamma_nu = get_damping(nu, cfg)

    # Spectral function A_D(nu, q)
    response = 1 / (Omega_q**2 - nu**2 - 1j * nu * gamma_nu)
    A_D = np.imag(response)

    # Thermal factors
    n_F = np.tanh(0.5 * nu / T)
    n_B = 1 / (np.tanh(0.5 * xi / T) + 1e-12)

    measure = (Q**2 * np.sin(TH)) / (4.0 * np.pi**2)
    integrand = A_D * (n_F + n_B) * measure

    int_th = simpson(integrand, th_pts, axis=1)
    return -np.pi * cfg['g']**2 * simpson(int_th, q_pts, axis=0)

def im_sigma_3d_mc(omega, T, cfg, N=150000):
    """Vectorized Monte Carlo integration. Faster for coarse parameter scans."""
    q = np.random.uniform(0, cfg['k0'], N)
    th = np.random.uniform(0, np.pi, N)
    xi = q**2 / (2*m) - v_F * q * np.cos(th) + E_F
    nu = omega - xi

    Omega_q = get_dispersion(q, cfg)
    gamma_nu = get_damping(nu, cfg)

    # Spectral function A_D(nu, q)
    response = 1 / (nu**2 - Omega_q**2 - 1j * nu * gamma_nu)
    A_D = np.imag(response)

    # Thermal factors
    n_F = np.tanh(0.5 * nu / T)
    n_B = 1 / (np.tanh(0.5 * xi / T) + 1e-12)

    # Jacobian-weighted MC average
    integrand = A_D * (n_F + n_B) * q**2 * np.sin(th) / (4.0 * np.pi**2)
    vol = (cfg['k0']**3 / 3.0) * 2.0 / (4.0 * np.pi**2) * np.pi  # Effective volume factor
    return -np.pi * cfg['g']**2 * np.mean(integrand) * vol * (4.0 * np.pi**2)

# Select integrator: 'simpson' (accurate) or 'mc' (faster for scans)
INTEGRATOR = 'simpson'
im_sigma_func = im_sigma_3d_simpson if INTEGRATOR == 'simpson' else im_sigma_3d_mc

=============================================================================
EXPONENT EXTRACTION (QCP-OPTIMIZED)
=============================================================================
def extract_exponent(T, cfg, n_omega=50):
    # Adaptive frequency window for QCP studies:
    # Lower bound tracks T to avoid numerical noise at T->0
    # Upper bound stays within the relevant electronic scale
    om_low = max(1e-4 * E_F, T / 5.0)
    om_high = min(2.0 * E_F, max(T * 10.0, 0.5 * E_F))
    omega_t = np.linspace(om_low, om_high, n_omega)
    
    im_S = np.array([im_sigma_func(w, T, cfg) for w in omega_t])
    mask = np.abs(im_S) > 1e-16
    if mask.sum() < 5: return {'n': np.nan, 'A': np.nan, 'R2': np.nan}

    log_om = np.log(omega_t[mask])
    log_imS = np.log(np.abs(im_S[mask]))
    n, log_A = np.polyfit(log_om, log_imS, 1)

    pred = n * log_om + log_A
    ss_res = np.sum((log_imS - pred)**2)
    ss_tot = np.sum((log_imS - log_imS.mean())**2)
    R2 = 1 - ss_res/ss_tot if ss_tot > 0 else np.nan
    return {'n': float(n), 'A': float(np.exp(log_A)), 'R2': float(R2)}

=============================================================================
PLOTTING & DIAGRAM GENERATION
=============================================================================
plt.rcParams.update({
    'font.size': 11, 'text.usetex': False, 'font.family': 'serif',
    'axes.linewidth': 1.0, 'lines.linewidth': 1.8, 'figure.dpi': 150,
    'axes.titlesize': 12, 'axes.labelsize': 11, 'legend.fontsize': 9
})

def plot_T_phase_diagram(x_vals, T_vals, x_param_name, cfg, title, xlabel):
    """Generates T vs Control Parameter phase diagrams."""
    n_mat = np.zeros((len(T_vals), len(x_vals)))
    X, Y = np.meshgrid(x_vals, T_vals)  # X: control param, Y: T

    for i, T in enumerate(tqdm(T_vals, desc=f'T-scan ({x_param_name})')):
        for j, x in enumerate(x_vals):
            cfg_scan = cfg.copy()
            cfg_scan[x_param_name] = x
            res = extract_exponent(T, cfg_scan)
            n_mat[i, j] = res['n']
            
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    pcm = ax.pcolormesh(X, Y, n_mat, shading='auto', cmap='RdYlGn_r', vmin=0.0, vmax=2.0)
    cbar = plt.colorbar(pcm, ax=ax, label='Scaling Exponent n')
    ax.contour(X, Y, n_mat, levels=[1.0, 1.5, 1.8], colors=['blue', 'black', 'white'], 
               linewidths=1.2, linestyles=['--', '-', '-.'])
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Temperature T')
    ax.set_title(f'Phase Diagram: {title}')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()

=============================================================================
CONFIGURATIONS & MAIN EXECUTION
=============================================================================
configs = [
    {'name': 'Ohmic + Constant',    'damping': 'ohmic',   'dispersion': 'constant',    'gamma0': 0.5, 'M': 0.5, 'Omega0': 0.3, 'k0': 1.0, 'g': 1.0},
    {'name': 'Ohmic + Linear',      'damping': 'ohmic',   'dispersion': 'linear',      'gamma0': 0.5, 'c': 1.0, 'k0': 1.0, 'g': 1.0},
    {'name': 'Ohmic + Quadratic',   'damping': 'ohmic',   'dispersion': 'quadratic',   'gamma0': 0.5, 'M': 0.5, 'k0': 1.0, 'g': 1.0},
    {'name': 'Drude + Constant',    'damping': 'drude',   'dispersion': 'constant',    'GammaD': 0.5, 'M': 0.5, 'omegaD': 1.0, 'Omega0': 0.3, 'k0': 1.0, 'g': 1.0},
    {'name': 'Drude + Linear',      'damping': 'drude',   'dispersion': 'linear',      'GammaD': 0.5, 'omegaD': 1.0, 'c': 1.0, 'k0': 1.0, 'g': 1.0},
    {'name': 'Drude + Quadratic',   'damping': 'drude',   'dispersion': 'quadratic',   'GammaD': 0.5, 'omegaD': 1.0, 'M': 0.5, 'k0': 1.0, 'g': 1.0},
]

# Scan Ranges
T_range = np.linspace(0.005 * E_F, 1.0 * E_F, 35)          # Low-T focus for QCP
damp_range = np.linspace(0.001 * E_F, 5 * E_F, 35)
k0_range = np.linspace(0.01 * k_F, 2 * k_F, 35)
omegaD_range = np.linspace(0.1 * E_F, 5 * E_F, 35)         # Fixed ascending order

for cfg in configs:
    print(f"\n{'='*50}")
    print(f"Processing: {cfg['name']} (Integration: {INTEGRATOR})")
    print(f"{'='*50}")
    
    d_key = 'gamma0' if cfg['damping'] == 'ohmic' else 'GammaD'

    # 1. T vs Damping
    plot_T_phase_diagram(damp_range, T_range, d_key, cfg, 
                         f'{cfg["name"]} (T vs Damping)', f'Damping strength {d_key}')

    # 2. T vs k0
    plot_T_phase_diagram(k0_range, T_range, 'k0', cfg, 
                         f'{cfg["name"]} (T vs k0)', 'Momentum scale k0')

    # 3. T vs omegaD (Drude only)
    if cfg['damping'] == 'drude':
        plot_T_phase_diagram(omegaD_range, T_range, 'omegaD', cfg, 
                             f'{cfg["name"]} (T vs omegaD)', 'Drude frequency omegaD')

print("\nAll phase diagrams generated successfully.")
print("QCP Interpretation Guide:")
print("  n ~ 2.0  -> Fermi Liquid (FL)")
print("  n ~ 1.0  -> Marginal Fermi Liquid (MFL)")
print("  n < 0.5  -> Bad Metal / Non-Fermi Liquid")
print("  Look for sharp boundaries or fan-shaped regions emerging as T -> 0.")
