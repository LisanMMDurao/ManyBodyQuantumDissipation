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
    if cfg['dispersion'] == 'constant': return cfg['k0']**2 / (2.0 * cfg['M'])
    elif cfg['dispersion'] == 'linear': return cfg['c'] * q/cfg['k0']
    elif cfg['dispersion'] == 'quadratic': return np.sqrt(q**2 / (2.0 * cfg['M'])+cfg['k0']**2 / (2.0 * cfg['M']))

def get_damping(nu, cfg):
    if cfg['damping'] == 'ohmic': return nu*cfg['gamma0']
    elif cfg['damping'] == 'drude': return cfg['GammaD'] / (1.0 + 1j*(nu / cfg['omegaD']))

# =============================================================================
# FAST NUMERICAL INTEGRATION (Simpson + Monte Carlo)
# =============================================================================
def im_sigma_3d_simpson(omega, T, cfg):
    """Deterministic 2D Simpson integration. Accurate and robust."""
    q_pts = np.linspace(1e-6, cfg['k0'], 61)  # Odd for Simpson
    th_pts = np.linspace(0, np.pi, 31)
    Q, TH = np.meshgrid(q_pts, th_pts, indexing='ij')
    
    xi = Q**2 / (2*m) - v_F * Q * np.cos(TH) + E_F
    nu = omega - xi
    
    Omega_q = get_dispersion(Q, cfg)
    gamma_nu = get_damping(nu, cfg)
    
    # Spectral function A_D(nu, q)
    response = 1/(Omega_q**2 - nu**2 -1j*nu*gamma_nu)#(nu**2 - Omega_q**2)**2 + (gamma_nu * nu)**2
    A_D = np.imag(response)
    
    # Thermal factors
    n_F = np.tanh(0.5*nu/T)
    n_B = 1/(np.tanh(0.5*xi/T)+1e-12)
    
    measure = (Q**2 * np.sin(TH)) / (4.0 * np.pi**2)
    integrand = A_D * (n_F + n_B) * measure
    
    int_th = simpson(integrand, th_pts, axis=1)
    return -np.pi * cfg['g']**2 * simpson(int_th, q_pts, axis=0)

def im_sigma_3d_mc(omega, T, cfg, N=150000):
    """Vectorized Monte Carlo integration. Faster for coarse parameter scans."""
    q = np.random.uniform(0, cfg['k0'], N)
    th = np.random.uniform(0, np.pi, N)
    
    #xi = q**2 / (2*m) - v_F * q * np.cos(th) + E_F
    xi = q**2 / (2*m) + v_F * q  #+ E_F
    nu = omega - xi
    
    Omega_q = get_dispersion(q, cfg)
    gamma_nu = get_damping(nu, cfg)
    
    # Spectral function A_D(nu, q)
    response = 1/(nu**2 - Omega_q**2-1j*nu*gamma_nu)#(nu**2 - Omega_q**2)**2 + (gamma_nu * nu)**2
    A_D = np.imag(response)
    
    # Thermal factors
    n_F = np.tanh(0.5*nu/T)
    n_B = 1/(np.tanh(0.5*xi/T)+1e-12)
    
    # Jacobian-weighted MC average
    integrand = A_D * (n_F + n_B) * q**2 * np.sin(th) / (4.0 * np.pi**2)
    vol = (cfg['k0']**3 / 3.0) * 2.0 / (4.0 * np.pi**2) * np.pi  # Effective volume factor
    return -np.pi * cfg['g']**2 * np.mean(integrand) * vol * (4.0 * np.pi**2)

# Select integrator: 'simpson' (accurate) or 'mc' (faster for scans)
INTEGRATOR = 'simpson'
im_sigma_func = im_sigma_3d_simpson if INTEGRATOR == 'simpson' else im_sigma_3d_mc

# =============================================================================
# EXPONENT EXTRACTION
# =============================================================================
def extract_exponent(T, cfg, window_frac=(0.05, 0.25), n_omega=50):
    #E_ref = max(cfg.get('Omega0', 0.0), cfg.get('c', 0.0)*cfg['k0'], 
    #            cfg['k0']**2/(2.0*cfg.get('M', 1.0)), T, 0.1)
    #omega_t = np.linspace(window_frac[0]*E_ref, window_frac[1]*E_ref, n_omega)
    om_low = 0.001*E_F#window_frac[0] * E_ref
    om_high = 2*E_F#window_frac[1] * E_ref
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

# =============================================================================
# PLOTTING & DIAGRAM GENERATION
# =============================================================================
plt.rcParams.update({
    'font.size': 11, 'text.usetex': False, 'font.family': 'serif',
    'axes.linewidth': 1.0, 'lines.linewidth': 1.8, 'figure.dpi': 150,
    'axes.titlesize': 12, 'axes.labelsize': 11, 'legend.fontsize': 9
})

def plot_phase_diagram(x_vals, y_vals, scan_func, cfg, title, xlabel, ylabel):
    n_mat = np.zeros((len(y_vals), len(x_vals)))
    X, Y = np.meshgrid(x_vals, y_vals)
    
    for i, y in enumerate(tqdm(y_vals, desc='Y-scan')):
        for j, x in enumerate(x_vals):
            cfg_scan = cfg.copy()
            cfg_scan[scan_func['x_param']] = x
            cfg_scan[scan_func['y_param']] = y
            res = extract_exponent(cfg.get('T_ref', 0.1), cfg_scan)
            n_mat[i, j] = res['n']
            
    fig = plt.figure(figsize=(7.5, 5.5))
    pcm = plt.pcolormesh(X, Y, n_mat, shading='auto', cmap='RdYlGn_r', vmin=0.0, vmax=2.0)
    plt.colorbar(pcm, label='Scaling Exponent n')
    plt.contour(X, Y, n_mat, levels=[1.0,1.5,1.8], colors=['blue','black','white'], linewidths=1.0)
    #plt.xscale('log'); plt.yscale('log')
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.title(f'Phase Diagram: {title}\n(T={cfg.get("T_ref", 0.1)})')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# =============================================================================
# CONFIGURATIONS
# =============================================================================
configs = [
    {'name': 'Ohmic + Constant',    'damping': 'ohmic',   'dispersion': 'constant',    'gamma0': 0.5, 'M': 0.5,'Omega0': 0.3, 'k0': 1.0, 'g': 1.0},
    {'name': 'Ohmic + Linear',      'damping': 'ohmic',   'dispersion': 'linear',      'gamma0': 0.5, 'c': 1.0, 'k0': 1.0, 'g': 1.0},
    {'name': 'Ohmic + Quadratic',   'damping': 'ohmic',   'dispersion': 'quadratic',   'gamma0': 0.5, 'M': 0.5, 'k0': 1.0, 'g': 1.0},
    {'name': 'Drude + Constant',    'damping': 'drude',   'dispersion': 'constant',    'GammaD': 0.5, 'M': 0.5,'omegaD': 1.0, 'Omega0': 0.3, 'k0': 1.0, 'g': 1.0},
    {'name': 'Drude + Linear',      'damping': 'drude',   'dispersion': 'linear',      'GammaD': 0.5, 'omegaD': 1.0, 'c': 1.0, 'k0': 1.0, 'g': 1.0},
    {'name': 'Drude + Quadratic',   'damping': 'drude',   'dispersion': 'quadratic',   'GammaD': 0.5, 'omegaD': 1.0, 'M': 0.5, 'k0': 1.0, 'g': 1.0},
]

damp_range = np.linspace(0.001*E_F, 5*E_F, 50)
k0_range = np.linspace(0.01*k_F, 2*k_F, 50)

# =============================================================================
# MAIN EXECUTION
# =============================================================================
for cfg in configs:
    cfg['T_ref'] = 0.1
    print(f"\n{'='*50}")
    print(f"Processing: {cfg['name']} (Integration: {INTEGRATOR})")
    print(f"{'='*50}")
    
    # 1. Standard Phase Diagram: Damping vs k0
    d_key = 'gamma0' if cfg['damping']=='ohmic' else 'GammaD'
    plot_phase_diagram(k0_range, damp_range, {'x_param':'k0', 'y_param':d_key}, 
                       cfg, f'{cfg["name"]} (Damping vs k0)', 
                       'Momentum scale k0', f'Damping strength {d_key}')
    
    # 2. DRUDE-SPECIFIC: omegaD vs k0
    if cfg['damping'] == 'drude':
        omegaD_range = np.linspace(5*E_F, 1.0*E_F, 505)
        plot_phase_diagram(k0_range, omegaD_range, {'x_param':'k0', 'y_param':'omegaD'}, 
                           cfg, f'{cfg["name"]} (omegaD vs k0)', 
                           'Momentum scale k0', 'Drude frequency omegaD')
        
        # 3. DRUDE-SPECIFIC: omegaD vs GammaD
        plot_phase_diagram(damp_range, omegaD_range, {'x_param':'GammaD', 'y_param':'omegaD'}, 
                           cfg, f'{cfg["name"]} (omegaD vs GammaD)', 
                           'Damping strength GammaD', 'Drude frequency omegaD')

print("\nAll phase diagrams generated successfully.")
print("Criteria: n ~ 1.0 -> MFL | n ~ 2.0 -> Fermi Liquid | n < 0.5 -> Bad Metal")
