import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# =============================================================================
# PHYSICAL CONSTANTS & BASE PARAMETERS (Natural Units: hbar = kB = 1)
# =============================================================================
E_F = 1.0
k_F = 1.0
m = 0.5          # E_F = k_F^2/(2m) = 1.0
v_F = k_F / m    # Fermi velocity = 2.0

# =============================================================================
# CORE PHYSICS FUNCTIONS
# =============================================================================
def fermi_dist(E, T):
    return 1.0 / (np.exp(np.clip(E/T, -50, 50)) + 1.0)

def bose_dist(E, T):
    E_abs = np.abs(E)
    return np.where(E_abs < 1e-8, T/1e-8, 1.0 / (np.exp(np.clip(E_abs/T, 0, 50)) - 1.0))

def get_dispersion(q, cfg):
    """Returns Omega_q based on dispersion type"""
    if cfg['dispersion'] == 'constant':
        return cfg['k0']**2/ (2.0 * cfg['M'])
    elif cfg['dispersion'] == 'linear':
        return cfg['c'] * (q/cfg['k0'])
    elif cfg['dispersion'] == 'quadratic':
        return np.sqrt((q)**2 / (2.0 * cfg['M'])+(cfg['k0'])**2 / (2.0 * cfg['M']))

def get_damping(nu, cfg):
    """Returns gamma(nu) based on damping type"""
    if cfg['damping'] == 'ohmic':
        return nu*cfg['gamma0']
    elif cfg['damping'] == 'drude':
        return 1/ (1.0 + 1j*(cfg['GammaD']*nu))

def im_sigma_3d(omega, T, cfg):
    """
    Exact 3D momentum integration of Im Sigma^R at Fermi surface.
    Uses physical scales only: k0, Omega0/c/M, damping params, T.
    """
    q_pts = np.linspace(1e-6, cfg['k0'], 60)
    th_pts = np.linspace(0, np.pi, 30)
    Q, TH = np.meshgrid(q_pts, th_pts, indexing='ij')
    
    # Electron energy relative to Fermi level
    xi = Q**2 / (2*m) - v_F * Q * np.cos(TH) + E_F
    nu = omega - xi  # Energy transfer to bath
    
    # Bath properties
    Omega_q = get_dispersion(Q, cfg)
    gamma_nu = get_damping(nu, cfg)
    
    # Spectral function A_D(nu, q)
    response = 1/(nu**2 - Omega_q**2-1j*nu*gamma_nu)#(nu**2 - Omega_q**2)**2 + (gamma_nu * nu)**2
    A_D = np.imag(response)
    
    # Thermal factors
    n_F = np.tanh(0.5*nu/T)
    n_B = 1/np.tanh(0.5*xi/T)
    # 3D measure: d^3q/(2pi)^3 -> q^2 sin(theta) dq dtheta / (4 pi^2)
    measure = (Q**2 * np.sin(TH)) / (4.0 * np.pi**2)
    integrand = A_D * (n_F + n_B) * measure
    
    # Numerical integration: theta then q
    int_th = np.trapezoid(integrand, th_pts, axis=1)
    res = np.trapezoid(int_th, q_pts, axis=0)
    
    # Retarded self-energy is strictly negative
    return -np.pi * cfg['g']**2 * res

def extract_exponent(T, cfg, window_frac=(0.05, 0.25), n_omega=30):
    """
    Extracts scaling exponent n from |Im Sigma| ~ A * omega^n in IR window.
    """
    # Reference energy scale for window placement
    #E_ref = max(cfg.get('Omega0', 0.0), 
    #            cfg.get('c', 0.0)*cfg['k0'], 
    #            cfg['k0']**2/(2.0*cfg.get('M', 1.0)), 
    #            T, 0.1)
    #E_K0 = cfg['k0']**2/(2.0*cfg.get('M', 1.0))           
    om_low = 0.001*E_F#window_frac[0] * E_ref
    om_high = 2*E_F#window_frac[1] * E_ref
    omega_t = np.linspace(om_low, om_high, n_omega)
    
    # Evaluate self-energy
    im_S = np.array([im_sigma_3d(w, T, cfg) for w in omega_t])
    
    # Mask: physically significant values
    mask = np.abs(im_S) > 1e-16
    if mask.sum() < 5:
        return {'n': np.nan, 'A': np.nan, 'R2': np.nan}
        
    log_om = np.log(omega_t[mask])
    log_imS = np.log(np.abs(im_S[mask]))
    
    # Log-log linear fit
    coeffs = np.polyfit(log_om, log_imS, 1)
    n, log_A = coeffs
    
    # R^2 calculation
    pred = n * log_om + log_A
    ss_res = np.sum((log_imS - pred)**2)
    ss_tot = np.sum((log_imS - log_imS.mean())**2)
    R2 = 1 - ss_res/ss_tot if ss_tot > 0 else np.nan
    
    return {'n': float(n), 'A': float(np.exp(log_A)), 'R2': float(R2)}

# =============================================================================
# CONFIGURATIONS FOR ALL 6 COMBINATIONS
# =============================================================================
configs = [
    {'name': 'Ohmic + Constant',    'damping': 'ohmic',   'dispersion': 'constant',    'gamma0': 0.5, 'Omega0': 0.3, 'M': 0.5, 'k0': 1.0, 'g': 1.0},
    {'name': 'Ohmic + Linear',      'damping': 'ohmic',   'dispersion': 'linear',      'gamma0': 0.5, 'c': 1.0, 'k0': 1.0, 'g': 1.0},
    {'name': 'Ohmic + Quadratic',   'damping': 'ohmic',   'dispersion': 'quadratic',   'gamma0': 0.5, 'M': 0.5, 'k0': 1.0, 'g': 1.0},
    {'name': 'Drude + Constant',    'damping': 'drude',   'dispersion': 'constant',    'GammaD': 0.5, 'omegaD': 10.0, 'M': 0.5, 'Omega0': 0.3, 'k0': 1.0, 'g': 1.0},
    {'name': 'Drude + Linear',      'damping': 'drude',   'dispersion': 'linear',      'GammaD': 0.5, 'omegaD': 10.0, 'c': 1.0, 'k0': 1.0, 'g': 1.0},
    {'name': 'Drude + Quadratic',   'damping': 'drude',   'dispersion': 'quadratic',   'GammaD': 0.5, 'omegaD': 10.0, 'M': 0.5, 'k0': 1.0, 'g': 1.0},
]

# =============================================================================
# PLOTTING SETUP
# =============================================================================
plt.rcParams.update({
    'font.size': 11, 'text.usetex': False, 'font.family': 'serif',
    'axes.linewidth': 1.0, 'lines.linewidth': 1.8, 'figure.dpi': 150,
    'axes.titlesize': 12, 'axes.labelsize': 11, 'legend.fontsize': 9
})

# =============================================================================
# GENERATE DIAGNOSTICS FOR ALL COMBINATIONS
# =============================================================================
for cfg in configs:
    print(f"\n{'='*60}")
    print(f"Processing: {cfg['name']}")
    print(f"{'='*60}")
    
    # --- 1. PHASE DIAGRAM: Damping Strength vs k0 ---
    #damp_vals = np.logspace(-2, 1, 30)
    #k0_vals = np.logspace(-1.0, 1, 30)
    damp_vals = np.linspace(0.001*E_F, 5*E_F, 20)
    #drude_vals = np.linspace(1*E_F, 10*E_F, 30)
    k0_vals = np.linspace(0.01*k_F, 2*k_F, 20)
    T_ref = 0.1
    n_mat = np.zeros((len(damp_vals), len(k0_vals)))
    R2_mat = np.zeros_like(n_mat)
    
    for i, d in enumerate(tqdm(damp_vals, desc='Damping scan')):
        cfg_copy = cfg.copy()
        # Set damping parameter
        if cfg['damping'] == 'ohmic':
            cfg_copy['gamma0'] = d
            for j, k0 in enumerate(k0_vals):
                cfg_copy['k0'] = k0
                res = extract_exponent(T_ref, cfg_copy)
                n_mat[i, j] = res['n']
                R2_mat[i, j] = res['R2']
        elif cfg['damping'] == 'drude':
                cfg_copy['GammaD'] = d
                cfg_copy['gamma0'] = d
                for j, k0 in enumerate(k0_vals):
                    cfg_copy['k0'] = k0
                    res = extract_exponent(T_ref, cfg_copy)
                    n_mat[i, j] = res['n']
                    R2_mat[i, j] = res['R2']
        else:
            cfg_copy['GammaD'] = d
            for j, k0 in enumerate(k0_vals):
                cfg_copy['k0'] = k0
                res = extract_exponent(T_ref, cfg_copy)
                n_mat[i, j] = res['n']
                R2_mat[i, j] = res['R2']
        
            
    K0_grid, Damp_grid = np.meshgrid(k0_vals, damp_vals)
    
    fig1 = plt.figure(figsize=(7.5, 5.5))
    pcm = plt.pcolormesh(K0_grid, Damp_grid, n_mat, shading='auto', 
                         cmap='RdYlGn_r', vmin=0.0, vmax=2.0)
    plt.colorbar(pcm, label='Scaling Exponent n')
    plt.contour(K0_grid, Damp_grid, n_mat, levels=[1.0,1.5,1.8], colors=['blue','black','white'], linewidths=1.0)
    #plt.xscale('log'); plt.yscale('log')
    damp_label = r'$\gamma_0$' if cfg['damping']=='ohmic' else r'$\Gamma_D$'
    plt.xlabel('Momentum scale k0'); plt.ylabel(f'Damping strength {damp_label}')
    plt.title(f'Phase Diagram: {cfg["name"]}\n(T={T_ref}, g=1.0)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # --- 2. TEMPERATURE MAP: n(T) and R2(T) ---
    T_scan = np.logspace(-1, 1.5, 40)
    n_T, R2_T = [], []
    for T in tqdm(T_scan, desc='Temperature scan'):
        res = extract_exponent(T, cfg)
        n_T.append(res['n'])
        R2_T.append(res['R2'])
        
    n_T = np.array(n_T)
    R2_T = np.array(R2_T)
    
    fig2 = plt.figure(figsize=(7.5, 5.5))
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    line1 = ax1.semilogx(T_scan, n_T, 'b-o', lw=2, label='Exponent n')
    ax1.axhline(1.0, color='k', ls='--', lw=1.5, label='MFL (n=1)')
    ax1.axhline(2.0, color='gray', ls=':', lw=1.5, label='Fermi Liquid (n=2)')
    ax1.set_xlabel('Temperature T/E_F')
    ax1.set_ylabel('Scaling Exponent n', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    
    line2 = ax2.semilogx(T_scan, R2_T, 'r-s', lw=2, label='Fit R2')
    ax2.set_ylabel('Quality R^2', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    #ax2.set_ylim(0.7, 1.02)
    ax2.axhline(0.9, color='orange', ls='-.', lw=1.5, label='R2=0.9 threshold')
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    plt.legend(lines, labels, loc='lower right')
    plt.title(f'Temperature Map: {cfg["name"]}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print(f"Phase diagram and temperature map generated for {cfg['name']}.")

print("\nAll combinations processed successfully.")
print("Diagnosis criteria:")
print("  n ~ 1.0  -> Marginal Fermi Liquid")
print("  n ~ 2.0  -> Fermi Liquid")
print("  n < 0.5  -> Bad Metal / Incoherent")
print("  R2 < 0.9 -> Power law breaks down (crossover/cutoff regime)")
