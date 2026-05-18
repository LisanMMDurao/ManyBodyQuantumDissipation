import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# PHYSICAL CONSTANTS (Natural Units)
# =============================================================================
E_F = 1.0
k_F = 1.0
m = 0.5          # E_F = k_F^2/(2m) = 1.0
v_F = k_F / m    # Fermi velocity = 2.0

# =============================================================================
# ANALYTICAL QCP FROM RESISTIVITY SCALING
# =============================================================================

# Static Lindhard polarization magnitude at q_c = 2k_F
N_0 = m * k_F / (2.0 * np.pi**2)  # DOS at Fermi level
ABS_PI_0 = N_0 / 2.0  # |Π₀(2k_F)| = (m k_F)/(4π²)

def get_omega_sq_at_qc(k0, cfg):
    """Compute Ω²(q_c) at q_c = 2k_F for given dispersion."""
    q_c = 2.0 * k_F
    
    if cfg['dispersion'] == 'constant':
        return (k0**2 / (2.0 * cfg['M']))**2
    elif cfg['dispersion'] == 'linear':
        return (cfg['c'] * q_c / k0)**2
    elif cfg['dispersion'] == 'quadratic':
        return (q_c**2 + k0**2) / (2.0 * cfg['M'])
    return np.nan

def mass_parameter(k0, g, cfg):
    """Mass r = Ω²(q_c) - g²|Π₀|: r>0 (FL), r=0 (QCP), r<0 (ordered)."""
    return get_omega_sq_at_qc(k0, cfg) - g**2 * ABS_PI_0

def crossover_temperature(k0, g, cfg):
    """Estimate T_cross ~ |r|^{1/2} where ρ(T) changes from T² to T scaling."""
    r = mass_parameter(k0, g, cfg)
    return np.sqrt(np.abs(r)) / E_F  # Normalize to E_F

def resistivity_scaling_exponent(k0, g, T, cfg):
    """
    Analytical estimate of α = d lnρ / d ln T from scaling theory.
    
    α = 2  (Fermi liquid)  if T << |r|^{1/2}
    α = 1  (Planckian)     if T >> |r|^{1/2}
    Smooth crossover via tanh interpolation.
    """
    T_cross = crossover_temperature(k0, g, cfg)
    if T_cross < 1e-6:  # At exact QCP
        return 1.0
    # Smooth interpolation: α = 2 - tanh((T - T_cross)/Δ)
    delta = 0.1 * T_cross  # Crossover width
    alpha = 2.0 - np.tanh((T - T_cross) / delta)
    return np.clip(alpha, 0.5, 2.0)

def solve_k0_critical_from_resistivity(g, cfg):
    """Find k₀^c where r=0 (resistivity scaling changes at T→0)."""
    def equation(k0):
        return mass_parameter(k0, g, cfg)
    
    # Initial guess based on dispersion
    if cfg['dispersion'] == 'constant':
        guess = (4 * cfg['M']**2 * g**2 * ABS_PI_0)**0.25
    elif cfg['dispersion'] == 'linear':
        guess = cfg['c'] * 2*k_F / (g * np.sqrt(ABS_PI_0))
    else:
        guess = 1.0
    
    # Newton-Raphson iteration
    k0 = guess
    for _ in range(50):
        f = equation(k0)
        # Numerical derivative
        dk0 = 1e-6 * k0
        df = (equation(k0 + dk0) - f) / dk0
        if abs(df) < 1e-12: break
        k0_new = k0 - f / df
        if abs(k0_new - k0) < 1e-8: break
        k0 = max(k0_new, 1e-6)
    
    return k0 if abs(equation(k0)) < 1e-4 else np.nan

def solve_g_critical_from_resistivity(k0, cfg):
    """Find g^c where r=0."""
    om_sq = get_omega_sq_at_qc(k0, cfg)
    if om_sq < 0: return np.nan
    return np.sqrt(om_sq / ABS_PI_0)

# =============================================================================
# PLOTTING: Analytical Resistivity-Based Phase Diagrams
# =============================================================================
plt.rcParams.update({
    'font.size': 11, 'text.usetex': False, 'font.family': 'serif',
    'axes.linewidth': 1.0, 'lines.linewidth': 1.5, 'figure.dpi': 150,
    'axes.titlesize': 12, 'axes.labelsize': 11, 'legend.fontsize': 9
})

def plot_resistivity_analytical_phase_diagram(cfg, title_suffix):
    """Plot T vs k₀ and T vs g phase diagrams from resistivity scaling theory."""
    g_fix = cfg.get('g', 1.0)
    k0_fix = cfg.get('k0', 1.0)
    
    # Compute QCP locations
    k0_c = solve_k0_critical_from_resistivity(g_fix, cfg)
    g_c = solve_g_critical_from_resistivity(k0_fix, cfg)
    
    # Parameter grids
    k0_range = np.linspace(0.01, 3.0 * k_F, 2000)
    g_range = np.linspace(0.1, 15.5, 2000)
    T_range_plot = np.linspace(0.001, 2.0, 2000)  # For color maps
    
    # Create meshgrids for color plots
    K0, T_mesh = np.meshgrid(k0_range, T_range_plot)
    G, T_mesh_g = np.meshgrid(g_range, T_range_plot)
    
    # Compute scaling exponent α(T, k₀) and α(T, g)
    alpha_k0 = np.zeros_like(K0)
    alpha_g = np.zeros_like(G)
    
    for i, T in enumerate(T_range_plot):
        for j, k0 in enumerate(k0_range):
            alpha_k0[i, j] = resistivity_scaling_exponent(k0, g_fix, T, cfg)
        for j, g in enumerate(g_range):
            alpha_g[i, j] = resistivity_scaling_exponent(k0_fix, g, T, cfg)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle(f'Analytical Resistivity Phase Diagrams: {cfg["name"]} ({title_suffix})', 
                 fontsize=13, y=0.95)
    
    # --- Panel 1: T vs k₀ ---
    ax1 = axes[0]
    pcm1 = ax1.pcolormesh(K0/k_F, T_mesh/E_F, alpha_k0, shading='auto', 
                          cmap='RdYlGn_r', vmin=0.5, vmax=2.0)
    fig.colorbar(pcm1, ax=ax1, label='Resistivity Exponent α (ρ ~ T^α)')
    
    # Contours for key regimes
    ax1.contour(K0/k_F, T_mesh/E_F, alpha_k0, levels=[1.0, 1.5, 2.0], 
                colors=['red', 'black', 'blue'], linewidths=1.2, linestyles=['-', '--', ':'])
    
    # Crossover line T_cross(k₀)
    T_cross_k0 = np.array([crossover_temperature(k, g_fix, cfg) for k in k0_range])
    ax1.plot(k0_range/k_F, T_cross_k0/E_F, 'white', linestyle='--', linewidth=2.0, 
             label=r'Crossover $T^* \sim |r|^{1/2}$')
    
    # Mark QCP
    if np.isfinite(k0_c):
        ax1.axvline(k0_c/k_F, color='yellow', linestyle='-.', linewidth=2.5, 
                    label=f'QCP: $k_0^c$ = {k0_c/k_F:.2f} $k_F$')
        #ax1.text(k0_c/k_F, 0.02, 'QCP', color='yellow', ha='center', va='bottom', 
        #         fontweight='bold', fontsize=10)
    
    ax1.set_xlabel('Momentum Scale $k_0 / k_F$')
    ax1.set_ylabel('Temperature $T / E_F$')
    ax1.set_title(f'T vs $k_0$ (Fixed $g={g_fix}$)')
    ax1.set_xlim(0.0, 1.0)
    ax1.set_ylim(0, 1.0)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=9)
    
    # --- Panel 2: T vs g ---
    ax2 = axes[1]
    pcm2 = ax2.pcolormesh(G, T_mesh_g/E_F, alpha_g, shading='auto', 
                          cmap='RdYlGn_r', vmin=0.5, vmax=2.5)
    fig.colorbar(pcm2, ax=ax2, label='Resistivity Exponent α (ρ ~ T^α)')
    
    ax2.contour(G, T_mesh_g/E_F, alpha_g, levels=[1.0, 1.5, 2.0], 
                colors=['red', 'black', 'blue'], linewidths=1.2, linestyles=['-', '--', ':'])
    
    # Crossover line T_cross(g)
    T_cross_g = np.array([crossover_temperature(k0_fix, g, cfg) for g in g_range])
    ax2.plot(g_range, T_cross_g/E_F, 'white', linestyle='--', linewidth=2.0, 
             label=r'Crossover $T^* \sim |r|^{1/2}$')
    
    # Mark QCP
    if np.isfinite(g_c):
        ax2.axhline(g_c, color='yellow', linestyle='-.', linewidth=2.5, 
                    label=f'QCP: $g^c$ = {g_c:.2f}')
        #ax2.text(0.15, g_c, 'QCP', color='yellow', ha='left', va='center', 
        #         fontweight='bold', fontsize=10)
    
    ax2.set_xlabel('Coupling Strength $g$')
    ax2.set_ylabel('Temperature $T / E_F$')
    ax2.set_title(f'T vs $g$ (Fixed $k_0={k0_fix}$)')
    #ax2.set_xlim(0.1, 2.5)
    ax2.set_ylim(0, 2.0)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', fontsize=9)
    
    fig.tight_layout()
    plt.show()
    
    # Print analytical QCP values
    print(f"\n📊 Analytical QCP from Resistivity Scaling ({cfg['name']}):")
    print(f"   At fixed g={g_fix}:  k₀^c = {k0_c/k_F if np.isfinite(k0_c) else 'N/A':.3f} k_F")
    print(f"   At fixed k₀={k0_fix}: g^c = {g_c if np.isfinite(g_c) else 'N/A':.3f}")
    print(f"   Crossover scale: T* ~ |Ω²(2k_F) - g²|Π₀||^(1/2)")

# =============================================================================
# CONFIGURATIONS & EXECUTION
# =============================================================================
configs = [
    {'name': 'Ohmic + Constant',  'damping': 'ohmic',   'dispersion': 'constant',  'gamma0': 0.5, 'M': 0.5, 'k0': 1.0, 'g': 1.0},
    #{'name': 'Ohmic + Linear',    'damping': 'ohmic',   'dispersion': 'linear',    'gamma0': 0.5, 'c': 1.0, 'k0': 1.0, 'g': 1.0},
    #{'name': 'Ohmic + Quadratic', 'damping': 'ohmic',   'dispersion': 'quadratic', 'gamma0': 0.5, 'M': 0.5, 'k0': 1.0, 'g': 1.0},
]

print("="*70)
print("ANALYTICAL QCP ESTIMATION FROM RESISTIVITY SCALING")
print("="*70)
print("QCP condition: Ω²(2k_F) = g² |Π₀(2k_F)|  (boson mass vanishes)")
print("Resistivity signature: α = d lnρ/d lnT changes from 2 → 1 at low T")
print("="*70)

for cfg in configs:
    d_key = 'gamma0' if cfg['damping'] == 'ohmic' else 'GammaD'
    d_val = cfg.get(d_key, 0.5)
    plot_resistivity_analytical_phase_diagram(cfg, f'Damping: {d_val}')

print("\n" + "="*70)
print("✅ ANALYTICAL RESISTIVITY PHASE DIAGRAMS COMPLETE")
print("="*70)
print("\n📖 Interpretation:")
print("   • Red region (α≈1): Planckian/Quantum Critical regime (ρ ~ T)")
print("   • Blue region (α≈2): Fermi Liquid regime (ρ ~ T²)")
print("   • White dashed line: Crossover scale T* ~ |r|^(1/2)")
print("   • Yellow dash-dot line: Exact QCP (r=0) from resistivity scaling")
print("\n🔍 Validation: Compare with numerical ρ(T) exponent extraction")
print("   The analytical QCP line should align with numerical α=1 contour at low T")

