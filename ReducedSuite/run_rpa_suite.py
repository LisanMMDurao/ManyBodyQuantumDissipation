"""
run_rpa_suite.py
----------------
Master driver for the RPA-dressed bath + electron observables suite.

This implements the broad, instrumented exploration the user proposed:
   "Build it broadly and let the results speak."

For each scan, we map electronic observables (vartheta_Sigma, Z, Gamma/T,
rho exponent alpha, etc.) as functions of:
   - g (electron-boson coupling)
   - k0 (bath momentum scale)
   - Omega0 (bath gap)
   - T (temperature)
and infer the corresponding bath control parameter r = min_q M(q).

The output is a set of (r, T) maps and parameter-sweep plots designed to
let physical patterns emerge without committing to a specific interpretation
in advance.
"""

from __future__ import annotations
import os, time, json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.colors import LogNorm, SymLogNorm

from bath_dressed import BathRPAModel, control_parameter_r, mass_function_corrected
from electron_diagnostics import (
    im_sigma_electron, im_sigma_grid, re_sigma_from_im,
    quasiparticle_residue, quasiparticle_gamma, planckian_ratio,
    resistivity_proxy, resistivity_exponent, fit_imsigma_exponent
)
from rpa_polarization import re_Pi_static, kohn_singularity_strength

# =============================================================================
# APS/PRB style
# =============================================================================

rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Computer Modern Roman", "Times New Roman", "DejaVu Serif"],
    "font.size":          9,
    "axes.labelsize":     9,
    "axes.titlesize":     9,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "legend.fontsize":    7.5,
    "xtick.direction":    "in",
    "ytick.direction":    "in",
    "xtick.top":          True,
    "ytick.right":        True,
    "xtick.major.size":   3.5,
    "ytick.major.size":   3.5,
    "axes.linewidth":     0.6,
    "lines.linewidth":    1.2,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.02,
    "text.usetex":        False,
    "mathtext.fontset":   "cm",
})

PRB_SINGLE = 3.375
PRB_DOUBLE = 7.0
OUTDIR = "."


# =============================================================================
# Phase 0: structural diagnostic of the bath (no electron gas yet)
# =============================================================================

def phase0_bath_structure(savedir=OUTDIR):
    """
    Show the structure of M(q) = Omega_q^2 + g^2 Re Pi(q, 0) as a function
    of g and k0 -- locate the QCP locus in parameter space.
    """
    print("\n[Phase 0] Bath structure: M(q) and control parameter r")

    g_array = np.linspace(0.1, 5.0, 60)
    q_grid = np.linspace(0.01, 3.0, 200)

    # Build r(g) for a few dispersion choices
    fig, axes = plt.subplots(1, 2, figsize=(PRB_DOUBLE, PRB_SINGLE),
                              gridspec_kw={"wspace": 0.30})

    # Panel (a): M(q) curves for several g, fixed Omega0
    ax = axes[0]
    cmap = plt.get_cmap("viridis")
    for i, g in enumerate(np.linspace(0.5, 4.0, 5)):
        p = BathRPAModel(dispersion="gapped", Omega0=0.5, c_disp=0.5,
                         damping="ohmic", gamma0=0.1, g=g,
                         kF=1.0, m_e=1.0, use_RPA=True)
        M = mass_function_corrected(q_grid, p)
        ax.plot(q_grid, M, color=cmap(i / 5), label=rf"$g={g:.1f}$")
    ax.axhline(0, color="black", lw=0.5)
    ax.axvline(2.0, color="red", lw=0.5, ls=":", label=r"$2k_F$")
    ax.set_xlabel(r"$q / k_F$")
    ax.set_ylabel(r"$M(q) = \Omega_q^2 + g^2 \mathrm{Re}\Pi(q,0)$")
    ax.set_title(r"(a) Mass function $M(q)$ for varying $g$")
    ax.legend(loc="upper right", fontsize=6.5, framealpha=0.9)
    ax.grid(True, alpha=0.2)

    # Panel (b): r vs g for several Omega0
    ax = axes[1]
    for Omega0 in [0.2, 0.5, 1.0, 1.5]:
        r_array = []
        for g in g_array:
            p = BathRPAModel(dispersion="gapped", Omega0=Omega0, c_disp=0.5,
                             damping="ohmic", gamma0=0.1, g=g,
                             kF=1.0, m_e=1.0, use_RPA=True)
            info = control_parameter_r(p, q_grid=q_grid)
            r_array.append(info["r"])
        ax.plot(g_array, r_array, label=rf"$\Omega_0={Omega0}$")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlabel(r"coupling $g$")
    ax.set_ylabel(r"control parameter $r$")
    ax.set_title(r"(b) $r(g)$ for varying $\Omega_0$")
    ax.legend(loc="upper right", fontsize=7, framealpha=0.9)
    ax.grid(True, alpha=0.2)

    fig.suptitle(r"Phase 0: bath QCP structure (no electronic feedback yet)",
                 y=1.02, fontsize=10)
    fig.savefig(os.path.join(savedir, "fig_phase0_bath_structure.pdf"))
    plt.close(fig)
    print("[ok] fig_phase0_bath_structure.pdf")


# =============================================================================
# Phase 1: electron observables in single-point parameter sweeps
# =============================================================================

def phase1_observables_vs_g(g_array: np.ndarray, T: float,
                             base_params: dict, savedir=OUTDIR):
    """
    Fix all parameters except g and T; sweep g; compute all electron
    observables at fixed T.

    Outputs:
      - Z(g), Gamma/T(g), alpha_rho(g), vth_Sigma(g) on a single panel
      - corresponding r(g) on a twin axis to relate the sweep to the QCP locus
    """
    print(f"\n[Phase 1] Observables vs g, at T = {T}")

    results = {
        "g_array": [],
        "r":       [],
        "Z":       [],
        "gamma_over_T": [],
        "alpha_rho": [],
        "vth_imSig": [],
    }

    omega_fit_grid = np.geomspace(1e-4, 0.3, 14)

    for g in g_array:
        params = dict(base_params)
        params["g"] = g
        p = BathRPAModel(**params)
        info_r = control_parameter_r(p)

        # Im Sigma over grid for vth fit
        im_sig = np.array([im_sigma_electron(om, T, p, n_q=81, n_theta=41)
                           for om in omega_fit_grid])
        fit = fit_imsigma_exponent(omega_fit_grid, im_sig, fit_range=(1e-3, 1e-2))

        # Re Sigma -> Z
        re_sig = re_sigma_from_im(omega_fit_grid, im_sig)
        Z_info = quasiparticle_residue(omega_fit_grid, re_sig, window=5e-3)

        # Planckian ratio
        gT = planckian_ratio(T, p)

        # rho exponent
        alpha = resistivity_exponent(T, p)

        results["g_array"].append(float(g))
        results["r"].append(info_r["r"])
        results["Z"].append(Z_info["Z"])
        results["gamma_over_T"].append(gT)
        results["alpha_rho"].append(alpha)
        results["vth_imSig"].append(fit["vth"])

        print(f"  g={g:.2f}  r={info_r['r']:+.3f}  Z={Z_info['Z']:.3f}  "
              f"Γ/T={gT:.3f}  α_ρ={alpha:.3f}  vth={fit['vth']:.3f}")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(PRB_DOUBLE, PRB_DOUBLE * 0.7),
                              gridspec_kw={"hspace": 0.35, "wspace": 0.30})

    g_arr = np.array(results["g_array"])
    r_arr = np.array(results["r"])

    def make_panel(ax, y_vals, y_label, title, hline=None):
        ax.plot(g_arr, y_vals, "o-", color="#1f4e79", ms=3.5)
        ax.set_xlabel(r"coupling $g$")
        ax.set_ylabel(y_label)
        ax.set_title(title)
        if hline is not None:
            ax.axhline(hline, color="gray", lw=0.6, ls="--")
        # Twin axis: r(g)
        ax_r = ax.twinx()
        ax_r.plot(g_arr, r_arr, "-", color="#a83232", lw=0.8, alpha=0.5)
        ax_r.axhline(0, color="#a83232", lw=0.4, ls=":", alpha=0.5)
        ax_r.set_ylabel(r"$r$", color="#a83232", fontsize=7)
        ax_r.tick_params(axis="y", labelcolor="#a83232", labelsize=7)
        ax.grid(True, alpha=0.2)

    make_panel(axes[0, 0], results["Z"], r"$Z$",
                r"(a) Quasiparticle residue", hline=1.0)
    make_panel(axes[0, 1], results["gamma_over_T"], r"$\Gamma_{qp}/T$",
                r"(b) Planckian diagnostic", hline=1.0)
    make_panel(axes[1, 0], results["alpha_rho"], r"$\alpha_\rho$",
                r"(c) Resistivity exponent")
    make_panel(axes[1, 1], results["vth_imSig"],
                r"$\vartheta_\Sigma$",
                r"(d) Self-energy exponent ($T=$" f"{T:.3f})")

    fig.suptitle(rf"Phase 1: electron observables vs $g$ at $T={T}$",
                 y=1.00, fontsize=10)
    fig.savefig(os.path.join(savedir, f"fig_phase1_observables_vs_g_T{T:.3f}.pdf"))
    plt.close(fig)
    print(f"[ok] fig_phase1_observables_vs_g_T{T:.3f}.pdf")

    return results


# =============================================================================
# Phase 2: (g, T) maps of all observables
# =============================================================================

def phase2_gT_maps(g_array: np.ndarray, T_array: np.ndarray,
                    base_params: dict, savedir=OUTDIR):
    """
    Build 2D maps of (Z, Gamma/T, alpha_rho, vth) over the (g, T) plane.

    Overlay: contour where r(g) = 0 -- the QCP locus of the bath.
    """
    print(f"\n[Phase 2] (g, T) maps: scanning {len(g_array)} g × {len(T_array)} T")

    Z_map     = np.full((len(T_array), len(g_array)), np.nan)
    gT_map    = np.full((len(T_array), len(g_array)), np.nan)
    alpha_map = np.full((len(T_array), len(g_array)), np.nan)
    vth_map   = np.full((len(T_array), len(g_array)), np.nan)
    r_array   = np.zeros(len(g_array))

    # r doesn't depend on T, compute once
    for j, g in enumerate(g_array):
        params = dict(base_params); params["g"] = g
        p = BathRPAModel(**params)
        r_array[j] = control_parameter_r(p)["r"]

    omega_fit_grid = np.geomspace(1e-4, 0.3, 10)

    t0 = time.time()
    for i, T in enumerate(T_array):
        print(f"  T = {T:.4f}  ({i+1}/{len(T_array)})...")
        for j, g in enumerate(g_array):
            params = dict(base_params); params["g"] = g
            p = BathRPAModel(**params)
            try:
                im_sig = np.array([im_sigma_electron(om, T, p,
                                                      n_q=61, n_theta=31)
                                    for om in omega_fit_grid])
                fit = fit_imsigma_exponent(omega_fit_grid, im_sig,
                                            fit_range=(1e-3, 1e-2))
                re_sig = re_sigma_from_im(omega_fit_grid, im_sig)
                Z_info = quasiparticle_residue(omega_fit_grid, re_sig,
                                                 window=5e-3)
                Z_map[i, j]     = Z_info["Z"]
                gT_map[i, j]    = planckian_ratio(T, p)
                alpha_map[i, j] = resistivity_exponent(T, p)
                vth_map[i, j]   = fit["vth"]
            except Exception as e:
                pass

    t_total = time.time() - t0
    print(f"  Phase 2 grid: {t_total:.1f}s")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(PRB_DOUBLE, PRB_DOUBLE * 0.7),
                              gridspec_kw={"hspace": 0.40, "wspace": 0.30})

    G, T_M = np.meshgrid(g_array, T_array)

    # Index of g where r changes sign (the bath QCP)
    sign_change = np.where(np.diff(np.sign(r_array)))[0]
    g_qcp = g_array[sign_change[0]] if len(sign_change) > 0 else None

    def overlay_qcp(ax):
        if g_qcp is not None:
            ax.axvline(g_qcp, color="yellow", lw=1.2, ls="-.",
                        label=rf"bath QCP: $g_c={g_qcp:.2f}$")
            ax.legend(loc="upper right", fontsize=6.5, framealpha=0.9)

    def panel(ax, M, title, cmap, vmin=None, vmax=None, label=""):
        pcm = ax.pcolormesh(G, T_M, M, shading="auto", cmap=cmap,
                             vmin=vmin, vmax=vmax, rasterized=True)
        fig.colorbar(pcm, ax=ax, label=label)
        ax.set_xlabel(r"$g$")
        ax.set_ylabel(r"$T / E_F$")
        ax.set_title(title)
        overlay_qcp(ax)

    panel(axes[0, 0], Z_map,
           r"(a) $Z(g, T)$", "viridis", vmin=0, vmax=1, label=r"$Z$")
    panel(axes[0, 1], np.log10(np.clip(gT_map, 1e-3, 1e2)),
           r"(b) $\log_{10}(\Gamma_{qp}/T)$", "plasma", label=r"$\log_{10}(\Gamma/T)$")
    panel(axes[1, 0], alpha_map,
           r"(c) $\alpha_\rho$ ($\rho\sim T^\alpha$)", "RdYlGn_r",
           vmin=0, vmax=2.5, label=r"$\alpha$")
    panel(axes[1, 1], vth_map,
           r"(d) $\vartheta_\Sigma$", "RdYlGn_r",
           vmin=0, vmax=2.5, label=r"$\vartheta$")

    # Common contours: alpha = 1 and alpha = 2
    for ax, M in [(axes[1, 0], alpha_map), (axes[1, 1], vth_map)]:
        for level, color in [(1.0, "darkred"), (2.0, "navy")]:
            try:
                ax.contour(G, T_M, M, levels=[level], colors=[color],
                            linewidths=1.0, alpha=0.9)
            except Exception:
                pass

    fig.suptitle(rf"Phase 2: electron observables on $(g, T)$ plane",
                 y=1.00, fontsize=10)
    fig.savefig(os.path.join(savedir, "fig_phase2_gT_maps.pdf"))
    plt.close(fig)
    print("[ok] fig_phase2_gT_maps.pdf")

    return {
        "Z_map":     Z_map,
        "gT_map":    gT_map,
        "alpha_map": alpha_map,
        "vth_map":   vth_map,
        "r_array":   r_array,
        "g_array":   g_array,
        "T_array":   T_array,
        "g_qcp":     g_qcp,
    }


# =============================================================================
# Phase 3: (r, T) plane — parameterize via r directly
# =============================================================================

def phase3_rT_plane(r_target_array: np.ndarray, T_array: np.ndarray,
                     base_params: dict, savedir=OUTDIR):
    """
    For each target r, find g (and Omega0) that achieves it, then compute
    observables. This produces the cleanest QCP fan-shape diagrams.

    Strategy: vary g at fixed Omega0; build interpolation r -> g.
    """
    print(f"\n[Phase 3] (r, T) plane: targeting {len(r_target_array)} r values, "
          f"{len(T_array)} T values")

    # Build calibration: scan g, get r(g), then invert
    g_calib = np.linspace(0.1, 6.0, 200)
    r_calib = np.zeros_like(g_calib)
    for j, g in enumerate(g_calib):
        params = dict(base_params); params["g"] = g
        r_calib[j] = control_parameter_r(BathRPAModel(**params))["r"]

    # r(g) monotone decreasing (since Re Pi < 0 -> M decreases with g)?
    # Sort r_calib in decreasing order and use as interpolation map.
    # We want g_for(r): given target r, find g.
    # The function is monotone if Re Pi has constant sign.
    # Let's interpolate properly:
    from scipy.interpolate import interp1d
    # If r_calib is monotone in g, build inverse
    sorted_idx = np.argsort(r_calib)
    r_sorted = r_calib[sorted_idx]
    g_sorted = g_calib[sorted_idx]
    # Unique r values
    _, uniq_idx = np.unique(r_sorted, return_index=True)
    r_uniq = r_sorted[uniq_idx]
    g_uniq = g_sorted[uniq_idx]
    g_of_r = interp1d(r_uniq, g_uniq, kind="linear",
                       bounds_error=False, fill_value="extrapolate")

    # Filter r targets to physically valid range
    r_min, r_max = r_uniq.min(), r_uniq.max()
    r_targets = r_target_array[(r_target_array >= r_min)
                                & (r_target_array <= r_max)]
    print(f"  Valid r range: [{r_min:.3f}, {r_max:.3f}]; "
          f"keeping {len(r_targets)} targets")

    # Compute observables
    Z_map     = np.full((len(T_array), len(r_targets)), np.nan)
    gT_map    = np.full((len(T_array), len(r_targets)), np.nan)
    alpha_map = np.full((len(T_array), len(r_targets)), np.nan)
    vth_map   = np.full((len(T_array), len(r_targets)), np.nan)

    omega_fit_grid = np.geomspace(1e-4, 0.3, 10)

    for i, T in enumerate(T_array):
        print(f"  T = {T:.4f}  ({i+1}/{len(T_array)})")
        for j, r_t in enumerate(r_targets):
            g_eff = float(g_of_r(r_t))
            if not np.isfinite(g_eff) or g_eff < 0:
                continue
            params = dict(base_params); params["g"] = g_eff
            p = BathRPAModel(**params)
            try:
                im_sig = np.array([im_sigma_electron(om, T, p,
                                                      n_q=61, n_theta=31)
                                    for om in omega_fit_grid])
                fit = fit_imsigma_exponent(omega_fit_grid, im_sig,
                                            fit_range=(1e-3, 1e-2))
                re_sig = re_sigma_from_im(omega_fit_grid, im_sig)
                Z_info = quasiparticle_residue(omega_fit_grid, re_sig,
                                                 window=5e-3)
                Z_map[i, j]     = Z_info["Z"]
                gT_map[i, j]    = planckian_ratio(T, p)
                alpha_map[i, j] = resistivity_exponent(T, p)
                vth_map[i, j]   = fit["vth"]
            except Exception:
                pass

    # Plot fan-shape diagrams
    fig, axes = plt.subplots(2, 2, figsize=(PRB_DOUBLE, PRB_DOUBLE * 0.7),
                              gridspec_kw={"hspace": 0.40, "wspace": 0.30})

    R, T_M = np.meshgrid(r_targets, T_array)

    def panel(ax, M, title, cmap, vmin=None, vmax=None, label=""):
        pcm = ax.pcolormesh(R, T_M, M, shading="auto", cmap=cmap,
                             vmin=vmin, vmax=vmax, rasterized=True)
        fig.colorbar(pcm, ax=ax, label=label)
        ax.axvline(0, color="yellow", lw=1.2, ls="-.",
                    label=r"QCP $r=0$")
        ax.set_xlabel(r"$r$ (bath control parameter)")
        ax.set_ylabel(r"$T / E_F$")
        ax.set_title(title)
        ax.legend(loc="upper right", fontsize=6.5, framealpha=0.9)

    panel(axes[0, 0], Z_map,
           r"(a) $Z(r, T)$", "viridis", vmin=0, vmax=1, label=r"$Z$")
    panel(axes[0, 1], np.log10(np.clip(gT_map, 1e-3, 1e2)),
           r"(b) $\log_{10}(\Gamma_{qp}/T)$", "plasma",
           label=r"$\log_{10}(\Gamma/T)$")
    panel(axes[1, 0], alpha_map,
           r"(c) $\alpha_\rho$", "RdYlGn_r", vmin=0, vmax=2.5, label=r"$\alpha$")
    panel(axes[1, 1], vth_map,
           r"(d) $\vartheta_\Sigma$", "RdYlGn_r",
           vmin=0, vmax=2.5, label=r"$\vartheta$")

    fig.suptitle(rf"Phase 3: electron response on $(r, T)$ plane "
                  r"(QCP at $r = 0$)", y=1.00, fontsize=10)
    fig.savefig(os.path.join(savedir, "fig_phase3_rT_plane.pdf"))
    plt.close(fig)
    print("[ok] fig_phase3_rT_plane.pdf")

    return {
        "Z_map":     Z_map,
        "gT_map":    gT_map,
        "alpha_map": alpha_map,
        "vth_map":   vth_map,
        "r_targets": r_targets,
        "T_array":   T_array,
        "g_of_r":    g_of_r,
    }


# =============================================================================
# Main driver
# =============================================================================

def main():
    print("=" * 70)
    print("RPA + GW + electron observables suite")
    print("=" * 70)
    t_start = time.time()

    os.makedirs(OUTDIR, exist_ok=True)

    # Base model: gapped dispersion, ohmic damping, fixed Omega0
    base_params = dict(
        dispersion="gapped",
        Omega0=0.5,
        c_disp=0.5,
        damping="ohmic",
        gamma0=0.1,
        kF=1.0,
        m_e=1.0,
        use_RPA=True,
    )

    # ---- Phase 0: bath structure
    phase0_bath_structure()

    # ---- Phase 1: observables vs g at one T (quick check)
    g_array_1 = np.linspace(0.5, 4.5, 12)
    phase1_observables_vs_g(g_array_1, T=0.05, base_params=base_params)

    # ---- Phase 2: (g, T) maps -- coarse grid
    g_array_2 = np.linspace(0.5, 4.5, 15)
    T_array_2 = np.geomspace(0.01, 0.3, 12)
    phase2_gT_maps(g_array_2, T_array_2, base_params)

    # ---- Phase 3: (r, T) plane via interpolation
    r_targets = np.linspace(-0.3, 0.3, 15)
    T_array_3 = np.geomspace(0.01, 0.3, 12)
    phase3_rT_plane(r_targets, T_array_3, base_params)

    t_total = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"Suite complete. Total time: {t_total:.1f} s")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
