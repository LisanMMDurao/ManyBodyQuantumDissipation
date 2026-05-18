"""
run_rpa_doping_suite.py
-----------------------
Master driver for the RPA + GW + doping suite.

Implements the user's full specification:
  1. Fractional doping delta_n/n_0 = -N_0 Re Sigma_e(0)/n_0 as primary control
  2. Both Z_at_zero (first-order) and Z_pole (self-consistent) computed
  3. Central figure on (r, delta_n) plane with observables overlaid
  4. (r, T) plane retained as secondary diagnostic

Output figures:
  fig_doping_calibration.pdf   : doping vs g/k0 mapping
  fig_Z_comparison.pdf         : Z_at_zero vs Z_pole side by side
  fig_central_doping_plane.pdf : main figure -- observables on (r, delta_n)
  fig_rT_with_doping.pdf       : (r, T) maps with doping overlaid
"""

from __future__ import annotations
import os, time, json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

from bath_dressed import BathRPAModel, control_parameter_r
from electron_diagnostics import (
    im_sigma_electron, im_sigma_grid, re_sigma_from_im,
    quasiparticle_residue, planckian_ratio,
    resistivity_proxy, resistivity_exponent, fit_imsigma_exponent
)
from doping import (re_sigma_at_zero, fractional_doping,
                    quasiparticle_pole_self_consistent, full_diagnostic)

# APS style
rcParams.update({
    "font.family":        "serif",
    "font.size":          9, "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize":    8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "xtick.direction":    "in", "ytick.direction": "in",
    "xtick.top":          True, "ytick.right": True,
    "xtick.major.size":   3.5, "ytick.major.size": 3.5,
    "axes.linewidth":     0.6, "lines.linewidth":  1.2,
    "savefig.dpi":        300, "savefig.bbox":     "tight",
    "savefig.pad_inches": 0.02, "text.usetex":     False,
    "mathtext.fontset":   "cm",
})

PRB_SINGLE = 3.375
PRB_DOUBLE = 7.0
OUTDIR = "."


# =============================================================================
# Figure 1: Doping calibration -- delta_n/n_0 vs (g, k_0)
# =============================================================================

def fig_doping_calibration(base_params: dict, T: float = 0.05,
                            savedir=OUTDIR):
    """
    Map delta_n/n_0 across (g, k_0). Shows how to use the bath parameters
    to access different "doping" levels.
    """
    print("\n[Fig 1] Doping calibration")

    g_arr = np.linspace(0.3, 4.0, 14)
    k0_arr = np.linspace(0.5, 2.0, 10)
    G, K0 = np.meshgrid(g_arr, k0_arr, indexing="ij")

    doping_map = np.full(G.shape, np.nan)
    r_map = np.full(G.shape, np.nan)

    for i, g in enumerate(g_arr):
        print(f"  g = {g:.2f}  ({i+1}/{len(g_arr)})...")
        for j, k0 in enumerate(k0_arr):
            params = dict(base_params); params["g"] = g; params["k0"] = k0
            p = BathRPAModel(**params)
            try:
                dop = fractional_doping(p, T=T, n_q=61, n_theta=31)
                doping_map[i, j] = dop["delta_n_over_n0"]
                r_map[i, j] = control_parameter_r(p)["r"]
            except Exception:
                pass

    fig, axes = plt.subplots(1, 2, figsize=(PRB_DOUBLE, PRB_SINGLE * 1.0),
                              gridspec_kw={"wspace": 0.30})

    # Panel (a): doping(g, k0)
    ax = axes[0]
    # Use symlog-like scale for the wide dynamic range
    # Clip extremes for visibility
    dop_clipped = np.clip(doping_map, -1.0, 1.0)
    pcm = ax.pcolormesh(G, K0, dop_clipped, shading="auto", cmap="RdBu_r",
                         vmin=-1.0, vmax=1.0, rasterized=True)
    fig.colorbar(pcm, ax=ax, label=r"$\delta n / n_0$ (clipped at $\pm 1$)")
    # Contours
    for lvl in [-0.5, -0.1, 0, 0.1, 0.5]:
        try:
            ax.contour(G, K0, doping_map, levels=[lvl], colors="black",
                        linewidths=0.6, alpha=0.5)
        except Exception:
            pass
    ax.set_xlabel(r"coupling $g$")
    ax.set_ylabel(r"bath momentum scale $k_0 / k_F$")
    ax.set_title(rf"(a) fractional doping $\delta n / n_0$ at $T={T}$")

    # Panel (b): r(g, k0) overlay
    ax = axes[1]
    pcm = ax.pcolormesh(G, K0, r_map, shading="auto", cmap="RdYlBu",
                         vmin=-1.0, vmax=1.0, rasterized=True)
    fig.colorbar(pcm, ax=ax, label=r"$r$ (bath control parameter)")
    # QCP contour
    try:
        ax.contour(G, K0, r_map, levels=[0.0], colors="yellow",
                    linewidths=1.5, linestyles="-")
    except Exception:
        pass
    ax.set_xlabel(r"coupling $g$")
    ax.set_ylabel(r"bath momentum scale $k_0 / k_F$")
    ax.set_title(r"(b) bath control parameter $r$")

    fig.suptitle(r"Calibration: doping and $r$ across bath parameter space",
                 y=1.02, fontsize=10)
    fig.savefig(os.path.join(savedir, "fig_doping_calibration.pdf"))
    plt.close(fig)
    print("[ok] fig_doping_calibration.pdf")

    return {"g_arr": g_arr, "k0_arr": k0_arr,
            "doping_map": doping_map, "r_map": r_map}


# =============================================================================
# Figure 2: Z comparison -- Z(0) vs Z(pole)
# =============================================================================

def fig_Z_comparison(base_params: dict, T: float = 0.05, savedir=OUTDIR):
    """
    Side by side: Z_at_zero (first-order) and Z_pole (self-consistent)
    as g sweeps across the QCP.
    """
    print("\n[Fig 2] Z(0) vs Z(pole) comparison")
    g_arr = np.linspace(0.3, 4.5, 22)

    Z_zero = []
    Z_pole = []
    omega_qp = []
    doping = []
    r_arr = []

    for g in g_arr:
        params = dict(base_params); params["g"] = g
        p = BathRPAModel(**params)
        diag = full_diagnostic(p, T)
        Z_zero.append(diag["Z_at_zero"])
        Z_pole.append(diag["Z_pole"])
        omega_qp.append(diag["omega_qp"])
        doping.append(diag["delta_n_over_n0"])
        r_arr.append(control_parameter_r(p)["r"])

    Z_zero = np.array(Z_zero)
    Z_pole = np.array(Z_pole)
    omega_qp = np.array(omega_qp)
    doping = np.array(doping)
    r_arr = np.array(r_arr)

    fig, axes = plt.subplots(1, 3, figsize=(PRB_DOUBLE, PRB_SINGLE),
                              gridspec_kw={"wspace": 0.35})

    # Panel (a): Z comparison vs g
    ax = axes[0]
    ax.plot(g_arr, Z_zero, "o-", color="#1f4e79", ms=3.5, label=r"$Z(\omega=0)$ (first-order)")
    ax.plot(g_arr, Z_pole, "s-", color="#c0392b", ms=3.5, label=r"$Z(\omega_{qp})$ (self-consistent)")
    # r = 0 marker
    sign_changes = np.where(np.diff(np.sign(r_arr)))[0]
    if len(sign_changes) > 0:
        g_qcp = g_arr[sign_changes[0]]
        ax.axvline(g_qcp, color="yellow", lw=1.0, ls="-.",
                    label=rf"bath QCP $g_c={g_qcp:.2f}$")
    ax.set_xlabel(r"$g$")
    ax.set_ylabel(r"$Z$")
    ax.set_ylim(-0.05, 1.1)
    ax.set_title(r"(a) Quasiparticle residue")
    ax.legend(loc="lower left", fontsize=6.5, framealpha=0.9)
    ax.grid(True, alpha=0.2)

    # Panel (b): pole position omega_qp(g)
    ax = axes[1]
    ax.plot(g_arr, omega_qp, "o-", color="#2c7a3e", ms=3.5)
    ax.axhline(0, color="black", lw=0.5)
    if len(sign_changes) > 0:
        ax.axvline(g_qcp, color="yellow", lw=1.0, ls="-.")
    ax.set_xlabel(r"$g$")
    ax.set_ylabel(r"$\omega_{qp}$ (in $E_F$ units)")
    ax.set_title(r"(b) Pole frequency shift")
    ax.grid(True, alpha=0.2)

    # Panel (c): doping(g)
    ax = axes[2]
    ax.plot(g_arr, doping, "o-", color="#8b4789", ms=3.5)
    ax.axhline(0, color="black", lw=0.5)
    if len(sign_changes) > 0:
        ax.axvline(g_qcp, color="yellow", lw=1.0, ls="-.")
    ax.set_xlabel(r"$g$")
    ax.set_ylabel(r"$\delta n / n_0$")
    ax.set_title(r"(c) Fractional doping")
    ax.grid(True, alpha=0.2)

    fig.suptitle(r"Comparison: pole position, doping, and Z definitions",
                 y=1.02, fontsize=10)
    fig.savefig(os.path.join(savedir, "fig_Z_comparison.pdf"))
    plt.close(fig)
    print("[ok] fig_Z_comparison.pdf")

    return {"g_arr": g_arr, "Z_zero": Z_zero, "Z_pole": Z_pole,
            "omega_qp": omega_qp, "doping": doping, "r_arr": r_arr}


# =============================================================================
# Figure 3 (CENTRAL): Observables on (delta_n, T) plane
# =============================================================================

def fig_central_doping_plane(base_params: dict, savedir=OUTDIR):
    """
    Central figure of the analysis: observables on the (delta_n, T) plane.

    For each (g, T), compute delta_n, then plot observables as functions of
    delta_n with T on the y-axis. This is the cuprate-style phase diagram.
    """
    print("\n[Fig 3 - CENTRAL] Observables on (delta_n, T) plane")

    g_arr = np.linspace(0.3, 4.0, 16)
    T_arr = np.geomspace(0.01, 0.3, 10)

    doping_map = np.full((len(T_arr), len(g_arr)), np.nan)
    r_map = np.full(len(g_arr), np.nan)
    Z_pole_map = np.full((len(T_arr), len(g_arr)), np.nan)
    gT_map = np.full((len(T_arr), len(g_arr)), np.nan)
    alpha_map = np.full((len(T_arr), len(g_arr)), np.nan)
    omega_qp_map = np.full((len(T_arr), len(g_arr)), np.nan)

    # r doesn't depend on T
    for j, g in enumerate(g_arr):
        params = dict(base_params); params["g"] = g
        r_map[j] = control_parameter_r(BathRPAModel(**params))["r"]

    for i, T in enumerate(T_arr):
        print(f"  T = {T:.4f}  ({i+1}/{len(T_arr)})")
        for j, g in enumerate(g_arr):
            params = dict(base_params); params["g"] = g
            p = BathRPAModel(**params)
            try:
                diag = full_diagnostic(p, T, n_q=61, n_theta=31)
                doping_map[i, j] = diag["delta_n_over_n0"]
                Z_pole_map[i, j] = diag["Z_pole"]
                omega_qp_map[i, j] = diag["omega_qp"]
                gT_map[i, j] = planckian_ratio(T, p)
                alpha_map[i, j] = resistivity_exponent(T, p)
            except Exception:
                pass

    # Build delta_n axis: use T-averaged or T=lowest doping at each g
    # to define a 1D doping axis. Doping varies slightly with T but mostly
    # with g.
    doping_1d = np.nanmean(doping_map, axis=0)
    # Sort by doping
    sort_idx = np.argsort(doping_1d)
    doping_sorted = doping_1d[sort_idx]
    # Reorder maps to be monotone in doping
    Z_pole_sorted = Z_pole_map[:, sort_idx]
    gT_sorted = gT_map[:, sort_idx]
    alpha_sorted = alpha_map[:, sort_idx]
    r_sorted = r_map[sort_idx]

    # Restrict to "physical" doping range (avoid the divergent g >> g_c region)
    # Clip doping to a physical window e.g. [-0.5, 1.0]
    mask = (doping_sorted >= -0.5) & (doping_sorted <= 1.0)
    if mask.sum() < 5:
        # Fall back to lowest 80% by g
        mask = np.ones_like(doping_sorted, dtype=bool)
        mask[int(0.85 * len(mask)):] = False

    doping_axis = doping_sorted[mask]
    Z_panel = Z_pole_sorted[:, mask]
    gT_panel = gT_sorted[:, mask]
    alpha_panel = alpha_sorted[:, mask]
    r_axis = r_sorted[mask]

    # Doping value at r = 0
    sign_changes = np.where(np.diff(np.sign(r_axis)))[0]
    doping_at_qcp = doping_axis[sign_changes[0]] if len(sign_changes) > 0 else None

    # Plot the central figure
    fig, axes = plt.subplots(2, 2, figsize=(PRB_DOUBLE, PRB_DOUBLE * 0.75),
                              gridspec_kw={"hspace": 0.38, "wspace": 0.30})

    D, T_M = np.meshgrid(doping_axis, T_arr)

    def add_qcp_marker(ax):
        if doping_at_qcp is not None:
            ax.axvline(doping_at_qcp, color="yellow", lw=1.3, ls="-.",
                        label=rf"bath QCP: $\delta n/n_0 \approx {doping_at_qcp:.3f}$")
            ax.legend(loc="upper right", fontsize=6.5, framealpha=0.9)

    def panel(ax, M, title, cmap, vmin=None, vmax=None, label=""):
        pcm = ax.pcolormesh(D, T_M, M, shading="auto", cmap=cmap,
                             vmin=vmin, vmax=vmax, rasterized=True)
        fig.colorbar(pcm, ax=ax, label=label)
        ax.set_xlabel(r"fractional doping $\delta n / n_0$")
        ax.set_ylabel(r"$T / E_F$")
        ax.set_title(title)
        add_qcp_marker(ax)

    panel(axes[0, 0], Z_panel,
           r"(a) $Z(\delta n, T)$ (self-consistent)", "viridis",
           vmin=0, vmax=1, label=r"$Z$")
    panel(axes[0, 1], np.log10(np.clip(gT_panel, 1e-3, 1e2)),
           r"(b) $\log_{10}(\Gamma_{qp}/T)$ (Planckian)", "plasma",
           label=r"$\log_{10}(\Gamma/T)$")
    panel(axes[1, 0], alpha_panel,
           r"(c) $\alpha_\rho$ ($\rho \sim T^\alpha$)", "RdYlGn_r",
           vmin=0, vmax=2.5, label=r"$\alpha_\rho$")
    # Contour overlay for alpha = 1 (Planckian) and alpha = 2 (FL)
    for level, color in [(1.0, "darkred"), (2.0, "navy")]:
        try:
            axes[1, 0].contour(D, T_M, alpha_panel, levels=[level],
                                colors=[color], linewidths=1.0, alpha=0.9)
        except Exception:
            pass

    # Panel (d): r vs delta_n curve (1D, for orientation)
    ax = axes[1, 1]
    ax.plot(doping_axis, r_axis, "o-", color="#1f4e79", ms=3.5)
    ax.axhline(0, color="red", lw=0.6, ls="--", label=r"$r = 0$ (QCP)")
    ax.axvline(doping_at_qcp if doping_at_qcp is not None else 0,
                color="yellow", lw=1.0, ls="-.")
    ax.set_xlabel(r"fractional doping $\delta n / n_0$")
    ax.set_ylabel(r"$r$ (bath control parameter)")
    ax.set_title(r"(d) $r$ vs $\delta n / n_0$")
    ax.legend(loc="upper right", fontsize=7, framealpha=0.9)
    ax.grid(True, alpha=0.2)

    fig.suptitle(r"Central figure: electronic phase diagram in $(\delta n, T)$ plane",
                 y=0.98, fontsize=10)
    fig.savefig(os.path.join(savedir, "fig_central_doping_plane.pdf"))
    plt.close(fig)
    print("[ok] fig_central_doping_plane.pdf")

    return {"doping_axis": doping_axis, "T_arr": T_arr,
            "Z_panel": Z_panel, "gT_panel": gT_panel,
            "alpha_panel": alpha_panel, "r_axis": r_axis,
            "doping_at_qcp": doping_at_qcp}


# =============================================================================
# Figure 4: (r, T) maps with doping contours overlaid
# =============================================================================

def fig_rT_with_doping(base_params: dict, savedir=OUTDIR):
    """
    The (r, T) plane with doping contours overlaid -- secondary diagnostic.
    """
    print("\n[Fig 4] (r, T) plane with doping overlay")

    # Build r-axis via g calibration
    g_calib = np.linspace(0.3, 4.0, 30)
    r_calib = []
    for g in g_calib:
        params = dict(base_params); params["g"] = g
        r_calib.append(control_parameter_r(BathRPAModel(**params))["r"])
    r_calib = np.array(r_calib)
    from scipy.interpolate import interp1d
    # Sort r monotonically (decreasing in g, so invert)
    sort_idx = np.argsort(r_calib)
    g_of_r = interp1d(r_calib[sort_idx], g_calib[sort_idx],
                       kind="linear", bounds_error=False, fill_value="extrapolate")

    r_axis = np.linspace(r_calib.min() + 0.01, r_calib.max() - 0.01, 14)
    T_arr = np.geomspace(0.01, 0.3, 10)

    Z_pole_map = np.full((len(T_arr), len(r_axis)), np.nan)
    gT_map = np.full((len(T_arr), len(r_axis)), np.nan)
    alpha_map = np.full((len(T_arr), len(r_axis)), np.nan)
    doping_map = np.full((len(T_arr), len(r_axis)), np.nan)

    for i, T in enumerate(T_arr):
        print(f"  T = {T:.4f}  ({i+1}/{len(T_arr)})")
        for j, r_t in enumerate(r_axis):
            g_eff = float(g_of_r(r_t))
            if not np.isfinite(g_eff) or g_eff < 0:
                continue
            params = dict(base_params); params["g"] = g_eff
            p = BathRPAModel(**params)
            try:
                diag = full_diagnostic(p, T, n_q=61, n_theta=31)
                Z_pole_map[i, j] = diag["Z_pole"]
                gT_map[i, j] = planckian_ratio(T, p)
                alpha_map[i, j] = resistivity_exponent(T, p)
                doping_map[i, j] = diag["delta_n_over_n0"]
            except Exception:
                pass

    fig, axes = plt.subplots(2, 2, figsize=(PRB_DOUBLE, PRB_DOUBLE * 0.75),
                              gridspec_kw={"hspace": 0.38, "wspace": 0.30})

    R, T_M = np.meshgrid(r_axis, T_arr)

    def panel(ax, M, title, cmap, vmin=None, vmax=None, label=""):
        pcm = ax.pcolormesh(R, T_M, M, shading="auto", cmap=cmap,
                             vmin=vmin, vmax=vmax, rasterized=True)
        fig.colorbar(pcm, ax=ax, label=label)
        ax.axvline(0, color="yellow", lw=1.2, ls="-.", label="QCP $r=0$")
        # Doping contours overlay
        try:
            cs = ax.contour(R, T_M, doping_map,
                              levels=[-0.2, -0.05, 0.0, 0.05, 0.2, 0.5],
                              colors="white", linewidths=0.7,
                              linestyles="-", alpha=0.85)
            ax.clabel(cs, inline=True, fontsize=5.5, fmt=r"$\delta n/n_0$=%.2f")
        except Exception:
            pass
        ax.set_xlabel(r"$r$")
        ax.set_ylabel(r"$T / E_F$")
        ax.set_title(title)
        ax.legend(loc="upper right", fontsize=6.5, framealpha=0.9)

    panel(axes[0, 0], Z_pole_map,
           r"(a) $Z_{\rm pole}(r, T)$", "viridis", vmin=0, vmax=1, label=r"$Z$")
    panel(axes[0, 1], np.log10(np.clip(gT_map, 1e-3, 1e2)),
           r"(b) $\log_{10}(\Gamma_{qp}/T)$", "plasma",
           label=r"$\log_{10}(\Gamma/T)$")
    panel(axes[1, 0], alpha_map,
           r"(c) $\alpha_\rho$", "RdYlGn_r", vmin=0, vmax=2.5, label=r"$\alpha$")
    panel(axes[1, 1], doping_map,
           r"(d) $\delta n/n_0(r, T)$", "RdBu_r", vmin=-0.5, vmax=0.5,
           label=r"$\delta n/n_0$")

    fig.suptitle(r"Phase diagram on $(r, T)$ plane with doping contours",
                 y=0.98, fontsize=10)
    fig.savefig(os.path.join(savedir, "fig_rT_with_doping.pdf"))
    plt.close(fig)
    print("[ok] fig_rT_with_doping.pdf")


# =============================================================================
# Main driver
# =============================================================================

def main():
    print("=" * 70)
    print("RPA + GW + Doping suite")
    print("=" * 70)
    t_start = time.time()

    os.makedirs(OUTDIR, exist_ok=True)

    base_params = dict(
        dispersion="gapped", Omega0=0.5, c_disp=0.5,
        damping="ohmic", gamma0=0.1, kF=1.0, m_e=1.0,
        use_RPA=True,
    )

    fig_doping_calibration(base_params)
    fig_Z_comparison(base_params, T=0.05)
    fig_central_doping_plane(base_params)
    fig_rT_with_doping(base_params)

    t_total = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"Suite complete. Total time: {t_total:.1f} s")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
