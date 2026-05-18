"""
run_sc_dispersion_suite.py
--------------------------
Comprehensive scan across three bath dispersions with full
self-consistent treatment of the chemical potential shift.

Dispersions tested:
  1. "gapped"     : Omega_q = sqrt(Omega0^2 + (c*q)^2)   (canonical, fully gapped)
  2. "linear"     : Omega_q = c*q                         (Goldstone-like, ungapped)
  3. "quadratic"  : Omega_q = q^2 / (2*M)                 (Galilean, ungapped)

For each dispersion:
  - Calibrate r(g) and delta_n(g) under self-consistency
  - Build (delta_n, T) phase diagrams of:
      * Z_pole (self-consistent quasiparticle residue)
      * Gamma/T (Planckian diagnostic)
      * alpha_rho (resistivity exponent)
  - Overlay r=0 and other physical contours

Output:
  fig_sc_three_dispersions.pdf   : main result, 3 dispersions x 3 observables
  fig_sc_doping_vs_g.pdf          : doping calibration per dispersion
  fig_sc_central_phase_diagram.pdf : central figure (best dispersion)
"""

from __future__ import annotations
import os, time, json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

from bath_dressed import BathRPAModel, control_parameter_r
from electron_diagnostics import (
    im_sigma_electron, resistivity_proxy, resistivity_exponent
)
from self_consistent import full_sc_diagnostic, im_sigma_electron_shifted
from doping import fractional_doping

rcParams.update({
    "font.family":        "serif", "font.size": 9,
    "axes.labelsize":     9, "axes.titlesize": 9,
    "xtick.labelsize":    7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
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
# Resistivity exponent with self-consistent mu_shift
# =============================================================================

def resistivity_proxy_sc(T: float, p: BathRPAModel, mu_shift: float,
                          n_omega: int = 25, omega_max_factor: float = 5.0
                          ) -> float:
    """rho(T) computed with shifted xi_k in Im Sigma."""
    if n_omega % 2 == 0:
        n_omega += 1
    om_max = max(omega_max_factor * T, 1e-3)
    omega_grid = np.linspace(1e-6, om_max, n_omega)
    df_dom = 1.0 / (4.0 * T * np.cosh(omega_grid / (2.0 * T)) ** 2)
    im_sig = np.array([im_sigma_electron_shifted(om, T, p, mu_shift=mu_shift,
                                                   n_q=61, n_theta=31)
                       for om in omega_grid])
    integrand = df_dom * np.abs(im_sig)
    return float(np.trapezoid(integrand, omega_grid))


def resistivity_exponent_sc(T: float, p: BathRPAModel, mu_shift: float,
                             delta_frac: float = 0.2) -> float:
    """alpha = d ln rho / d ln T via 3-point fit."""
    dT = max(T * delta_frac, 1e-4)
    T_pts = np.array([max(T - dT, 1e-5), T, T + dT])
    rho_pts = np.array([resistivity_proxy_sc(t, p, mu_shift) for t in T_pts])
    mask = rho_pts > 1e-20
    if mask.sum() < 2:
        return np.nan
    return float(np.polyfit(np.log(T_pts[mask]), np.log(rho_pts[mask]), 1)[0])


def planckian_ratio_sc(T: float, p: BathRPAModel, mu_shift: float) -> float:
    """Gamma_qp / T at omega = 0 with shifted mu."""
    im_sig = im_sigma_electron_shifted(T, T, p, mu_shift=mu_shift,
                                         n_q=61, n_theta=31)
    return 2.0 * abs(im_sig) / max(T, 1e-12)


# =============================================================================
# Per-dispersion configuration
# =============================================================================

DISPERSION_SETUPS = {
    "gapped": {
        "label":  r"gapped: $\Omega_q = \sqrt{\Omega_0^2 + (cq)^2}$",
        "params": dict(dispersion="gapped", Omega0=0.5, c_disp=0.5,
                       damping="ohmic", gamma0=0.1, kF=1.0, m_e=1.0,
                       use_RPA=True),
        "g_range": np.linspace(0.3, 4.0, 14),
    },
    "linear": {
        "label":  r"linear: $\Omega_q = c\,q$",
        "params": dict(dispersion="linear", c_disp=0.5,
                       damping="ohmic", gamma0=0.1, kF=1.0, m_e=1.0,
                       use_RPA=True),
        "g_range": np.linspace(0.3, 2.5, 14),
    },
    "quadratic": {
        "label":  r"quadratic: $\Omega_q = q^2/(2M)$",
        "params": dict(dispersion="quadratic", M=0.5,
                       damping="ohmic", gamma0=0.1, kF=1.0, m_e=1.0,
                       use_RPA=True),
        "g_range": np.linspace(0.3, 4.0, 14),
    },
}


# =============================================================================
# Calibration: doping and r vs g, per dispersion
# =============================================================================

def calibrate_dispersion(disp_name: str, T_ref: float = 0.05) -> dict:
    """Compute r(g) and delta_n(g) for one dispersion via self-consistency."""
    print(f"\n[Calibration] {disp_name}")
    setup = DISPERSION_SETUPS[disp_name]
    g_arr = setup["g_range"]
    r_vals = []
    doping_vals = []
    Z_vals = []
    mu_shifts = []
    converged_flags = []

    for g in g_arr:
        params = dict(setup["params"]); params["g"] = g
        p = BathRPAModel(**params)
        r = control_parameter_r(p)["r"]
        try:
            diag = full_sc_diagnostic(p, T=T_ref, n_q=61, n_theta=31)
            r_vals.append(r)
            doping_vals.append(diag["delta_n_over_n0"])
            Z_vals.append(diag["Z_pole"])
            mu_shifts.append(diag["mu_shift"])
            converged_flags.append(diag["converged"])
            print(f"  g={g:.2f}  r={r:+.4f}  μ_shift={diag['mu_shift']:+.4e}  "
                  f"δn/n0={diag['delta_n_over_n0']:+.4e}  "
                  f"Z={diag['Z_pole']:.3f}  conv={diag['converged']}")
        except Exception as e:
            r_vals.append(np.nan)
            doping_vals.append(np.nan)
            Z_vals.append(np.nan)
            mu_shifts.append(np.nan)
            converged_flags.append(False)
            print(f"  g={g:.2f}  FAILED: {e}")

    return {
        "disp_name":     disp_name,
        "g_arr":         g_arr,
        "r_vals":        np.array(r_vals),
        "doping_vals":   np.array(doping_vals),
        "Z_vals":        np.array(Z_vals),
        "mu_shifts":     np.array(mu_shifts),
        "converged":     np.array(converged_flags),
    }


# =============================================================================
# Figure: doping & r vs g for all 3 dispersions
# =============================================================================

def fig_doping_vs_g(calib_data: dict, savepath: str):
    """Compare delta_n(g) and r(g) across dispersions."""
    fig, axes = plt.subplots(1, 3, figsize=(PRB_DOUBLE, PRB_SINGLE),
                              gridspec_kw={"wspace": 0.32})

    colors = {"gapped": "#1f4e79", "linear": "#2c7a3e", "quadratic": "#c0392b"}

    for disp_name, calib in calib_data.items():
        col = colors[disp_name]
        label = DISPERSION_SETUPS[disp_name]["label"]
        g_arr = calib["g_arr"]
        conv = calib["converged"]
        # Panel (a) doping vs g
        axes[0].plot(g_arr[conv], calib["doping_vals"][conv], "o-",
                      color=col, ms=3.5, label=label)
        axes[0].plot(g_arr[~conv], calib["doping_vals"][~conv], "x",
                      color=col, ms=5, alpha=0.5)
        # Panel (b) r vs g
        axes[1].plot(g_arr[conv], calib["r_vals"][conv], "o-",
                      color=col, ms=3.5)
        axes[1].plot(g_arr[~conv], calib["r_vals"][~conv], "x",
                      color=col, ms=5, alpha=0.5)
        # Panel (c) doping vs r
        # Sort by r for clean line
        sort_idx = np.argsort(calib["r_vals"][conv])
        axes[2].plot(calib["r_vals"][conv][sort_idx],
                      calib["doping_vals"][conv][sort_idx], "o-",
                      color=col, ms=3.5)

    axes[0].set_xlabel(r"coupling $g$")
    axes[0].set_ylabel(r"fractional doping $\delta n / n_0$")
    axes[0].set_title(r"(a) doping vs $g$")
    axes[0].legend(loc="upper left", fontsize=6.5, framealpha=0.9)
    axes[0].grid(True, alpha=0.2)

    axes[1].set_xlabel(r"coupling $g$")
    axes[1].set_ylabel(r"bath control parameter $r$")
    axes[1].set_title(r"(b) $r$ vs $g$")
    axes[1].axhline(0, color="red", lw=0.6, ls="--")
    axes[1].grid(True, alpha=0.2)

    axes[2].set_xlabel(r"$r$")
    axes[2].set_ylabel(r"$\delta n / n_0$")
    axes[2].set_title(r"(c) doping vs $r$")
    axes[2].axvline(0, color="red", lw=0.6, ls="--", label=r"$r = 0$ (QCP)")
    axes[2].legend(loc="upper right", fontsize=6.5, framealpha=0.9)
    axes[2].grid(True, alpha=0.2)

    fig.suptitle(r"Self-consistent doping and bath control parameter "
                  r"across dispersions", y=1.02, fontsize=10)
    fig.savefig(savepath)
    plt.close(fig)
    print(f"[ok] {os.path.basename(savepath)}")


# =============================================================================
# Phase diagram: (delta_n, T) for one dispersion
# =============================================================================

def compute_phase_diagram(disp_name: str, T_arr: np.ndarray,
                           verbose: bool = False) -> dict:
    """For each (g, T), compute (delta_n, Z, Gamma/T, alpha) with self-consistency."""
    print(f"\n[Phase diagram] {disp_name}")
    setup = DISPERSION_SETUPS[disp_name]
    g_arr = setup["g_range"]

    n_g, n_T = len(g_arr), len(T_arr)
    doping_map  = np.full((n_T, n_g), np.nan)
    Z_map       = np.full((n_T, n_g), np.nan)
    gT_map      = np.full((n_T, n_g), np.nan)
    alpha_map   = np.full((n_T, n_g), np.nan)
    r_arr       = np.full(n_g, np.nan)
    mu_shift_map = np.full((n_T, n_g), np.nan)

    # r doesn't depend on T
    for j, g in enumerate(g_arr):
        params = dict(setup["params"]); params["g"] = g
        r_arr[j] = control_parameter_r(BathRPAModel(**params))["r"]

    for i, T in enumerate(T_arr):
        print(f"  T = {T:.4f}  ({i+1}/{n_T})")
        for j, g in enumerate(g_arr):
            params = dict(setup["params"]); params["g"] = g
            p = BathRPAModel(**params)
            try:
                diag = full_sc_diagnostic(p, T, n_q=51, n_theta=27)
                doping_map[i, j] = diag["delta_n_over_n0"]
                Z_map[i, j]      = diag["Z_pole"]
                mu_shift_map[i, j] = diag["mu_shift"]
                gT_map[i, j]     = planckian_ratio_sc(T, p, diag["mu_shift"])
                alpha_map[i, j]  = resistivity_exponent_sc(T, p, diag["mu_shift"])
            except Exception as e:
                if verbose:
                    print(f"    (g={g}, T={T}) FAILED: {e}")
                pass

    return {
        "disp_name":  disp_name,
        "g_arr":      g_arr,
        "T_arr":      T_arr,
        "r_arr":      r_arr,
        "doping_map": doping_map,
        "Z_map":      Z_map,
        "gT_map":     gT_map,
        "alpha_map":  alpha_map,
        "mu_shift_map": mu_shift_map,
    }


# =============================================================================
# Main comparative figure: 3 dispersions x 3 observables
# =============================================================================

def fig_three_dispersions(phase_data: dict, savepath: str):
    """3x3 grid: rows = dispersions, columns = (Z, Gamma/T, alpha)."""
    n_disp = len(phase_data)
    fig, axes = plt.subplots(n_disp, 3, figsize=(PRB_DOUBLE, PRB_DOUBLE * 0.85),
                              gridspec_kw={"hspace": 0.50, "wspace": 0.35})

    if n_disp == 1:
        axes = axes.reshape(1, -1)

    for row, (disp_name, data) in enumerate(phase_data.items()):
        g_arr = data["g_arr"]
        T_arr = data["T_arr"]
        # Use doping as x-axis: convert g to doping for each T (then take median across T)
        doping_1d = np.nanmedian(data["doping_map"], axis=0)
        sort_idx = np.argsort(doping_1d)
        doping_sorted = doping_1d[sort_idx]
        Z_sorted = data["Z_map"][:, sort_idx]
        gT_sorted = data["gT_map"][:, sort_idx]
        alpha_sorted = data["alpha_map"][:, sort_idx]
        r_sorted = data["r_arr"][sort_idx]

        # Restrict to physical doping
        mask = (doping_sorted >= -0.2) & (doping_sorted <= 1.0)
        if mask.sum() < 3:
            mask = np.ones_like(doping_sorted, dtype=bool)

        D = doping_sorted[mask]
        Z_p = Z_sorted[:, mask]
        gT_p = gT_sorted[:, mask]
        alpha_p = alpha_sorted[:, mask]
        r_p = r_sorted[mask]

        D_mesh, T_mesh = np.meshgrid(D, T_arr)

        # Doping where r = 0
        sign_changes = np.where(np.diff(np.sign(r_p)))[0]
        doping_qcp = D[sign_changes[0]] if len(sign_changes) > 0 else None

        # Column 0: Z
        ax = axes[row, 0]
        pcm = ax.pcolormesh(D_mesh, T_mesh, Z_p, shading="auto",
                             cmap="viridis", vmin=0, vmax=1, rasterized=True)
        fig.colorbar(pcm, ax=ax, label=r"$Z$", fraction=0.04, pad=0.02)
        if doping_qcp is not None:
            ax.axvline(doping_qcp, color="yellow", lw=1.2, ls="-.",
                        label=rf"$r=0$ at $\delta n/n_0={doping_qcp:.3f}$")
            ax.legend(loc="upper right", fontsize=5.5, framealpha=0.85)
        ax.set_xlabel(r"$\delta n / n_0$")
        ax.set_ylabel(r"$T / E_F$")
        ax.set_title(r"$Z_{\mathrm{pole}}(\delta n, T)$", fontsize=8)
        # Row label
        if row == 0:
            ax.text(-0.30, 0.5, DISPERSION_SETUPS[disp_name]["label"],
                     transform=ax.transAxes, rotation=90, va="center",
                     ha="center", fontsize=8, weight="bold")
        else:
            ax.text(-0.30, 0.5, DISPERSION_SETUPS[disp_name]["label"],
                     transform=ax.transAxes, rotation=90, va="center",
                     ha="center", fontsize=8, weight="bold")

        # Column 1: Gamma/T
        ax = axes[row, 1]
        gT_safe = np.log10(np.clip(gT_p, 1e-3, 1e2))
        pcm = ax.pcolormesh(D_mesh, T_mesh, gT_safe, shading="auto",
                             cmap="plasma", rasterized=True)
        fig.colorbar(pcm, ax=ax, label=r"$\log_{10}(\Gamma/T)$",
                      fraction=0.04, pad=0.02)
        if doping_qcp is not None:
            ax.axvline(doping_qcp, color="yellow", lw=1.2, ls="-.")
        # Contour for Planckian Gamma/T = 1
        try:
            ax.contour(D_mesh, T_mesh, gT_p, levels=[1.0],
                        colors="white", linewidths=1.0)
        except Exception:
            pass
        ax.set_xlabel(r"$\delta n / n_0$")
        ax.set_ylabel(r"$T / E_F$")
        ax.set_title(r"$\Gamma_{qp}/T$", fontsize=8)

        # Column 2: alpha_rho
        ax = axes[row, 2]
        pcm = ax.pcolormesh(D_mesh, T_mesh, alpha_p, shading="auto",
                             cmap="RdYlGn_r", vmin=0, vmax=2.5, rasterized=True)
        fig.colorbar(pcm, ax=ax, label=r"$\alpha_\rho$",
                      fraction=0.04, pad=0.02)
        if doping_qcp is not None:
            ax.axvline(doping_qcp, color="yellow", lw=1.2, ls="-.")
        for level, color in [(1.0, "darkred"), (2.0, "navy")]:
            try:
                ax.contour(D_mesh, T_mesh, alpha_p, levels=[level],
                            colors=[color], linewidths=0.8, alpha=0.7)
            except Exception:
                pass
        ax.set_xlabel(r"$\delta n / n_0$")
        ax.set_ylabel(r"$T / E_F$")
        ax.set_title(r"$\alpha_\rho$ ($\rho\sim T^{\alpha}$)", fontsize=8)

    fig.suptitle(r"Self-consistent observables on $(\delta n, T)$ plane: "
                  r"comparison across bath dispersions",
                  y=0.99, fontsize=10)
    fig.savefig(savepath)
    plt.close(fig)
    print(f"[ok] {os.path.basename(savepath)}")


# =============================================================================
# Driver
# =============================================================================

def main():
    print("=" * 70)
    print("Self-consistent + multi-dispersion suite")
    print("=" * 70)
    t_start = time.time()
    os.makedirs(OUTDIR, exist_ok=True)

    # --- Calibration: get r(g), doping(g) for all dispersions at one T
    calib_data = {}
    for disp_name in ["gapped", "linear", "quadratic"]:
        calib_data[disp_name] = calibrate_dispersion(disp_name, T_ref=0.05)

    fig_doping_vs_g(calib_data,
                     os.path.join(OUTDIR, "fig_sc_doping_vs_g.pdf"))

    # --- Full (delta_n, T) phase diagrams: smaller grid, all 3 dispersions
    T_arr = np.geomspace(0.01, 0.3, 8)
    phase_data = {}
    for disp_name in ["gapped", "linear", "quadratic"]:
        phase_data[disp_name] = compute_phase_diagram(disp_name, T_arr)

    fig_three_dispersions(phase_data,
                            os.path.join(OUTDIR, "fig_sc_three_dispersions.pdf"))

    # Save raw data
    np.savez(os.path.join(OUTDIR, "sc_dispersion_data.npz"),
             **{f"{disp}_{key}": v
                for disp, data in phase_data.items()
                for key, v in data.items()
                if isinstance(v, np.ndarray)})

    t_total = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"Suite complete. Total time: {t_total:.1f} s")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
