"""
fig_crossover_suite.py
----------------------
Comprehensive characterization of the doping-induced crossover.

Level 1 — Kinematic characterization:
  Fig C: vth_Sigma(delta_n) and alpha_rho(delta_n) at 3 temperatures
  Fig D: T*(delta_n) crossover line (alpha_rho = 1.5)

Level 2 — Universal scaling:
  Fig E: Im Sigma(omega, T) scaling collapse for several delta_n values
  Fig F: Gamma_qp/T vs delta_n + Planckian locus

Self-consistent throughout. No contours, no R² filter.
"""

from __future__ import annotations
import os, time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

from bath_dressed import BathRPAModel, control_parameter_r
from self_consistent import (im_sigma_electron_shifted,
                              solve_self_consistent_shift)

rcParams.update({
    "font.family": "serif", "font.size": 9,
    "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "xtick.major.size": 3.5, "ytick.major.size": 3.5,
    "axes.linewidth": 0.6, "lines.linewidth": 1.2,
    "savefig.dpi": 300, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02, "text.usetex": False,
    "mathtext.fontset": "cm",
})

PRB_SINGLE = 3.375
PRB_DOUBLE = 7.0
OUTDIR = "."


# =============================================================================
# Core diagnostic at (g, T): mu_shift, doping, Im Sigma curve, rho, alpha
# =============================================================================

def diagnose_point(g: float, T: float, base_params: dict,
                    n_q: int = 41, n_theta: int = 21,
                    omega_grid: np.ndarray = None,
                    n_omega_rho: int = 17) -> dict:
    """
    For one (g, T): solve self-consistency, compute Im Sigma curve,
    extract vth and rho, return everything.
    """
    if omega_grid is None:
        omega_grid = np.geomspace(1e-4, 1.0, 24)

    params = dict(base_params); params["g"] = g
    p = BathRPAModel(**params)

    # Self-consistency
    sc = solve_self_consistent_shift(p, T, n_q=n_q, n_theta=n_theta,
                                       max_iter=20, tol=1e-4, damping=0.3)
    mu_shift = sc["mu_shift"]
    converged = sc["converged"]

    # Doping
    N_0 = p.m_e * p.kF / (2.0 * np.pi ** 2)
    n_0 = p.kF ** 3 / (3.0 * np.pi ** 2)
    delta_n = float(-N_0 * mu_shift / n_0)

    # Im Sigma curve
    im_sig = np.array([
        im_sigma_electron_shifted(om, T, p, mu_shift=mu_shift,
                                   n_q=n_q, n_theta=n_theta)
        for om in omega_grid
    ])

    # vth from log-log fit in [2T, 1.0]
    om_lo = max(2.0 * T, 1e-4)
    om_hi = 1.0
    mask = (omega_grid >= om_lo) & (omega_grid <= om_hi) & (np.abs(im_sig) > 0)
    if mask.sum() >= 3:
        slope, _ = np.polyfit(np.log(omega_grid[mask]),
                               np.log(np.abs(im_sig[mask])), 1)
        vth = float(slope)
    else:
        vth = np.nan

    # rho(T) — single T value here
    n = n_omega_rho if n_omega_rho % 2 == 1 else n_omega_rho + 1
    om_max = max(5.0 * T, 1e-3)
    om_rho = np.linspace(1e-5, om_max, n)
    df_dom = 1.0 / (4.0 * T * np.cosh(om_rho / (2.0 * T)) ** 2)
    ims_rho = np.array([
        im_sigma_electron_shifted(om, T, p, mu_shift=mu_shift,
                                   n_q=n_q, n_theta=n_theta)
        for om in om_rho
    ])
    rho = float(np.trapezoid(df_dom * np.abs(ims_rho), om_rho))

    # Gamma_qp/T at omega = T
    ims_at_T = im_sigma_electron_shifted(T, T, p, mu_shift=mu_shift,
                                           n_q=n_q, n_theta=n_theta)
    gamma_over_T = 2.0 * abs(ims_at_T) / max(T, 1e-12)

    return {
        "g":          g,
        "T":          T,
        "mu_shift":   mu_shift,
        "delta_n":    delta_n,
        "converged":  converged,
        "omega_grid": omega_grid,
        "im_sigma":   im_sig,
        "vth":        vth,
        "rho":        rho,
        "gamma_over_T": gamma_over_T,
    }


def compute_rho_curve(g: float, T_array: np.ndarray, base_params: dict,
                       n_q: int = 41, n_theta: int = 21) -> dict:
    """rho(T) curve at fixed g, returns rho array + delta_n at each T."""
    rho_vals = np.zeros_like(T_array)
    delta_n_vals = np.zeros_like(T_array)
    for i, T in enumerate(T_array):
        d = diagnose_point(g, T, base_params, n_q=n_q, n_theta=n_theta)
        rho_vals[i] = d["rho"]
        delta_n_vals[i] = d["delta_n"]
    return {"T_array": T_array, "rho": rho_vals, "delta_n": delta_n_vals,
            "delta_n_avg": float(np.nanmean(delta_n_vals))}


# =============================================================================
# FIG C: vth(delta_n) and alpha(delta_n) at 3 temperatures
# =============================================================================

def fig_C_kinematic(base_params: dict, g_array: np.ndarray,
                     T_values: list, savepath: str) -> dict:
    """
    Two panels: vth_Sigma(delta_n) and alpha_rho(delta_n), three T each.

    For alpha_rho: use 3-point sliding fit on a finer T grid centered at each T_value.
    """
    print(f"\n[Fig C] kinematic crossover, {len(T_values)} T values, {len(g_array)} g")

    results = {}
    # Build local T grids for sliding alpha fits (3 points per T_value)
    for T_ref in T_values:
        T_triplet = np.array([T_ref * 0.8, T_ref, T_ref * 1.25])
        data_T = []
        for i_g, g in enumerate(g_array):
            print(f"  T_ref={T_ref:.3f}, g={g:.2f} ({i_g+1}/{len(g_array)})")
            # vth: from omega-fit at T_ref
            d_ref = diagnose_point(g, T_ref, base_params)
            # alpha: 3-point sliding fit
            rho_triplet = []
            dn_triplet = []
            for T in T_triplet:
                d_tmp = diagnose_point(g, T, base_params, n_omega_rho=13)
                rho_triplet.append(d_tmp["rho"])
                dn_triplet.append(d_tmp["delta_n"])
            rho_triplet = np.array(rho_triplet)
            valid = (rho_triplet > 0) & np.isfinite(rho_triplet)
            if valid.sum() >= 2:
                alpha = float(np.polyfit(np.log(T_triplet[valid]),
                                          np.log(rho_triplet[valid]), 1)[0])
            else:
                alpha = np.nan
            data_T.append({
                "g":        g,
                "delta_n":  d_ref["delta_n"],
                "vth":      d_ref["vth"],
                "alpha":    alpha,
            })
        results[T_ref] = data_T

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(PRB_DOUBLE, PRB_SINGLE * 1.15),
                              gridspec_kw={"wspace": 0.30})
    colors = ["#1f4e79", "#c0392b", "#2c7a3e"]
    markers = ["o", "s", "^"]

    for k, T_ref in enumerate(T_values):
        data = results[T_ref]
        dn = np.array([d["delta_n"] for d in data])
        vth = np.array([d["vth"] for d in data])
        alpha = np.array([d["alpha"] for d in data])
        # Sort by doping
        idx = np.argsort(dn)
        dn_s, vth_s, alpha_s = dn[idx], vth[idx], alpha[idx]

        axes[0].plot(dn_s, vth_s, marker=markers[k], color=colors[k],
                      ms=5, lw=1.2, label=rf"$T = {T_ref}$")
        axes[1].plot(dn_s, alpha_s, marker=markers[k], color=colors[k],
                      ms=5, lw=1.2, label=rf"$T = {T_ref}$")

    # Reference lines
    for ax, label_FL, label_MFL in [(axes[0], "FL", "MFL"), (axes[1], "FL", "MFL")]:
        ax.axhline(2.0, color="gray", lw=0.6, ls="--", alpha=0.6)
        ax.axhline(1.0, color="gray", lw=0.6, ls=":", alpha=0.6)
        ax.text(0.02, 2.05, label_FL, fontsize=7, color="0.4")
        ax.text(0.02, 1.05, label_MFL, fontsize=7, color="0.4")
        ax.grid(True, alpha=0.2)

    axes[0].set_xlabel(r"$\delta n / n_0$")
    axes[0].set_ylabel(r"$\vartheta_\Sigma$")
    axes[0].set_title(r"(a) Self-energy exponent")
    axes[0].set_ylim(-0.2, 2.7)
    axes[0].legend(loc="upper right", fontsize=7, framealpha=0.9)

    axes[1].set_xlabel(r"$\delta n / n_0$")
    axes[1].set_ylabel(r"$\alpha_\rho$")
    axes[1].set_title(r"(b) Resistivity exponent")
    axes[1].set_ylim(-0.2, 2.7)
    axes[1].legend(loc="upper right", fontsize=7, framealpha=0.9)

    fig.suptitle(r"Crossover kinematics: exponents vs doping at three temperatures",
                 y=1.02, fontsize=10)
    fig.savefig(savepath)
    plt.close(fig)
    print(f"[ok] {os.path.basename(savepath)}")
    return results


# =============================================================================
# FIG D: T*(delta_n) crossover line
# =============================================================================

def fig_D_Tstar(base_params: dict, g_array: np.ndarray,
                 T_array: np.ndarray, savepath: str,
                 alpha_threshold: float = 1.5) -> dict:
    """
    Build alpha_rho(delta_n, T) map, extract T* where alpha = 1.5.

    For each delta_n (column), find T* by linear interpolation in
    the alpha(T) curve to find alpha = 1.5 crossing.
    """
    print(f"\n[Fig D] T*(delta_n) — {len(g_array)} g × {len(T_array)} T")

    n_g, n_T = len(g_array), len(T_array)
    delta_n_map = np.full((n_T, n_g), np.nan)
    alpha_map   = np.full((n_T, n_g), np.nan)

    # For each g, compute rho(T) curve, then sliding alpha
    for j, g in enumerate(g_array):
        print(f"  g = {g:.2f}  ({j+1}/{n_g})")
        rho_T = np.zeros(n_T)
        for i, T in enumerate(T_array):
            d = diagnose_point(g, T, base_params, n_omega_rho=13)
            rho_T[i] = d["rho"]
            delta_n_map[i, j] = d["delta_n"]

        # Sliding 3-point alpha
        log_T = np.log(T_array)
        log_rho = np.log(np.maximum(rho_T, 1e-30))
        for i in range(n_T):
            lo = max(0, i - 1)
            hi = min(n_T, i + 2)
            if hi - lo >= 2 and (rho_T[lo:hi] > 0).all():
                slope = np.polyfit(log_T[lo:hi], log_rho[lo:hi], 1)[0]
                alpha_map[i, j] = float(slope)

    # Median doping per g
    dn_1d = np.nanmedian(delta_n_map, axis=0)
    sort_idx = np.argsort(dn_1d)
    D = dn_1d[sort_idx]
    alpha_sorted = alpha_map[:, sort_idx]

    # For each column (doping), find T* where alpha crosses alpha_threshold
    T_star = np.full(len(D), np.nan)
    for j in range(len(D)):
        alpha_col = alpha_sorted[:, j]
        valid = np.isfinite(alpha_col)
        if valid.sum() < 2:
            continue
        T_col = T_array[valid]
        a_col = alpha_col[valid]
        # alpha increases with T typically; T* is where it crosses 1.5
        # Find sign changes of (alpha - threshold)
        diff = a_col - alpha_threshold
        sign_changes = np.where(np.diff(np.sign(diff)) != 0)[0]
        if len(sign_changes) == 0:
            # No crossing in our T range
            continue
        # Use the first crossing
        i0 = sign_changes[0]
        # Linear interp in log T
        T_lo, T_hi = T_col[i0], T_col[i0 + 1]
        a_lo, a_hi = a_col[i0], a_col[i0 + 1]
        if abs(a_hi - a_lo) < 1e-12:
            T_star[j] = T_lo
        else:
            log_Tstar = (np.log(T_lo) + (alpha_threshold - a_lo) /
                         (a_hi - a_lo) * (np.log(T_hi) - np.log(T_lo)))
            T_star[j] = float(np.exp(log_Tstar))

    # Plot
    fig, ax = plt.subplots(figsize=(PRB_SINGLE * 1.5, PRB_SINGLE * 1.1))

    # Background: alpha_rho map (no contours, no filter)
    D_mesh, T_mesh = np.meshgrid(D, T_array)
    pcm = ax.pcolormesh(D_mesh, T_mesh, alpha_sorted, shading="auto",
                         cmap="RdYlGn_r", vmin=0.0, vmax=2.0, rasterized=True)
    fig.colorbar(pcm, ax=ax, label=r"$\alpha_\rho$")

    # T* line
    valid_Tstar = np.isfinite(T_star)
    ax.plot(D[valid_Tstar], T_star[valid_Tstar], "ko-", ms=5, lw=1.5,
             label=rf"$T^*(\delta n)$: $\alpha_\rho = {alpha_threshold}$")

    ax.set_xlabel(r"$\delta n / n_0$")
    ax.set_ylabel(r"$T / E_F$")
    ax.set_title(rf"Crossover scale $T^*(\delta n)$")
    ax.legend(loc="upper right", fontsize=7, framealpha=0.9)

    fig.savefig(savepath)
    plt.close(fig)
    print(f"[ok] {os.path.basename(savepath)}")
    return {"D": D, "T_array": T_array, "alpha_map": alpha_sorted,
            "T_star": T_star}


# =============================================================================
# FIG E: Im Sigma scaling collapse at fixed delta_n
# =============================================================================

def fig_E_scaling_collapse(base_params: dict, g_values_for_collapse: list,
                            T_array: np.ndarray, savepath: str) -> dict:
    """
    For each g (which gives a delta_n), compute Im Sigma(omega) at several
    T values. Test scaling collapse:
        |Im Sigma(omega, T)| / T^vth  vs  omega/T
    by trying multiple vth candidates.
    """
    print(f"\n[Fig E] Scaling collapse: {len(g_values_for_collapse)} g, {len(T_array)} T")

    results = []
    for g in g_values_for_collapse:
        print(f"  g = {g:.2f}")
        data_per_T = []
        for T in T_array:
            d = diagnose_point(g, T, base_params)
            data_per_T.append(d)
        # delta_n is approximately fixed (varies slightly with T)
        dn = np.nanmedian([d["delta_n"] for d in data_per_T])
        results.append({"g": g, "delta_n": dn, "data_per_T": data_per_T})

    # Plot: 2 rows × len(g_values_for_collapse) cols
    n_g = len(g_values_for_collapse)
    fig, axes = plt.subplots(2, n_g, figsize=(PRB_DOUBLE, PRB_DOUBLE * 0.65),
                              gridspec_kw={"hspace": 0.40, "wspace": 0.35})

    cmap = plt.get_cmap("plasma")
    T_min, T_max = T_array.min(), T_array.max()

    # Try several vth values; pick the one that gives best visual collapse
    # We'll plot two rows: top = raw curves, bottom = best collapse
    vth_candidates = [1.0, 1.5, 2.0]

    for col, res in enumerate(results):
        g = res["g"]
        dn = res["delta_n"]
        data = res["data_per_T"]

        # Top: raw Im Sigma(omega) curves
        ax = axes[0, col]
        for d in data:
            T = d["T"]
            color = cmap((T - T_min) / max(T_max - T_min, 1e-12))
            om = d["omega_grid"]
            ims = np.abs(d["im_sigma"])
            keep = (om < 0.5)
            ax.loglog(om[keep], ims[keep], "o-", ms=2.5, lw=0.9, color=color,
                       label=rf"$T={T:.3f}$")
        ax.set_xlabel(r"$\omega / E_F$")
        if col == 0:
            ax.set_ylabel(r"$|\mathrm{Im}\,\Sigma|$")
        ax.set_title(rf"$\delta n/n_0 = {dn:+.3f}$ ($g={g}$)", fontsize=7.5)
        ax.legend(loc="lower right", fontsize=5.5, framealpha=0.9)
        ax.grid(True, which="both", alpha=0.15)

        # Bottom: pick best vth by computing collapse residual
        # For each candidate vth: rescale, compute spread across T
        best_vth = None
        best_spread = np.inf
        for vth in vth_candidates:
            # Compute scaled curves on a common omega/T grid
            xT_common = np.geomspace(2.0, 20.0, 25)  # omega/T from 2 to 20
            y_curves = []
            for d in data:
                T = d["T"]
                om = d["omega_grid"]
                ims = np.abs(d["im_sigma"])
                xT = om / T
                y_scaled = ims / (T ** vth)
                # Interpolate onto common grid (in log space)
                in_range = (xT > xT_common.min()) & (xT < xT_common.max())
                if in_range.sum() < 5:
                    continue
                from scipy.interpolate import interp1d
                interp = interp1d(np.log(xT[in_range]),
                                   np.log(y_scaled[in_range] + 1e-30),
                                   bounds_error=False, fill_value=np.nan)
                y_interp = np.exp(interp(np.log(xT_common)))
                y_curves.append(y_interp)
            if len(y_curves) < 2:
                continue
            y_curves = np.array(y_curves)
            # Spread: std of log(y) across T, averaged over omega/T
            with np.errstate(divide="ignore", invalid="ignore"):
                logy = np.log(y_curves)
            spread = np.nanmean(np.nanstd(logy, axis=0))
            if spread < best_spread:
                best_spread = spread
                best_vth = vth

        # Plot best collapse
        ax = axes[1, col]
        for d in data:
            T = d["T"]
            color = cmap((T - T_min) / max(T_max - T_min, 1e-12))
            om = d["omega_grid"]
            ims = np.abs(d["im_sigma"])
            xT = om / T
            y_scaled = ims / (T ** best_vth)
            keep = (xT > 0.1) & (xT < 30)
            ax.loglog(xT[keep], y_scaled[keep], "o-", ms=2.5, lw=0.9,
                       color=color, label=rf"$T={T:.3f}$")
        ax.set_xlabel(r"$\omega / T$")
        if col == 0:
            ax.set_ylabel(r"$|\mathrm{Im}\,\Sigma| / T^\vartheta$")
        ax.set_title(rf"best $\vartheta = {best_vth}$ (spread={best_spread:.3f})",
                     fontsize=7.5)
        ax.grid(True, which="both", alpha=0.15)

    fig.suptitle(r"Scaling collapse test: $|\mathrm{Im}\,\Sigma|/T^\vartheta$ vs $\omega/T$",
                 y=1.00, fontsize=10)
    fig.savefig(savepath)
    plt.close(fig)
    print(f"[ok] {os.path.basename(savepath)}")
    return results


# =============================================================================
# FIG F: Gamma_qp/T vs delta_n
# =============================================================================

def fig_F_planckian(base_params: dict, g_array: np.ndarray,
                     T_values: list, savepath: str) -> dict:
    """
    Gamma_qp/T vs delta_n at multiple T. Identify Planckian regime (Gamma/T = 1).
    """
    print(f"\n[Fig F] Planckian diagnostic, {len(T_values)} T × {len(g_array)} g")

    data = {}
    for T in T_values:
        results_T = []
        for j, g in enumerate(g_array):
            d = diagnose_point(g, T, base_params)
            results_T.append({"g": g, "delta_n": d["delta_n"],
                              "gamma_over_T": d["gamma_over_T"]})
        data[T] = results_T

    fig, ax = plt.subplots(figsize=(PRB_SINGLE * 1.5, PRB_SINGLE * 1.1))

    colors = ["#1f4e79", "#c0392b", "#2c7a3e"]
    markers = ["o", "s", "^"]

    for k, T_ref in enumerate(T_values):
        res = data[T_ref]
        dn = np.array([r["delta_n"] for r in res])
        gT = np.array([r["gamma_over_T"] for r in res])
        idx = np.argsort(dn)
        ax.semilogy(dn[idx], gT[idx], marker=markers[k], color=colors[k],
                     ms=5, lw=1.2, label=rf"$T = {T_ref}$")

    # Planckian line
    ax.axhline(1.0, color="black", lw=0.8, ls="--", alpha=0.7,
                label=r"Planckian $\Gamma/T = 1$")

    ax.set_xlabel(r"$\delta n / n_0$")
    ax.set_ylabel(r"$\Gamma_{qp} / T$")
    ax.set_title(r"Planckian diagnostic across doping")
    ax.legend(loc="lower right", fontsize=7, framealpha=0.9)
    ax.grid(True, which="both", alpha=0.2)

    fig.savefig(savepath)
    plt.close(fig)
    print(f"[ok] {os.path.basename(savepath)}")
    return data


# =============================================================================
# Driver
# =============================================================================

def main():
    print("=" * 70)
    print("Crossover characterization suite (Levels 1+2)")
    print("=" * 70)
    t_start = time.time()
    os.makedirs(OUTDIR, exist_ok=True)

    base_params = dict(
        dispersion="gapped", Omega0=0.5, c_disp=0.5,
        damping="ohmic", gamma0=0.1, kF=1.0, m_e=1.0,
        use_RPA=True,
    )

    T_values = [0.02, 0.05, 0.10]

    # Fig C: kinematic exponents at 3 T
    g_array_C = np.linspace(0.5, 2.5, 10)
    fig_C_kinematic(base_params, g_array_C, T_values,
                     savepath=os.path.join(OUTDIR, "fig_C_kinematic.pdf"))

    # Fig D: T* line
    g_array_D = np.linspace(0.5, 2.5, 10)
    T_array_D = np.geomspace(0.015, 0.2, 9)
    fig_D_Tstar(base_params, g_array_D, T_array_D,
                 savepath=os.path.join(OUTDIR, "fig_D_Tstar.pdf"))

    # Fig E: scaling collapse at 4 representative dopings
    g_values_E = [1.0, 1.5, 1.9, 2.1]  # roughly: deep FL, FL edge, near QCP, beyond
    T_array_E = np.geomspace(0.02, 0.15, 6)
    fig_E_scaling_collapse(base_params, g_values_E, T_array_E,
                            savepath=os.path.join(OUTDIR, "fig_E_collapse.pdf"))

    # Fig F: Planckian
    g_array_F = np.linspace(0.5, 2.5, 12)
    fig_F_planckian(base_params, g_array_F, T_values,
                     savepath=os.path.join(OUTDIR, "fig_F_planckian.pdf"))

    t_total = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"Done. Total time: {t_total:.1f}s = {t_total/60:.1f} min")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
