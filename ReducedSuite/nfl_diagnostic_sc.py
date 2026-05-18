"""
nfl_diagnostic_sc.py
--------------------
Direct measurement of the NFL exponent vth_Sigma in the self-consistent
treatment, designed to answer the question:
   "Is the gas an MFL/Varma-like NFL (Z = 1, Im Sigma ~ omega) or a
    Fermi liquid (Z = 1, Im Sigma ~ omega^2)?"

For each (g, T) point we:
  1. Solve self-consistent mu_shift
  2. Compute Im Sigma(omega) on a wide log-spaced grid
  3. Fit Im Sigma in a clean IR window omega > 2T (above thermal scale)
     to two competing forms:
       (a) power law:   Im Sigma = A * omega^vth
       (b) MFL Varma:   Im Sigma = a * omega * log(b/|omega|) + c
  4. Compare goodness of fit and report which form wins.

Following Patel-Lunts-Sachdev 2024 (PNAS), the MFL form is the smoking
gun signature of strange metal physics with surviving quasiparticle.

If vth significantly < 2 in regions where Z = 1: evidence of NFL with
surviving quasiparticle (MFL/Varma-like).

Output:
  fig_nfl_curves.pdf       : sample Im Sigma curves at multiple g values
  fig_nfl_vth_map.pdf      : vth_Sigma(delta_n, T) map (the central result)
  fig_nfl_form_test.pdf    : power law vs MFL Varma fit comparison
"""

from __future__ import annotations
import os, time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.optimize import curve_fit

from bath_dressed import BathRPAModel, control_parameter_r
from self_consistent import (full_sc_diagnostic, im_sigma_electron_shifted,
                              solve_self_consistent_shift)

rcParams.update({
    "font.family": "serif", "font.size": 9,
    "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
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
# Wide-omega Im Sigma scan, self-consistent
# =============================================================================

def imsig_curve_sc(p: BathRPAModel, T: float,
                    omega_grid: np.ndarray = None,
                    n_q: int = 51, n_theta: int = 27,
                    sc_max_iter: int = 30, sc_tol: float = 1e-4) -> dict:
    """
    Solve mu_shift self-consistently, then compute Im Sigma on wide log-omega
    grid. Returns the full curve and convergence info.
    """
    if omega_grid is None:
        # Span 3 decades for clean log-log fit
        omega_grid = np.geomspace(1e-4, 0.3, 35)

    sc = solve_self_consistent_shift(p, T,
                                       n_q=n_q, n_theta=n_theta,
                                       max_iter=sc_max_iter, tol=sc_tol,
                                       damping=0.3)
    mu_shift = sc["mu_shift"]

    im_sig = np.array([
        im_sigma_electron_shifted(om, T, p, mu_shift=mu_shift,
                                   n_q=n_q, n_theta=n_theta)
        for om in omega_grid
    ])
    return {
        "omega_grid":  omega_grid,
        "im_sigma":    im_sig,
        "mu_shift":    mu_shift,
        "converged":   sc["converged"],
        "n_iter":      sc["n_iter"],
    }


# =============================================================================
# Two competing fit forms
# =============================================================================

def fit_power_law(omegas: np.ndarray, imsig: np.ndarray,
                  window: tuple, subtract_offset: bool = True) -> dict:
    """
    Fit |Im Sigma(omega) - Im Sigma(omega_min)| = A * omega^vth in a clean window.

    Following Patel-Sachdev 2024 (Fig. 5 caption): the relevant scaling is of
    the omega-dependent part, after subtracting the residual scattering offset:
        n = -d ln(|Im Sigma(omega) - Im Sigma(0)|) / d ln omega

    Returns vth, A, R^2, n_pts, residuals, offset.
    """
    om_lo, om_hi = window
    mask = ((omegas >= om_lo) & (omegas <= om_hi)
            & np.isfinite(imsig))
    if mask.sum() < 4:
        return {"vth": np.nan, "A": np.nan, "r2": np.nan,
                "n_pts": int(mask.sum()), "form": "power_law",
                "offset": np.nan, "converged": False}

    om_fit = omegas[mask]
    imS_fit = imsig[mask]

    # Subtract offset: use the smallest-omega value as the residual
    if subtract_offset:
        offset = float(imsig[0]) if len(imsig) > 0 else 0.0
        imS_subtracted = imS_fit - offset
    else:
        offset = 0.0
        imS_subtracted = imS_fit

    # Only keep points where the omega-dependent part has well-defined sign
    abs_imS = np.abs(imS_subtracted)
    valid = abs_imS > 1e-30
    if valid.sum() < 4:
        return {"vth": np.nan, "A": np.nan, "r2": np.nan,
                "n_pts": int(valid.sum()), "form": "power_law",
                "offset": offset, "converged": False}

    log_om = np.log(om_fit[valid])
    log_imS = np.log(abs_imS[valid])
    slope, intercept = np.polyfit(log_om, log_imS, 1)
    y_pred = slope * log_om + intercept
    ss_res = np.sum((log_imS - y_pred) ** 2)
    ss_tot = np.sum((log_imS - log_imS.mean()) ** 2)
    r2 = 1.0 - ss_res / max(ss_tot, 1e-30)
    return {
        "vth":       float(slope),
        "A":         float(np.exp(intercept)),
        "r2":        float(r2),
        "n_pts":     int(valid.sum()),
        "form":      "power_law",
        "offset":    offset,
        "converged": r2 > 0.85,
    }


def fit_MFL_varma(omegas: np.ndarray, imsig: np.ndarray,
                  window: tuple) -> dict:
    """
    Fit |Im Sigma| = a * omega * log(b / |omega|) + c in a clean window.

    Patel-Sachdev 2024 (Fig. 8): a in [0.083, 0.088], b in [41, 46],
    c in [0.04, 0.12]. Their b is far above the omega range, ensuring the
    log is positive throughout the fitting window.

    Returns a, b, c, R^2, n_pts.
    """
    om_lo, om_hi = window
    mask = ((omegas >= om_lo) & (omegas <= om_hi)
            & np.isfinite(imsig))
    if mask.sum() < 4:
        return {"a": np.nan, "b": np.nan, "c": np.nan, "r2": np.nan,
                "n_pts": int(mask.sum()), "form": "MFL_varma",
                "converged": False}

    om_fit = omegas[mask]
    y_fit = np.abs(imsig[mask])

    def varma_form(om, a, b, c):
        # Guard against b/om <= 0 (would give log of zero/negative)
        ratio = np.clip(b / np.abs(om), 1.001, 1e30)
        return a * om * np.log(ratio) + c

    # Initial guess: estimate scaling from upper portion of window
    a0 = max((y_fit[-1] - y_fit[0]) / (om_fit[-1] - om_fit[0]), 1e-6)
    b0 = max(om_fit.max() * 5, 1.0)  # UV cutoff above fit range
    c0 = max(y_fit[0], 1e-30)

    try:
        # Bounds: a > 0, b > om_hi (UV cutoff above range), c >= 0
        bounds_lo = [1e-10, om_fit.max() * 1.05, 0]
        bounds_hi = [1e3, 1e5, max(y_fit) * 2]
        popt, _ = curve_fit(varma_form, om_fit, y_fit,
                            p0=[a0, b0, c0],
                            bounds=(bounds_lo, bounds_hi),
                            maxfev=8000)
        a, b, c = popt
        y_pred = varma_form(om_fit, a, b, c)
        ss_res = np.sum((y_fit - y_pred) ** 2)
        ss_tot = np.sum((y_fit - y_fit.mean()) ** 2)
        r2 = 1.0 - ss_res / max(ss_tot, 1e-30)
        return {
            "a":         float(a),
            "b":         float(b),
            "c":         float(c),
            "r2":        float(r2),
            "n_pts":     int(mask.sum()),
            "form":      "MFL_varma",
            "converged": r2 > 0.85 and a > 0,
        }
    except Exception as e:
        return {"a": np.nan, "b": np.nan, "c": np.nan, "r2": np.nan,
                "n_pts": int(mask.sum()), "form": "MFL_varma",
                "converged": False, "error": str(e)}


def fit_both_and_compare(omegas: np.ndarray, imsig: np.ndarray,
                         T: float,
                         window_factor: tuple = (2.0, 30.0)) -> dict:
    """
    Fit both power-law and Varma MFL forms in the window
        omega in [window_factor[0] * T, min(window_factor[1] * T, omega_max)]

    Returns combined dict with both fits and a 'best_form' flag based on R^2.
    """
    om_lo = window_factor[0] * T
    om_hi = min(window_factor[1] * T, omegas.max() * 0.9)
    if om_hi <= om_lo:
        # Fallback: use upper portion of omega grid
        om_lo = omegas[max(len(omegas) // 3, 1)]
        om_hi = omegas[-2]

    window = (om_lo, om_hi)
    pl = fit_power_law(omegas, imsig, window)
    mfl = fit_MFL_varma(omegas, imsig, window)

    # Decide which is "best" by R^2 in the same domain
    # Both fits use the same mask (window), so R^2 directly comparable
    best = "tie"
    if pl["converged"] and mfl["converged"]:
        if pl["r2"] > mfl["r2"] + 0.01:
            best = "power_law"
        elif mfl["r2"] > pl["r2"] + 0.01:
            best = "MFL_varma"
        else:
            best = "tie"
    elif pl["converged"]:
        best = "power_law"
    elif mfl["converged"]:
        best = "MFL_varma"
    else:
        best = "neither"

    return {
        "power_law":   pl,
        "MFL_varma":   mfl,
        "best_form":   best,
        "window":      window,
        "T":           T,
    }


# =============================================================================
# Sample curves at representative g, fixed T
# =============================================================================

def plot_imsig_curves_with_fits(base_params: dict, T_fix: float = 0.05,
                                 g_values: list = None, savepath: str = None):
    """
    Plot Im Sigma(omega) at several g values for fixed T, log-log.
    Overlay both power-law and MFL Varma fits.
    """
    print(f"\n[Curves] Im Sigma + fits at T = {T_fix}")

    if g_values is None:
        g_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    fig, axes = plt.subplots(2, 3, figsize=(PRB_DOUBLE, PRB_DOUBLE * 0.6),
                              gridspec_kw={"hspace": 0.40, "wspace": 0.30})

    results = []
    for idx, g in enumerate(g_values):
        ax = axes[idx // 3, idx % 3]
        params = dict(base_params); params["g"] = g
        p = BathRPAModel(**params)

        curve = imsig_curve_sc(p, T_fix, n_q=51, n_theta=27)
        om = curve["omega_grid"]
        imS = np.abs(curve["im_sigma"])

        # Mask out zeros
        valid = imS > 1e-30
        ax.loglog(om[valid], imS[valid], "o", ms=3, color="#1f4e79",
                   label="data")

        # Vertical line at omega = T (thermal scale)
        ax.axvline(T_fix, color="gray", lw=0.6, ls=":", alpha=0.7)
        ax.text(T_fix * 1.1, imS[valid].min() * 2,
                rf"$\omega = T$", fontsize=6, color="gray")

        # Fits
        fit = fit_both_and_compare(om, curve["im_sigma"], T_fix)
        pl, mfl = fit["power_law"], fit["MFL_varma"]
        om_fit_range = np.geomspace(fit["window"][0], fit["window"][1], 40)

        if pl["converged"]:
            ax.loglog(om_fit_range, pl["A"] * om_fit_range ** pl["vth"],
                       "-", color="#c0392b", lw=1.0,
                       label=rf"$\omega^{{{pl['vth']:.2f}}}$ R$^2$={pl['r2']:.2f}")

        if mfl["converged"]:
            varma_curve = (mfl["a"] * om_fit_range *
                           np.log(mfl["b"] / np.abs(om_fit_range)) + mfl["c"])
            varma_curve = np.maximum(varma_curve, 1e-30)
            ax.loglog(om_fit_range, varma_curve,
                       "--", color="#2c7a3e", lw=1.0,
                       label=rf"MFL R$^2$={mfl['r2']:.2f}")

        # Get r for this g
        r = control_parameter_r(p)["r"]
        ax.set_title(rf"$g={g:.2f}$  ($r={r:+.2f}$, $\mu_s={curve['mu_shift']:+.2e}$)",
                      fontsize=7)
        ax.set_xlabel(r"$\omega / E_F$")
        ax.set_ylabel(r"$|\mathrm{Im}\,\Sigma^R(\omega)|$")
        ax.legend(loc="lower right", fontsize=5.5, framealpha=0.9)
        ax.grid(True, which="both", alpha=0.15)

        results.append({"g": g, "r": r, "fit": fit, "curve": curve})

    fig.suptitle(rf"Im $\Sigma(\omega)$ vs $g$ at $T={T_fix}$, with power-law and MFL fits",
                 y=1.00, fontsize=10)
    fig.savefig(savepath)
    plt.close(fig)
    print(f"[ok] {os.path.basename(savepath)}")
    return results


# =============================================================================
# Build vth(delta_n, T) and best-form maps
# =============================================================================

def build_vth_map(base_params: dict, savepath_data: str = None):
    """
    For (g, T) grid, compute:
      - delta_n (doping)
      - Z_pole
      - vth_Sigma (power-law fit)
      - MFL_a, MFL_b, MFL_c (Varma fit)
      - best_form_flag

    Save raw data for downstream plotting.
    """
    print("\n[Map] Building vth_Sigma(delta_n, T) map")
    g_arr = np.linspace(0.5, 3.5, 12)
    T_arr = np.geomspace(0.02, 0.20, 8)

    n_g, n_T = len(g_arr), len(T_arr)
    delta_n_map = np.full((n_T, n_g), np.nan)
    Z_map       = np.full((n_T, n_g), np.nan)
    vth_map     = np.full((n_T, n_g), np.nan)
    vth_r2_map  = np.full((n_T, n_g), np.nan)
    mfl_r2_map  = np.full((n_T, n_g), np.nan)
    best_form_map = np.full((n_T, n_g), -1)  # -1=neither, 0=power, 1=MFL, 2=tie
    r_arr       = np.full(n_g, np.nan)

    # r doesn't depend on T
    for j, g in enumerate(g_arr):
        params = dict(base_params); params["g"] = g
        r_arr[j] = control_parameter_r(BathRPAModel(**params))["r"]

    t0 = time.time()
    for i, T in enumerate(T_arr):
        print(f"  T = {T:.4f}  ({i+1}/{n_T})...")
        for j, g in enumerate(g_arr):
            params = dict(base_params); params["g"] = g
            p = BathRPAModel(**params)
            try:
                curve = imsig_curve_sc(p, T, n_q=41, n_theta=21,
                                        sc_max_iter=20)
                om = curve["omega_grid"]
                imS = curve["im_sigma"]
                mu_shift = curve["mu_shift"]

                # Doping
                N_0 = p.m_e * p.kF / (2.0 * np.pi ** 2)
                n_0 = p.kF ** 3 / (3.0 * np.pi ** 2)
                delta_n_map[i, j] = float(-N_0 * mu_shift / n_0)

                # Z (always ~1 in self-consistent treatment)
                Z_map[i, j] = 1.0  # by construction at converged mu_shift

                # Fits
                fit = fit_both_and_compare(om, imS, T)
                pl = fit["power_law"]
                mfl = fit["MFL_varma"]
                if pl["converged"]:
                    vth_map[i, j] = pl["vth"]
                    vth_r2_map[i, j] = pl["r2"]
                if mfl["converged"]:
                    mfl_r2_map[i, j] = mfl["r2"]

                form_flag = {"power_law": 0, "MFL_varma": 1, "tie": 2,
                             "neither": -1}[fit["best_form"]]
                best_form_map[i, j] = form_flag
            except Exception as e:
                pass

    elapsed = time.time() - t0
    print(f"  Map built in {elapsed:.1f}s")

    data = {
        "g_arr":         g_arr,
        "T_arr":         T_arr,
        "r_arr":         r_arr,
        "delta_n_map":   delta_n_map,
        "Z_map":         Z_map,
        "vth_map":       vth_map,
        "vth_r2_map":    vth_r2_map,
        "mfl_r2_map":    mfl_r2_map,
        "best_form_map": best_form_map,
    }

    if savepath_data is not None:
        np.savez(savepath_data, **data)
        print(f"  Saved raw data: {os.path.basename(savepath_data)}")

    return data


# =============================================================================
# Plot vth_Sigma(delta_n, T) and best-form maps
# =============================================================================

def plot_vth_map(data: dict, savepath: str):
    """
    Central diagnostic: vth_Sigma(delta_n, T) plus best-form flag.
    """
    print("\n[Plot] vth_Sigma maps")
    g_arr   = data["g_arr"]
    T_arr   = data["T_arr"]
    delta_n = data["delta_n_map"]
    vth     = data["vth_map"]
    r2_pl   = data["vth_r2_map"]
    r2_mfl  = data["mfl_r2_map"]
    best    = data["best_form_map"]
    r_arr   = data["r_arr"]

    # Use median doping per g for the x-axis
    doping_1d = np.nanmedian(delta_n, axis=0)
    sort_idx = np.argsort(doping_1d)
    D = doping_1d[sort_idx]

    vth_sorted = vth[:, sort_idx]
    r2_pl_sorted = r2_pl[:, sort_idx]
    r2_mfl_sorted = r2_mfl[:, sort_idx]
    best_sorted = best[:, sort_idx]
    r_sorted = r_arr[sort_idx]

    # Restrict to physical doping
    mask = (D >= -0.1) & (D <= 1.5)
    if mask.sum() < 4:
        mask = np.ones_like(D, dtype=bool)

    D_p = D[mask]
    vth_p = vth_sorted[:, mask]
    r2_pl_p = r2_pl_sorted[:, mask]
    r2_mfl_p = r2_mfl_sorted[:, mask]
    best_p = best_sorted[:, mask]
    r_p = r_sorted[mask]

    # Doping at r = 0
    sign_changes = np.where(np.diff(np.sign(r_p)))[0]
    doping_qcp = D_p[sign_changes[0]] if len(sign_changes) > 0 else None

    fig, axes = plt.subplots(2, 2, figsize=(PRB_DOUBLE, PRB_DOUBLE * 0.7),
                              gridspec_kw={"hspace": 0.40, "wspace": 0.30})

    D_mesh, T_mesh = np.meshgrid(D_p, T_arr)

    # Panel (a): vth_Sigma map
    ax = axes[0, 0]
    pcm = ax.pcolormesh(D_mesh, T_mesh, vth_p, shading="auto",
                         cmap="RdYlGn_r", vmin=0.5, vmax=2.5, rasterized=True)
    fig.colorbar(pcm, ax=ax, label=r"$\vartheta_\Sigma$")
    # Contour at vth = 1 (MFL) and vth = 2 (FL)
    for level, color, ls in [(1.0, "darkred", "-"), (1.5, "black", "--"),
                              (2.0, "navy", ":")]:
        try:
            ax.contour(D_mesh, T_mesh, vth_p, levels=[level],
                        colors=[color], linewidths=1.0, linestyles=[ls])
        except Exception:
            pass
    if doping_qcp is not None:
        ax.axvline(doping_qcp, color="yellow", lw=1.2, ls="-.",
                    label=rf"$r=0$: $\delta n/n_0 = {doping_qcp:.2f}$")
        ax.legend(loc="upper right", fontsize=6.5, framealpha=0.9)
    ax.set_xlabel(r"$\delta n / n_0$")
    ax.set_ylabel(r"$T / E_F$")
    ax.set_title(r"(a) $\vartheta_\Sigma(\delta n, T)$ [power-law fit]")

    # Panel (b): R^2 power-law
    ax = axes[0, 1]
    pcm = ax.pcolormesh(D_mesh, T_mesh, r2_pl_p, shading="auto",
                         cmap="viridis", vmin=0.5, vmax=1.0, rasterized=True)
    fig.colorbar(pcm, ax=ax, label=r"$R^2$ power-law")
    if doping_qcp is not None:
        ax.axvline(doping_qcp, color="yellow", lw=1.2, ls="-.")
    ax.set_xlabel(r"$\delta n / n_0$")
    ax.set_ylabel(r"$T / E_F$")
    ax.set_title(r"(b) Power-law fit quality")

    # Panel (c): R^2 MFL
    ax = axes[1, 0]
    pcm = ax.pcolormesh(D_mesh, T_mesh, r2_mfl_p, shading="auto",
                         cmap="viridis", vmin=0.5, vmax=1.0, rasterized=True)
    fig.colorbar(pcm, ax=ax, label=r"$R^2$ MFL Varma")
    if doping_qcp is not None:
        ax.axvline(doping_qcp, color="yellow", lw=1.2, ls="-.")
    ax.set_xlabel(r"$\delta n / n_0$")
    ax.set_ylabel(r"$T / E_F$")
    ax.set_title(r"(c) MFL Varma $\omega\log(b/|\omega|)$ fit quality")

    # Panel (d): best form flag
    ax = axes[1, 1]
    # Custom discrete colormap
    from matplotlib.colors import ListedColormap, BoundaryNorm
    colors_disc = ["#bdc3c7", "#c0392b", "#2c7a3e", "#f1c40f"]  # neither, power, MFL, tie
    cmap_disc = ListedColormap(colors_disc)
    norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5, 2.5], cmap_disc.N)
    pcm = ax.pcolormesh(D_mesh, T_mesh, best_p, shading="auto",
                         cmap=cmap_disc, norm=norm, rasterized=True)
    cbar = fig.colorbar(pcm, ax=ax, ticks=[-1, 0, 1, 2])
    cbar.ax.set_yticklabels(["neither", "power-law", "MFL Varma", "tie"],
                             fontsize=6.5)
    if doping_qcp is not None:
        ax.axvline(doping_qcp, color="yellow", lw=1.2, ls="-.")
    ax.set_xlabel(r"$\delta n / n_0$")
    ax.set_ylabel(r"$T / E_F$")
    ax.set_title(r"(d) Best-fit form (winner by $R^2$)")

    fig.suptitle(r"NFL diagnostic: direct extraction of $\vartheta_\Sigma$ "
                  r"from $\mathrm{Im}\,\Sigma(\omega)$ (self-consistent)",
                  y=0.99, fontsize=10)
    fig.savefig(savepath)
    plt.close(fig)
    print(f"[ok] {os.path.basename(savepath)}")


# =============================================================================
# Driver
# =============================================================================

def main():
    print("=" * 70)
    print("NFL diagnostic (self-consistent): direct vth_Sigma extraction")
    print("=" * 70)
    t_start = time.time()
    os.makedirs(OUTDIR, exist_ok=True)

    # Reference model: gapped (the only one with accessible QCP)
    base_params = dict(
        dispersion="gapped", Omega0=0.5, c_disp=0.5,
        damping="ohmic", gamma0=0.1, kF=1.0, m_e=1.0,
        use_RPA=True,
    )

    # 1. Sample curves with fits
    g_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    curves_results = plot_imsig_curves_with_fits(
        base_params, T_fix=0.05, g_values=g_values,
        savepath=os.path.join(OUTDIR, "fig_nfl_curves.pdf"))

    # Print summary of fits
    print("\nSummary of fits at T = 0.05:")
    print(f"  {'g':>5s} {'r':>8s} {'mu_shift':>12s} "
          f"{'vth':>8s} {'R²pl':>7s} {'R²mfl':>7s} {'best':>10s}")
    for res in curves_results:
        pl = res["fit"]["power_law"]
        mfl = res["fit"]["MFL_varma"]
        print(f"  {res['g']:>5.2f} {res['r']:>+8.4f} "
              f"{res['curve']['mu_shift']:>+12.4e} "
              f"{pl['vth']:>8.3f} {pl['r2']:>7.3f} {mfl['r2']:>7.3f} "
              f"{res['fit']['best_form']:>10s}")

    # 2. Build full map
    data = build_vth_map(base_params,
                          savepath_data=os.path.join(OUTDIR,
                                                       "nfl_diagnostic_sc_data.npz"))
    plot_vth_map(data, savepath=os.path.join(OUTDIR, "fig_nfl_vth_map.pdf"))

    t_total = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"Done. Total time: {t_total:.1f}s")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
