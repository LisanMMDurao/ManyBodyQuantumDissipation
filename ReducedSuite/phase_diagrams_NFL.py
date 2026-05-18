"""
phase_diagrams_NFL.py
---------------------
Generates phase diagrams for the non-Fermi liquid (NFL) emergence
in the many-body Caldeira-Leggett model with parametrized damping.

Convention: the self-energy exponent on the Fermi surface is

    vartheta_Sigma(s, alpha, beta) = s + 1 + s*(2 - alpha - 4*beta)/(2*beta + alpha)

which simplifies, for the NFL condition vartheta < 2, to

    s*(3 - alpha) < beta + alpha          (NFL condition)

Phase classification:
    Fermi liquid (FL):              vartheta >= 2
    Marginal Fermi liquid (MFL):    vartheta = 1 (Varma)
    Non-Fermi liquid (NFL):         0 < vartheta < 2
    Singular regime:                vartheta <= 0  (breakdown)

APS style: PRB single-column 3.375", serif, inward ticks, 300 dpi.
Output files:
    fig1_frequency_damping.pdf  - (s, beta) plane at alpha = 0
    fig2_momentum_damping.pdf   - (alpha, beta) plane at s = 1
    fig3_mixed_damping.pdf      - (s, alpha) plane for beta = 1, 2
    fig4_3D_atlas.pdf           - Consolidated 3D visualization
    fig5_marginal_lines.pdf     - Varma line vartheta = 1 in (s, alpha)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import os

# =============================================================================
# APS / PRB style setup
# =============================================================================

rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Computer Modern Roman", "Times New Roman", "DejaVu Serif"],
    "font.size":          9,
    "axes.labelsize":     9,
    "axes.titlesize":     9,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "legend.fontsize":    8,
    "xtick.direction":    "in",
    "ytick.direction":    "in",
    "xtick.top":          True,
    "ytick.right":        True,
    "xtick.major.size":   3.5,
    "ytick.major.size":   3.5,
    "xtick.minor.size":   2.0,
    "ytick.minor.size":   2.0,
    "xtick.major.width":  0.6,
    "ytick.major.width":  0.6,
    "axes.linewidth":     0.6,
    "lines.linewidth":    1.2,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.02,
    "text.usetex":        False,
    "mathtext.fontset":   "cm",
})

PRB_SINGLE = 3.375    # in
PRB_DOUBLE = 7.0      # in

# Output folder
OUTDIR = "."
os.makedirs(OUTDIR, exist_ok=True)


# =============================================================================
# Core function: self-energy exponent
# =============================================================================

def vartheta_sigma(s, alpha, beta):
    """
    Self-energy exponent on Fermi surface, corrected expression:

        vartheta = s + 1 + s*(2 - alpha - 4*beta) / (2*beta + alpha)

    Equivalent simplified form:
        vartheta = 1 + s*(3 - alpha) / (2*beta + alpha)   -- ORIGINAL (incorrect)
        vartheta = s + 1 + s*(2 - alpha - 4*beta)/(2*beta + alpha)  -- CORRECTED

    NFL condition: vartheta < 2  <=>  s*(3 - alpha) < beta + alpha
    """
    denom = 2.0 * beta + alpha
    # Avoid division by zero (gapped case beta=0, alpha=0 must be excluded)
    denom = np.where(np.abs(denom) < 1e-12, np.nan, denom)
    return s + 1.0 + s * (2.0 - alpha - 4.0 * beta) / denom


def classify_phase(vth, mfl_tol=0.02):
    """
    Classification with explicit tolerance for the marginal Varma window.

    Phase codes:
        0 -> FL          (vth >= 2)
        1 -> NFL         (1 + mfl_tol < vth < 2)
        2 -> MFL/Varma   (|vth - 1| <= mfl_tol)
        3 -> Sub-marginal (0 < vth < 1 - mfl_tol)
        4 -> Singular   (vth <= 0)
        nan -> Undefined (denom = 0)

    The tolerance is intentionally narrow so that exact equality vth = 1
    (which occurs throughout the beta = 1 slice for ANY (s, alpha) -- a
    nontrivial physical degeneracy of the corrected exponent expression)
    appears as a thin coherent MFL region rather than as numerical stripes.
    """
    out = np.full_like(vth, np.nan, dtype=float)
    out = np.where(vth >= 2.0,                              0, out)
    out = np.where((vth < 2.0) & (vth > 1.0 + mfl_tol),     1, out)
    out = np.where(np.abs(vth - 1.0) <= mfl_tol,            2, out)
    out = np.where((vth > 0.0) & (vth < 1.0 - mfl_tol),     3, out)
    out = np.where(vth <= 0.0,                              4, out)
    return out


# Discrete colormap and labels for phases
PHASE_COLORS = [
    "#dfe7f5",  # 0 FL          - light blue/grey
    "#f6c6a1",  # 1 NFL         - orange
    "#c83737",  # 2 MFL Varma   - deep red
    "#9c1aff",  # 3 Sub-marginal - purple
    "#1f1f1f",  # 4 Singular    - near black
]
PHASE_LABELS = ["FL", "NFL", "MFL", "Sub-marg.", "Singular"]
phase_cmap = ListedColormap(PHASE_COLORS)
phase_norm = BoundaryNorm(np.arange(-0.5, 5.5, 1.0), phase_cmap.N)


def phase_legend_handles():
    """Manual legend handles in the order of PHASE_LABELS."""
    return [mpatches.Patch(facecolor=c, edgecolor="0.2", linewidth=0.5, label=lab)
            for c, lab in zip(PHASE_COLORS, PHASE_LABELS)]


# =============================================================================
# Figure 1: Frequency damping (alpha = 0)
#   Axes: s (x) vs beta (y), alpha fixed at 0
# =============================================================================

def fig_frequency_damping():
    s_vals    = np.linspace(0.01, 2.0, 400)
    beta_vals = np.linspace(0.5,  3.0, 400)
    S, B = np.meshgrid(s_vals, beta_vals)
    ALPHA = 0.0

    vth = vartheta_sigma(S, ALPHA, B)
    cls = classify_phase(vth)

    fig, axes = plt.subplots(1, 2, figsize=(PRB_DOUBLE, PRB_SINGLE),
                             gridspec_kw={"wspace": 0.30})

    # --- (a) Phase classification
    ax = axes[0]
    im = ax.pcolormesh(S, B, cls, cmap=phase_cmap, norm=phase_norm,
                       shading="auto", rasterized=True)
    # NFL boundary at alpha = 0: vth = 2 <=> 3*s = beta + 0 - (s-2*beta)... use direct
    # The actual vth = 2 contour:
    ax.contour(S, B, vth, levels=[2.0], colors="k", linewidths=1.0,
               linestyles="-")
    # Marginal contour vth = 1: from analysis this is the line beta = 1
    ax.axhline(1.0, color="darkred", lw=1.2, ls="-",
               label=r"MFL line $\beta=1$ ($\vartheta\equiv 1$)")
    # Mark Ohmic canonical points
    ax.scatter([1.0, 1.0], [1.0, 2.0], s=30, c="white", edgecolors="k",
               zorder=5, marker="o")
    ax.annotate("Ohmic\n+ linear\n(MFL)", xy=(1.0, 1.0), xytext=(1.3, 1.4),
                fontsize=6.5, ha="left",
                arrowprops=dict(arrowstyle="-", lw=0.5, color="k"))
    ax.annotate("Ohmic\n+ quadratic\n(sub-marg.)", xy=(1.0, 2.0), xytext=(1.3, 2.5),
                fontsize=6.5, ha="left",
                arrowprops=dict(arrowstyle="-", lw=0.5, color="k"))
    ax.set_xlim(s_vals.min(), s_vals.max())
    ax.set_ylim(beta_vals.min(), beta_vals.max())
    ax.set_xlabel(r"frequency exponent $s$")
    ax.set_ylabel(r"dispersion exponent $\beta$")
    ax.set_title(r"(a) Phase diagram, $\alpha=0$")
    ax.legend(loc="upper left", framealpha=0.95, edgecolor="0.3", fontsize=7)

    # --- (b) Exponent value
    ax = axes[1]
    levels = np.linspace(0.0, 4.0, 21)
    cf = ax.contourf(S, B, np.clip(vth, 0, 4), levels=levels,
                     cmap="viridis", extend="max")
    cont1 = ax.contour(S, B, vth, levels=[1.0], colors="white",
                       linewidths=1.3, linestyles="-")
    cont2 = ax.contour(S, B, vth, levels=[2.0], colors="white",
                       linewidths=1.0, linestyles="--")
    ax.clabel(cont1, fmt={1.0: r"$\vartheta=1$"}, fontsize=7, inline=True)
    ax.clabel(cont2, fmt={2.0: r"$\vartheta=2$"}, fontsize=7, inline=True)
    ax.set_xlabel(r"frequency exponent $s$")
    ax.set_ylabel(r"dispersion exponent $\beta$")
    ax.set_title(r"(b) $\vartheta_\Sigma(s,\beta)$, $\alpha=0$")
    cbar = fig.colorbar(cf, ax=ax, pad=0.02, fraction=0.05)
    cbar.set_label(r"$\vartheta_\Sigma$")

    # Common phase legend
    handles = phase_legend_handles()
    fig.legend(handles=handles, loc="lower center",
               ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.04))

    fig.savefig(os.path.join(OUTDIR, "fig1_frequency_damping.pdf"))
    plt.close(fig)
    print("[ok] fig1_frequency_damping.pdf")


# =============================================================================
# Figure 2: Momentum damping (s = 1)
#   Axes: alpha (x) vs beta (y), s fixed at 1
# =============================================================================

def fig_momentum_damping():
    alpha_vals = np.linspace(0.0, 3.0, 400)
    beta_vals  = np.linspace(0.5, 3.0, 400)
    A, B = np.meshgrid(alpha_vals, beta_vals)
    S_FIXED = 1.0

    vth = vartheta_sigma(S_FIXED, A, B)
    cls = classify_phase(vth)

    fig, axes = plt.subplots(1, 2, figsize=(PRB_DOUBLE, PRB_SINGLE),
                             gridspec_kw={"wspace": 0.30})

    # --- (a) Phase classification
    ax = axes[0]
    im = ax.pcolormesh(A, B, cls, cmap=phase_cmap, norm=phase_norm,
                       shading="auto", rasterized=True)
    # NFL boundary at s = 1: vth = 2 contour
    ax.contour(A, B, vth, levels=[2.0], colors="k", linewidths=1.0,
               linestyles="-")
    # MFL exact line at beta = 1
    ax.axhline(1.0, color="darkred", lw=1.2, ls="-",
               label=r"MFL line $\beta=1$ ($\vartheta\equiv 1$)")
    # Mark canonical points
    ax.scatter([0.0, 0.5, 1.0], [2.0, 2.0, 2.0], s=30, c="white",
               edgecolors="k", zorder=5, marker="o")
    ax.annotate(r"$\alpha=0.5$ threshold" "\n" r"(sub-marg.)",
                xy=(0.5, 2.0), xytext=(0.7, 2.5),
                fontsize=6.5, ha="left",
                arrowprops=dict(arrowstyle="-", lw=0.5, color="k"))
    ax.set_xlim(alpha_vals.min(), alpha_vals.max())
    ax.set_ylim(beta_vals.min(), beta_vals.max())
    ax.set_xlabel(r"momentum exponent $\alpha$")
    ax.set_ylabel(r"dispersion exponent $\beta$")
    ax.set_title(r"(a) Phase diagram, $s=1$")
    ax.legend(loc="upper right", framealpha=0.95, edgecolor="0.3", fontsize=7)

    # --- (b) Exponent value
    ax = axes[1]
    levels = np.linspace(0.0, 4.0, 21)
    cf = ax.contourf(A, B, np.clip(vth, 0, 4), levels=levels,
                     cmap="viridis", extend="max")
    cont1 = ax.contour(A, B, vth, levels=[1.0], colors="white",
                       linewidths=1.3, linestyles="-")
    cont2 = ax.contour(A, B, vth, levels=[2.0], colors="white",
                       linewidths=1.0, linestyles="--")
    ax.clabel(cont1, fmt={1.0: r"$\vartheta=1$"}, fontsize=7, inline=True)
    ax.clabel(cont2, fmt={2.0: r"$\vartheta=2$"}, fontsize=7, inline=True)
    ax.set_xlabel(r"momentum exponent $\alpha$")
    ax.set_ylabel(r"dispersion exponent $\beta$")
    ax.set_title(r"(b) $\vartheta_\Sigma(\alpha,\beta)$, $s=1$")
    cbar = fig.colorbar(cf, ax=ax, pad=0.02, fraction=0.05)
    cbar.set_label(r"$\vartheta_\Sigma$")

    handles = phase_legend_handles()
    fig.legend(handles=handles, loc="lower center",
               ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.04))

    fig.savefig(os.path.join(OUTDIR, "fig2_momentum_damping.pdf"))
    plt.close(fig)
    print("[ok] fig2_momentum_damping.pdf")


# =============================================================================
# Figure 3: Mixed damping
#   Two panels: beta = 1 (linear dispersion) and beta = 2 (quadratic)
#   Axes: s (x) vs alpha (y)
# =============================================================================

def fig_mixed_damping():
    """
    Mixed damping (s, alpha) phase plane for three representative beta values.

    Note on beta = 1: The corrected exponent simplifies to vartheta == 1
    identically for ANY (s, alpha) when beta = 1. This is a nontrivial
    physical degeneracy -- linear bath dispersion generates a Marginal Fermi
    Liquid (Varma) generically, without fine-tuning. We display it as a
    dedicated panel with annotation rather than as numerical stripes.
    """
    s_vals     = np.linspace(0.01, 2.0, 400)
    alpha_vals = np.linspace(0.0,  3.0, 400)
    S, A = np.meshgrid(s_vals, alpha_vals)

    fig, axes = plt.subplots(1, 3, figsize=(PRB_DOUBLE, PRB_SINGLE * 0.95),
                             gridspec_kw={"wspace": 0.32, "bottom": 0.22,
                                          "top": 0.88, "left": 0.07,
                                          "right": 0.98})

    for ax, beta, label in zip(axes, [0.5, 1.0, 2.0],
                               [r"$\beta=0.5$ (sub-linear)",
                                r"$\beta=1$ (linear, MFL generic)",
                                r"$\beta=2$ (quadratic)"]):
        if abs(beta - 1.0) < 1e-9:
            # Degenerate case: entire plane is MFL.
            cls = np.full_like(S, 2.0)  # phase code 2 = MFL
            ax.pcolormesh(S, A, cls, cmap=phase_cmap, norm=phase_norm,
                          shading="auto", rasterized=True)
            ax.text(0.50, 0.55,
                    r"$\vartheta_\Sigma \equiv 1$"
                    "\n"
                    r"$\forall\,(s,\alpha)$",
                    transform=ax.transAxes,
                    ha="center", va="center",
                    fontsize=10, color="white",
                    bbox=dict(boxstyle="round,pad=0.4",
                              facecolor="black", alpha=0.55,
                              edgecolor="none"))
        else:
            vth = vartheta_sigma(S, A, beta)
            cls = classify_phase(vth)
            ax.pcolormesh(S, A, cls, cmap=phase_cmap, norm=phase_norm,
                          shading="auto", rasterized=True)
            s_curve = np.linspace(s_vals.min(), s_vals.max(), 400)
            alpha_curve = (3 * s_curve - beta) / (s_curve + 1)
            valid = (alpha_curve >= alpha_vals.min()) & \
                    (alpha_curve <= alpha_vals.max())
            ax.plot(s_curve[valid], alpha_curve[valid], "k-", lw=1.0,
                    label=r"NFL bdry $\vartheta=2$")
            ax.legend(loc="upper right", framealpha=0.95, edgecolor="0.3",
                      fontsize=7)

        ax.set_xlim(s_vals.min(), s_vals.max())
        ax.set_ylim(alpha_vals.min(), alpha_vals.max())
        ax.set_xlabel(r"frequency exponent $s$")
        if ax is axes[0]:
            ax.set_ylabel(r"momentum exponent $\alpha$")
        ax.set_title(label, fontsize=8.5)

    handles = phase_legend_handles()
    fig.legend(handles=handles, loc="lower center",
               ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.02))

    fig.suptitle(r"Mixed damping phase diagrams in $(s,\alpha)$",
                 y=0.97, fontsize=10)

    fig.savefig(os.path.join(OUTDIR, "fig3_mixed_damping.pdf"))
    plt.close(fig)
    print("[ok] fig3_mixed_damping.pdf")


# =============================================================================
# Figure 4: 3D atlas - NFL volume in (s, alpha, beta) parameter space
# =============================================================================

def fig_3D_atlas():
    """
    3D atlas: NFL region in (s, alpha, beta) parameter space.

    Strategy: for each beta slice, paint the (s, alpha) plane at height z=beta
    in 4 colors corresponding to FL / NFL / MFL / sub-marginal. Use
    facecolors with the phase cmap and let plot_surface render the patches.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from matplotlib.colors import to_rgba

    beta_slices = [0.5, 0.75, 1.0, 1.5, 2.0, 2.5]
    s_vals     = np.linspace(0.01, 2.0, 60)
    alpha_vals = np.linspace(0.0,  3.0, 60)
    S, A = np.meshgrid(s_vals, alpha_vals)

    fig = plt.figure(figsize=(PRB_SINGLE * 1.6, PRB_SINGLE * 1.4))
    ax = fig.add_subplot(111, projection="3d")

    for beta in beta_slices:
        vth = vartheta_sigma(S, A, beta)
        cls = classify_phase(vth)
        # Build facecolor array: shape (nrows, ncols, 4) RGBA
        facecols = np.empty(S.shape + (4,))
        for code in range(5):
            mask = (cls == code)
            facecols[mask] = to_rgba(PHASE_COLORS[code], alpha=0.85)
        nan_mask = np.isnan(cls)
        facecols[nan_mask] = (1.0, 1.0, 1.0, 0.0)  # transparent

        Z = np.full_like(S, beta)
        ax.plot_surface(S, A, Z, facecolors=facecols,
                        rstride=1, cstride=1,
                        edgecolor="none", shade=False, antialiased=False)

        # Add a thin frame at each beta for clarity
        ax.plot([s_vals.min(), s_vals.max(), s_vals.max(), s_vals.min(), s_vals.min()],
                [alpha_vals.min(), alpha_vals.min(), alpha_vals.max(), alpha_vals.max(), alpha_vals.min()],
                [beta]*5, color="0.4", lw=0.4)

    ax.set_xlabel(r"$s$", labelpad=2)
    ax.set_ylabel(r"$\alpha$", labelpad=2)
    ax.set_zlabel(r"$\beta$", labelpad=2)
    ax.set_title(r"Phase atlas across $(s,\alpha,\beta)$", fontsize=10)
    ax.view_init(elev=22, azim=-58)

    # Legend
    handles = phase_legend_handles()
    ax.legend(handles=handles, loc="upper left",
              bbox_to_anchor=(-0.05, 0.95),
              fontsize=7, framealpha=0.95, edgecolor="0.3")

    fig.savefig(os.path.join(OUTDIR, "fig4_3D_atlas.pdf"))
    plt.close(fig)
    print("[ok] fig4_3D_atlas.pdf")


# =============================================================================
# Figure 5: Marginal Varma lines vartheta = 1 in (s, alpha) for several beta
# =============================================================================

def fig_marginal_lines():
    """
    Range of self-energy exponent vartheta as function of beta.

    For each beta, we scan (s, alpha) in the physical range and plot:
        - min, max, and mean of vartheta
        - the canonical phases (FL: vth >= 2, NFL: 1<vth<2, MFL: vth=1,
          sub-marginal: 0<vth<1, singular: vth<=0)

    This makes explicit the central analytical result:
        - beta < 1: large vth (FL/NFL regime)
        - beta = 1: vth identically 1 (MFL Varma generic)
        - beta > 1: small vth (sub-marginal / singular)
    """
    s_vals     = np.linspace(0.01, 2.0, 200)
    alpha_vals = np.linspace(0.0,  3.0, 200)
    S, A = np.meshgrid(s_vals, alpha_vals)

    beta_grid = np.linspace(0.1, 3.5, 300)
    vth_min  = np.zeros_like(beta_grid)
    vth_max  = np.zeros_like(beta_grid)
    vth_mean = np.zeros_like(beta_grid)

    for i, beta in enumerate(beta_grid):
        vth = vartheta_sigma(S, A, beta)
        vth_min[i]  = np.nanmin(vth)
        vth_max[i]  = np.nanmax(vth)
        vth_mean[i] = np.nanmean(vth)

    fig, ax = plt.subplots(figsize=(PRB_SINGLE * 1.1, PRB_SINGLE * 1.0))

    # Shaded range of vth as (s, alpha) vary
    ax.fill_between(beta_grid, vth_min, vth_max,
                    color="#7aa0d6", alpha=0.35,
                    label=r"range of $\vartheta_\Sigma$ over $(s,\alpha)$")
    ax.plot(beta_grid, vth_mean, color="#1a3a73", lw=1.2,
            label=r"mean $\vartheta_\Sigma$")

    # Phase boundary lines
    ax.axhline(2.0, color="0.2",       lw=0.8, ls="-",  label=r"FL boundary $\vartheta=2$")
    ax.axhline(1.0, color="darkred",   lw=0.8, ls="--", label=r"MFL Varma $\vartheta=1$")
    ax.axhline(0.0, color="purple",    lw=0.8, ls=":",  label=r"breakdown $\vartheta=0$")

    # Mark special beta = 1
    ax.axvline(1.0, color="darkred", lw=0.6, alpha=0.4)
    ax.scatter([1.0], [1.0], s=40, c="darkred", zorder=5,
               edgecolor="white", linewidth=0.6)
    ax.annotate(r"$\beta=1$: MFL generic" "\n" r"($\vartheta\equiv 1$ in plane)",
                xy=(1.0, 1.0), xytext=(1.5, 1.5),
                fontsize=7, ha="left",
                arrowprops=dict(arrowstyle="->", lw=0.5, color="darkred"))

    # Phase regions on the right side
    ax.text(3.45, 2.6,  "FL",        ha="right", va="center", fontsize=7,
            color="0.2", fontweight="bold")
    ax.text(3.45, 1.5,  "NFL",       ha="right", va="center", fontsize=7,
            color="0.2", fontweight="bold")
    ax.text(3.45, 0.5,  "sub-marg.", ha="right", va="center", fontsize=7,
            color="0.2", fontweight="bold")
    ax.text(3.45, -0.3, "singular",  ha="right", va="center", fontsize=7,
            color="0.2", fontweight="bold")

    ax.set_xlim(beta_grid.min(), beta_grid.max())
    ax.set_ylim(-0.7, 3.5)
    ax.set_xlabel(r"dispersion exponent $\beta$")
    ax.set_ylabel(r"self-energy exponent $\vartheta_\Sigma$")
    ax.set_title(r"Range of $\vartheta_\Sigma$ across damping families")
    ax.legend(loc="upper right", fontsize=6.5, framealpha=0.95,
              edgecolor="0.3", ncol=1)

    fig.savefig(os.path.join(OUTDIR, "fig5_marginal_lines.pdf"))
    plt.close(fig)
    print("[ok] fig5_marginal_lines.pdf")


# =============================================================================
# Sanity checks (printed to stdout)
# =============================================================================

def sanity_checks():
    print("\n=== Sanity checks ===")
    cases = [
        # (s, alpha, beta, description)
        (1.0, 0.0, 2.0, "Ohmic + quadratic dispersion"),
        (1.0, 0.0, 1.0, "Ohmic + linear dispersion"),
        (0.5, 0.0, 2.0, "Sub-ohmic + quadratic"),
        (1.0, 1.0, 2.0, "Ohmic + momentum damping + quadratic"),
        (1.0, 0.5, 2.0, "Ohmic + alpha=0.5 + quadratic"),
        (0.5, 1.0, 1.0, "Sub-ohmic mixed"),
    ]
    print(f"{'case':<48s} {'vartheta':>10s} {'phase':>10s}")
    print("-" * 72)
    for s, alpha, beta, desc in cases:
        vth = vartheta_sigma(s, alpha, beta)
        cls = classify_phase(np.array([vth]))[0]
        phase = PHASE_LABELS[int(cls)] if not np.isnan(cls) else "n/a"
        print(f"{desc:<48s} {vth:>10.4f} {phase:>10s}")
    print()


# =============================================================================
# Driver
# =============================================================================

if __name__ == "__main__":
    print("Generating NFL phase diagrams in APS / PRB style ...")
    sanity_checks()
    fig_frequency_damping()
    fig_momentum_damping()
    fig_mixed_damping()
    fig_3D_atlas()
    fig_marginal_lines()
    print("\nAll figures written to", os.path.abspath(OUTDIR))
