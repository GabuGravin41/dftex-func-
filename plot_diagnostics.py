"""
Publication-Quality Visualization Suite for One-Electron DFT Diagnostics
Designed to JCTC / JCP / PRB Author Standards.
Produces 600-DPI publication figures in vector PDF and high-res PNG formats.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# Set publication style parameters
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Computer Modern Roman", "Times New Roman"],
    "mathtext.fontset": "cm",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "lines.linewidth": 1.75,
    "lines.markersize": 6,
    "axes.linewidth": 1.0,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "figure.dpi": 600,
    "savefig.dpi": 600,
    "savefig.bbox": "tight"
})

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Colorblind-safe palette (Okabe-Ito inspired)
COLORS = {
    "UHF": "#000000",          # Black reference
    "PBE": "#0072B2",          # Blue
    "r2SCAN": "#009E73",       # Green
    "SCAN": "#D55E00",         # Vermilion
    "B3LYP": "#CC79A7",        # Reddish purple
    "CIDER26XCCHEM": "#E69F00",# Orange
    "CIDER23X": "#56B4E9"      # Sky blue
}


def plot_basis_incompleteness(csv_path="master_dft_one_electron_dataset.csv"):
    """
    Figure 1: Basis-Incompleteness of the Same-Basis Reference across Isoelectronic Series.
    Demonstrates the non-monotonic behavior across Z in def2-TZVP vs other bases.
    """
    if not os.path.exists(csv_path):
        print(f"File {csv_path} not found. Skipping plot.")
        return

    df = pd.read_csv(csv_path)
    uhf_df = df[df["functional"] == "UHF"].copy()
    if uhf_df.empty:
        return

    # Map ions to Z
    z_map = {"H": 1, "He+": 2, "Li2+": 3, "Be3+": 4}
    uhf_df["Z"] = uhf_df["system"].map(z_map)
    uhf_df = uhf_df.dropna(subset=["Z"])
    uhf_df["E_exact"] = -0.5 * (uhf_df["Z"] ** 2)
    uhf_df["Incompleteness_mHa"] = (uhf_df["E"] - uhf_df["E_exact"]) * 1000.0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Panel A: Incompleteness vs Z for each basis
    bases = uhf_df["basis"].unique()
    markers = ["o", "s", "^", "D"]
    for idx, b in enumerate(bases):
        sub = uhf_df[uhf_df["basis"] == b].sort_values("Z")
        ax1.plot(sub["Z"], sub["Incompleteness_mHa"], marker=markers[idx % len(markers)],
                 label=f"{b.upper()}", linewidth=1.8)

    ax1.set_xlabel(r"Nuclear Charge $Z$")
    ax1.set_ylabel(r"Basis Incompleteness $\Delta E = E_{\mathrm{UHF}} - (-Z^2/2)$ [mHa]")
    ax1.set_title(r"\textbf{(a)} Finite-Basis Incompleteness vs. Exact Hydrogenic Limit")
    ax1.set_xticks([1, 2, 3, 4])
    ax1.set_xticklabels([r"$\mathrm{H}\,(Z=1)$", r"$\mathrm{He}^+\,(Z=2)$", 
                         r"$\mathrm{Li}^{2+}\,(Z=3)$", r"$\mathrm{Be}^{3+}\,(Z=4)$"])
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(frameon=True, edgecolor="none", facecolor="#f8f8f8")

    # Panel B: Naive vs Same-Basis SIE for PBE / CIDER / B3LYP
    df_tzvp = df[(df["basis"] == "def2-tzvp") & (df["R"].isna())].copy()
    df_tzvp["Z"] = df_tzvp["system"].map(z_map)
    df_tzvp = df_tzvp.dropna(subset=["Z"])

    # Reference UHF energy per Z
    uhf_ref = df_tzvp[df_tzvp["functional"] == "UHF"].set_index("Z")["E"].to_dict()
    df_tzvp["E_UHF"] = df_tzvp["Z"].map(uhf_ref)
    df_tzvp["E_exact"] = -0.5 * (df_tzvp["Z"] ** 2)
    df_tzvp["SIE_naive"] = (df_tzvp["E"] - df_tzvp["E_exact"]) * 1000.0
    df_tzvp["SIE_same"] = (df_tzvp["E"] - df_tzvp["E_UHF"]) * 1000.0

    target_funcs = [f for f in ["PBE", "r2SCAN", "B3LYP", "CIDER26XCCHEM"] if f in df_tzvp["functional"].values]
    for f in target_funcs:
        sub = df_tzvp[df_tzvp["functional"] == f].sort_values("Z")
        color = COLORS.get(f, "#333333")
        ax2.plot(sub["Z"], sub["SIE_same"], marker="o", color=color, label=f"{f} (Same-basis)", linewidth=1.8)
        ax2.plot(sub["Z"], sub["SIE_naive"], marker="x", linestyle="--", color=color, alpha=0.6, label=f"{f} (Naive)")

    ax2.set_xlabel(r"Nuclear Charge $Z$")
    ax2.set_ylabel(r"Self-Interaction Error [mHa]")
    ax2.set_title(r"\textbf{(b)} Same-Basis Reference vs. Naive Reference Inflation")
    ax2.set_xticks([1, 2, 3, 4])
    ax2.set_xticklabels([r"$\mathrm{H}$", r"$\mathrm{He}^+$", r"$\mathrm{Li}^{2+}$", r"$\mathrm{Be}^{3+}$"])
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(frameon=True, edgecolor="none", facecolor="#f8f8f8", fontsize=8.5, ncol=2)

    plt.tight_layout()
    pdf_out = os.path.join(OUTPUT_DIR, "fig1_basis_incompleteness.pdf")
    png_out = os.path.join(OUTPUT_DIR, "fig1_basis_incompleteness.png")
    fig.savefig(pdf_out)
    fig.savefig(png_out)
    plt.close(fig)
    print(f"Saved: {pdf_out} and {png_out}")


def plot_h2plus_dissociation(csv_path="master_dft_one_electron_dataset.csv"):
    """
    Figure 2: Stretched H2+ Dissociation Curve, Delocalization Plunge, and Symmetry-Breaking Bifurcation.
    """
    if not os.path.exists(csv_path):
        print(f"File {csv_path} not found. Skipping plot.")
        return

    df = pd.read_csv(csv_path)
    h2_df = df[(df["system"] == "H2+") & (df["R"].notna())].copy()
    if h2_df.empty:
        return

    h2_df["R"] = h2_df["R"].astype(float)
    uhf_ref = h2_df[h2_df["functional"] == "UHF"].set_index("R")["E"].to_dict()
    h2_df["E_UHF"] = h2_df["R"].map(uhf_ref)
    h2_df["Delta_mHa"] = (h2_df["E"] - h2_df["E_UHF"]) * 1000.0

    fig = plt.figure(figsize=(12, 9))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)

    ax1 = fig.add_subplot(gs[0, :])   # Top: Total Energy E(R)
    ax2 = fig.add_subplot(gs[1, 0])   # Bottom Left: Signed Delta(R)
    ax3 = fig.add_subplot(gs[1, 1])   # Bottom Right: Charge Asymmetry |qa - qb|

    # --- Top Panel: E(R) ---
    uhf_curve = h2_df[h2_df["functional"] == "UHF"].sort_values("R")
    ax1.plot(uhf_curve["R"], uhf_curve["E"], color=COLORS["UHF"], linewidth=2.2, label=r"Exact UHF (Full-CI in basis)")

    funcs = [f for f in ["PBE", "r2SCAN", "B3LYP", "CIDER26XCCHEM"] if f in h2_df["functional"].unique()]
    for f in funcs:
        sub = h2_df[h2_df["functional"] == f].sort_values("R")
        color = COLORS.get(f, "#444444")
        # Split into stable vs unstable
        stable_pts = sub[sub["stable"] == True]
        unstable_pts = sub[sub["stable"] == False]
        ax1.plot(sub["R"], sub["E"], color=color, marker="o", label=f"{f}", linewidth=1.6)

    ax1.set_xlabel(r"Internuclear Distance $R$ [\AA]")
    ax1.set_ylabel(r"Total Electronic Energy $E(R)$ [Ha]")
    ax1.set_title(r"\textbf{(a)} $\mathrm{H}_2^+$ Potential Energy Dissociation Curve", fontsize=12)
    ax1.set_xlim(0.4, 15.2)
    ax1.set_ylim(-0.65, -0.38)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(frameon=True, edgecolor="none", facecolor="#f8f8f8", ncol=3)

    # --- Bottom Left: Signed Delta(R) ---
    ax2.axhline(0, color="black", linestyle="-", linewidth=0.8, alpha=0.7)
    for f in funcs:
        sub = h2_df[h2_df["functional"] == f].sort_values("R")
        color = COLORS.get(f, "#444444")
        ax2.plot(sub["R"], sub["Delta_mHa"], color=color, marker="o", linewidth=1.6, label=f)

    ax2.set_xlabel(r"Internuclear Distance $R$ [\AA]")
    ax2.set_ylabel(r"Signed Functional Error $\Delta = E_{\mathrm{DFT}} - E_{\mathrm{UHF}}$ [mHa]")
    ax2.set_title(r"\textbf{(b)} Delocalization Catastrophe \& Post-Bifurcation Branch", fontsize=11)
    ax2.set_xlim(0.4, 15.2)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(frameon=True, edgecolor="none", facecolor="#f8f8f8")

    # Annotate Delocalization plunge
    ax2.text(3.5, -45, r"$\leftarrow$ Delocalization Over-binding ($\Delta < 0$)", fontsize=9, color="#666666")
    ax2.text(8.0, 5, r"$\leftarrow$ Localized Branch ($\Delta \approx \mathrm{SIE}_{\mathrm{atom}}$)", fontsize=9, color="#666666")

    # --- Bottom Right: Mulliken Charge Asymmetry |qa - qb| ---
    for f in funcs:
        sub = h2_df[h2_df["functional"] == f].sort_values("R")
        color = COLORS.get(f, "#444444")
        asym = (sub["q_Ha"] - sub["q_Hb"]).abs()
        ax3.plot(sub["R"], asym, color=color, marker="s", linewidth=1.6, label=f)

    ax3.set_xlabel(r"Internuclear Distance $R$ [\AA]")
    ax3.set_ylabel(r"Proton Charge Asymmetry $|q(\mathrm{H}_a) - q(\mathrm{H}_b)|$")
    ax3.set_title(r"\textbf{(c)} Order Parameter for Spatial Symmetry Breaking", fontsize=11)
    ax3.set_xlim(0.4, 15.2)
    ax3.set_ylim(-0.05, 1.05)
    ax3.axhline(0.0, color="gray", linestyle=":", label="Symmetric (0.5 / 0.5)")
    ax3.axhline(1.0, color="gray", linestyle="--", label="Localized (1.0 / 0.0)")
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.legend(frameon=True, edgecolor="none", facecolor="#f8f8f8", fontsize=8.5)

    plt.tight_layout()
    pdf_out = os.path.join(OUTPUT_DIR, "fig2_h2plus_bifurcation.pdf")
    png_out = os.path.join(OUTPUT_DIR, "fig2_h2plus_bifurcation.png")
    fig.savefig(pdf_out)
    fig.savefig(png_out)
    plt.close(fig)
    print(f"Saved: {pdf_out} and {png_out}")


if __name__ == "__main__":
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "master_dft_one_electron_dataset.csv"
    print("Generating publication figures...")
    plot_basis_incompleteness(csv_file)
    plot_h2plus_dissociation(csv_file)
    print("Figure generation complete.")
