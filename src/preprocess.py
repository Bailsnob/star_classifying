"""
Phase 1 - Preprocessing and exploratory data analysis.

Loads the cached Gaia DR3 sample, derives the features used downstream
(absolute magnitude, color), performs light cleaning, prints a class-balance
summary, and saves two EDA plots: the spectral-type distribution and the
HR diagram.

Usage:
    python src/preprocess.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FIG_DIR = Path(__file__).resolve().parent.parent / "figures"

SPECTRAL_ORDER = ["O", "B", "A", "F", "G", "K", "M", "CSTAR"]


def load_raw(path: Path | None = None) -> pd.DataFrame:
    path = path or DATA_DIR / "gaia_dr3_sample.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python src/fetch_data.py` first."
        )
    return pd.read_parquet(path)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive photometric features used by the supervised model."""
    df = df.copy()

    # Absolute G magnitude from parallax (mas):
    #   M_G = m_G + 5 * log10(parallax) - 10
    df["abs_g_mag"] = df["phot_g_mean_mag"] + 5.0 * np.log10(df["parallax"]) - 10.0

    # BP - RP color (also provided as bp_rp; recompute where missing).
    df["bp_rp_color"] = df["bp_rp"]
    recompute = df["bp_rp_color"].isna()
    df.loc[recompute, "bp_rp_color"] = (
        df.loc[recompute, "phot_bp_mean_mag"] - df.loc[recompute, "phot_rp_mean_mag"]
    )

    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with non-finite values in the columns we rely on."""
    needed = [
        "abs_g_mag",
        "bp_rp_color",
        "phot_g_mean_mag",
        "teff_gspphot",
        "lum_flame",
        "spectraltype_esphs",
    ]
    before = len(df)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=needed)
    df = df[df["spectraltype_esphs"].isin(SPECTRAL_ORDER)].reset_index(drop=True)
    print(f"[prep] cleaned {before:,} -> {len(df):,} rows")
    return df


def plot_class_distribution(df: pd.DataFrame, out: Path) -> None:
    counts = df["spectraltype_esphs"].value_counts().reindex(SPECTRAL_ORDER).fillna(0)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(counts.index, counts.values)
    ax.set_xlabel("Spectral type (spectraltype_esphs)")
    ax.set_ylabel("Count")
    ax.set_title("Spectral-type class distribution")
    ax.set_yscale("log")
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{int(v):,}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[prep] saved -> {out}")


def plot_hr_diagram(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    sc = ax.scatter(
        df["teff_gspphot"],
        df["lum_flame"],
        c=df["bp_rp_color"],
        cmap="RdYlBu_r",
        s=3,
        alpha=0.3,
    )
    ax.set_xlabel("Effective temperature teff_gspphot [K]")
    ax.set_ylabel("Luminosity lum_flame [L_sun]")
    ax.set_title("HR diagram (Gaia DR3 sample)")
    ax.invert_xaxis()  # hot stars on the left, by convention
    ax.set_yscale("log")
    fig.colorbar(sc, ax=ax, label="BP - RP color")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[prep] saved -> {out}")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    df = load_raw()
    df = add_features(df)
    df = clean(df)

    print("[prep] class distribution (cleaned):")
    print(
        df["spectraltype_esphs"]
        .value_counts()
        .reindex(SPECTRAL_ORDER)
        .fillna(0)
        .astype(int)
        .to_string()
    )

    plot_class_distribution(df, FIG_DIR / "class_distribution.png")
    plot_hr_diagram(df, FIG_DIR / "hr_diagram.png")

    out_path = DATA_DIR / "gaia_dr3_clean.parquet"
    df.to_parquet(out_path, index=False)
    print(f"[prep] saved cleaned dataset -> {out_path}")


if __name__ == "__main__":
    main()
