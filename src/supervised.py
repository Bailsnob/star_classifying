"""
Phase 1 - Supervised classification of spectral type from photometry.

Target : spectraltype_esphs  (O, B, A, F, G, K, M, CSTAR)
Features: photometry / colors only -- deliberately NOT temperature -- so the
          model learns the color-to-type relation rather than copying a
          temperature bucket.

Because the classes are highly imbalanced, we use stratified splits, class
weighting, and report macro-F1 plus a confusion matrix rather than raw accuracy.

Usage:
    python src/supervised.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    ConfusionMatrixDisplay,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FIG_DIR = Path(__file__).resolve().parent.parent / "figures"

# Photometric features only (no teff / lum, those are reserved for Phase 2).
FEATURES = [
    "bp_rp_color",
    "abs_g_mag",
    "phot_g_mean_mag",
    "phot_bp_mean_mag",
    "phot_rp_mean_mag",
]
TARGET = "spectraltype_esphs"
SPECTRAL_ORDER = ["O", "B", "A", "F", "G", "K", "M", "CSTAR"]


def load_clean() -> pd.DataFrame:
    path = DATA_DIR / "gaia_dr3_clean.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python src/preprocess.py` first."
        )
    return pd.read_parquet(path)


def build_models() -> dict:
    """Return the models to evaluate, each handling class imbalance."""
    return {
        "logreg": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
            ),
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        ),
    }


def plot_confusion(y_true, y_pred, labels, out: Path, title: str) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, cmap="Blues", colorbar=True, values_format=".2f")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[sup] saved -> {out}")


def plot_feature_importance(model, out: Path) -> None:
    rf = model
    if not hasattr(rf, "feature_importances_"):
        return
    importances = pd.Series(rf.feature_importances_, index=FEATURES).sort_values()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(importances.index, importances.values)
    ax.set_xlabel("Importance")
    ax.set_title("Random forest feature importance")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[sup] saved -> {out}")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = load_clean()
    X = df[FEATURES].to_numpy()
    y = df[TARGET].to_numpy()

    present_labels = [c for c in SPECTRAL_ORDER if c in set(y)]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )
    print(f"[sup] train={len(X_train):,}  test={len(X_test):,}")

    results = {}
    for name, model in build_models().items():
        print(f"\n[sup] === {name} ===")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        macro_f1 = f1_score(y_test, y_pred, average="macro", labels=present_labels)
        results[name] = macro_f1
        print(f"[sup] macro-F1 = {macro_f1:.3f}")
        print(
            classification_report(
                y_test, y_pred, labels=present_labels, zero_division=0
            )
        )

        plot_confusion(
            y_test,
            y_pred,
            present_labels,
            FIG_DIR / f"confusion_{name}.png",
            f"Confusion matrix (row-normalized) - {name}",
        )
        if name == "random_forest":
            plot_feature_importance(model, FIG_DIR / "feature_importance.png")

    print("\n[sup] macro-F1 summary:")
    for name, score in sorted(results.items(), key=lambda kv: -kv[1]):
        print(f"  {name:<14} {score:.3f}")


if __name__ == "__main__":
    main()
