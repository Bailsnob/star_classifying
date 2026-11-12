"""
Phase 2 - Unsupervised clustering on the HR diagram.

Clusters the cleaned Gaia sample on temperature and luminosity only
(log10 teff_gspphot, log10 lum_flame). Spectral type is never a feature.
Photometry stays out so this is not a second copy of the supervised model.

KMeans and a full-covariance Gaussian mixture each choose k on the HR
diagram. A temperature-only clustering at that same k is the baseline.
The baseline holds k fixed to control the feature comparison. Adjusted
Rand Index can also compare partitions with different numbers of groups.

The comparison against spectraltype_esphs is Adjusted Rand Index plus a
row-normalized contingency table. Whether luminosity adds stable structure
beyond temperature is a hypothesis. ARI differences alone do not establish
giant/dwarf separation or validate physical populations.

Usage:
    python src/unsupervised.py
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FIG_DIR = Path(__file__).resolve().parent.parent / "figures"

TARGET = "spectraltype_esphs"
SPECTRAL_ORDER = ["O", "B", "A", "F", "G", "K", "M", "CSTAR"]

RANDOM_STATE = 42
K_RANGE = range(2, 11)
SILHOUETTE_SAMPLE = 10_000
# Among k values this close to the best silhouette, keep the smallest.
SILHOUETTE_TOL = 0.01
GMM_MAX_ITER = 200
# Similar median temperature and different median luminosity: exploratory hints.
TEFF_SPLIT_DEX = 0.05
LUM_SPLIT_DEX = 0.5
# A cluster "contains" a spectral type when it holds at least this share of it.
TYPE_SHARE = 0.10


def load_clean() -> pd.DataFrame:
    path = DATA_DIR / "gaia_dr3_clean.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python src/preprocess.py` first."
        )
    return pd.read_parquet(path)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Log-transform temperature and luminosity and drop non-positive rows.

    clean() only removes NaNs and infs. log10 of a non-positive luminosity
    or temperature is undefined, so that guard lives here.
    """
    required = ["teff_gspphot", "lum_flame", TARGET]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"Cleaned sample is missing columns: {missing}")

    before = len(df)
    df = df.copy()
    positive = (df["teff_gspphot"] > 0) & (df["lum_flame"] > 0)
    df = df.loc[positive].copy()
    df["log_teff"] = np.log10(df["teff_gspphot"].to_numpy(dtype=float))
    df["log_lum"] = np.log10(df["lum_flame"].to_numpy(dtype=float))
    finite = np.isfinite(df["log_teff"]) & np.isfinite(df["log_lum"])
    df = df.loc[finite].reset_index(drop=True)
    if df.empty:
        raise RuntimeError(
            "No rows left after requiring positive finite temperature and luminosity."
        )
    print(
        f"[uns] dropped {before - len(df):,} rows with non-positive "
        "or non-finite teff/lum"
    )
    print(f"[uns] clustering sample: {len(df):,} rows")
    return df


def scale_hr(df: pd.DataFrame) -> np.ndarray:
    """Standardize log temperature and log luminosity.

    KMeans assumes spherical clusters in Euclidean space. Luminosity spans
    decades and the HR diagram is read in log-log coordinates, so both axes
    are logged before scaling. There is no held-out set in this phase, so
    the scaler is fit on every row. Rows are not class-weighted: cluster
    density should follow the sample.
    """
    scaler = StandardScaler()
    return scaler.fit_transform(df[["log_teff", "log_lum"]].to_numpy(dtype=float))


def silhouette_index(n: int) -> np.ndarray:
    """One fixed row subset, reused for every k. Pairwise silhouette on the
    full sample is too expensive once the catalog sample is large.
    """
    if n < 2:
        raise RuntimeError("Need at least two rows to compute a silhouette score.")
    if n <= SILHOUETTE_SAMPLE:
        return np.arange(n)
    rng = np.random.default_rng(RANDOM_STATE)
    return rng.choice(n, size=SILHOUETTE_SAMPLE, replace=False)


def safe_silhouette(X_sub: np.ndarray, labels_sub: np.ndarray) -> float:
    n_labels = len(np.unique(labels_sub))
    if not 2 <= n_labels < len(X_sub):
        return float("nan")
    return float(silhouette_score(X_sub, labels_sub))


def choose_k_silhouette(rows: list[dict]) -> int:
    """Smallest k within SILHOUETTE_TOL of the best mean silhouette."""
    finite = [row for row in rows if np.isfinite(row["silhouette"])]
    if not finite:
        raise RuntimeError("Silhouette could not be computed for any k.")
    best = max(row["silhouette"] for row in finite)
    pool = [row for row in finite if row["silhouette"] >= best - SILHOUETTE_TOL]
    return int(min(pool, key=lambda row: row["k"])["k"])


def choose_k_bic(rows: list[dict]) -> int:
    """Minimum finite BIC among converged fits. Ties keep the smaller k."""
    eligible = [row for row in rows
                if row.get("converged", True) and np.isfinite(row["bic"])]
    if not eligible:
        raise RuntimeError("No converged GMM with finite BIC is available for selection.")
    return int(min(eligible, key=lambda row: (row["bic"], row["k"]))["k"])


def search_kmeans(X: np.ndarray, idx: np.ndarray) -> tuple[pd.DataFrame, int, np.ndarray]:
    """Fit every k on the full sample. Choose k by subsample silhouette.

    Inertia is recorded for the elbow plot and is not used to pick k.
    """
    print("\n[uns] === KMeans k search ===")
    rows: list[dict] = []
    stored: dict[int, np.ndarray] = {}
    X_sub = X[idx]
    for k in K_RANGE:
        print(f"[uns]   k={k} fitting on {len(X):,} rows ...", flush=True)
        model = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
        labels = model.fit_predict(X)
        silhouette = safe_silhouette(X_sub, labels[idx])
        rows.append({"k": k, "inertia": float(model.inertia_), "silhouette": silhouette})
        stored[k] = labels
        sil_text = f"{silhouette:.3f}" if np.isfinite(silhouette) else "nan"
        print(f"[uns]   k={k}  inertia={model.inertia_:.1f}  silhouette={sil_text}")

    k_star = choose_k_silhouette(rows)
    best = max(row["silhouette"] for row in rows if np.isfinite(row["silhouette"]))
    chosen_sil = next(row["silhouette"] for row in rows if row["k"] == k_star)
    print(
        f"[uns] selected k={k_star} "
        f"(silhouette {chosen_sil:.3f}; best {best:.3f}; "
        f"smallest k within {SILHOUETTE_TOL})"
    )
    labels = stored.pop(k_star)
    del stored
    return pd.DataFrame(rows), k_star, labels


def search_gmm(X: np.ndarray) -> tuple[pd.DataFrame, int, np.ndarray]:
    """Fit a full-covariance mixture for every k. Choose k by minimum BIC.

    Full covariance lets a component tilt along the main sequence instead of
    cutting that diagonal into round blobs. A density component need not be
    a distinct physical population. Failed fits are recorded but excluded
    from selection. Exact BIC ties keep the smaller k.
    """
    print("\n[uns] === GMM k search ===")
    rows: list[dict] = []
    stored: dict[int, np.ndarray] = {}
    for k in K_RANGE:
        print(f"[uns]   k={k} fitting on {len(X):,} rows ...", flush=True)
        model = GaussianMixture(
            n_components=k,
            covariance_type="full",
            n_init=5,
            max_iter=GMM_MAX_ITER,
            random_state=RANDOM_STATE,
        )
        labels = model.fit_predict(X)
        bic = float(model.bic(X))
        converged = bool(model.converged_)
        rows.append({"k": k, "bic": bic, "converged": converged,
                     "n_iter": int(model.n_iter_)})
        stored[k] = labels
        print(f"[uns]   k={k}  bic={bic:.1f}  n_iter={model.n_iter_}  converged={converged}")
        if not converged or not np.isfinite(bic):
            warnings.warn(f"GMM k={k} excluded: nonconverged fit or nonfinite BIC.",
                          RuntimeWarning, stacklevel=2)

    k_star = choose_k_bic(rows)
    chosen_bic = min(row["bic"] for row in rows if row["k"] == k_star)
    print(f"[uns] selected k={k_star} (minimum eligible BIC {chosen_bic:.1f} among tested k)")
    if len(rows) > 1 and k_star == max(row["k"] for row in rows):
        warnings.warn(
            f"GMM selected k={k_star} at the upper search boundary; "
            "this is the best tested k, not an established unrestricted optimum.",
            RuntimeWarning, stacklevel=2,
        )
    labels = stored.pop(k_star)
    del stored
    return pd.DataFrame(rows), k_star, labels


def fit_temperature_only(X_teff: np.ndarray, k: int, kind: str) -> np.ndarray:
    """Cluster on standardized log temperature alone, at the HR model's k."""
    print(f"[uns]   fitting {kind} k={k} on log temperature only ...", flush=True)
    if kind == "kmeans":
        model = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
    elif kind == "gmm":
        model = GaussianMixture(
            n_components=k,
            covariance_type="full",
            n_init=5,
            max_iter=GMM_MAX_ITER,
            random_state=RANDOM_STATE,
        )
    else:
        raise ValueError(f"Unknown model kind: {kind}")
    labels = model.fit_predict(X_teff)
    if kind == "gmm" and not model.converged_:
        raise RuntimeError(f"Temperature-only GMM k={k} did not converge; ARI comparison aborted.")
    return labels


def relabel_hot_to_cool(
    labels: np.ndarray, log_teff: np.ndarray, requested_k: int, name: str
) -> np.ndarray:
    """Renumber clusters so 0 is the hottest median temperature.

    Adjusted Rand Index ignores the numbering. The contingency table is
    easier to read when columns run from hot to cool. Components with no
    members are dropped, so the new ids are contiguous.
    """
    frame = pd.DataFrame({"cluster": labels, "log_teff": log_teff})
    order = (
        frame.groupby("cluster", as_index=False)["log_teff"]
        .median()
        .sort_values(["log_teff", "cluster"], ascending=[False, True])
    )
    mapping = {int(old): new for new, old in enumerate(order["cluster"].tolist())}
    n_present = len(mapping)
    n_empty = requested_k - n_present
    if n_empty > 0:
        noun = "component" if n_empty == 1 else "components"
        print(
            f"[uns] note: {name} dropped {n_empty} empty {noun} before relabeling"
        )
    out = np.empty(len(labels), dtype=int)
    for old, new in mapping.items():
        out[labels == old] = new
    return out


def summarize(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    rows = []
    for cluster in range(int(labels.max()) + 1):
        sub = df.loc[labels == cluster]
        if sub.empty:
            continue
        counts = sub[TARGET].value_counts()
        log_lum = sub["log_lum"].to_numpy()
        p10, p90 = np.percentile(log_lum, [10, 90])
        rows.append(
            {
                "cluster": int(cluster),
                "n": int(len(sub)),
                "median_teff": float(sub["teff_gspphot"].median()),
                "median_lum": float(sub["lum_flame"].median()),
                "median_log_teff": float(sub["log_teff"].median()),
                "median_log_lum": float(sub["log_lum"].median()),
                "log_lum_p10": float(p10),
                "log_lum_p90": float(p90),
                "mode_type": counts.index[0],
                "mode_frac": float(counts.iloc[0] / len(sub)),
            }
        )
    return pd.DataFrame(rows)


def find_splits(summary: pd.DataFrame) -> list[tuple[int, int, float, float]]:
    """Pairs close in log temperature and separated in log luminosity."""
    pairs = []
    records = summary.to_dict("records")
    for left, right in combinations(records, 2):
        d_teff = abs(left["median_log_teff"] - right["median_log_teff"])
        d_lum = abs(left["median_log_lum"] - right["median_log_lum"])
        if d_teff < TEFF_SPLIT_DEX and d_lum > LUM_SPLIT_DEX:
            pairs.append(
                (int(left["cluster"]), int(right["cluster"]), float(d_teff), float(d_lum))
            )
    return pairs


def assign_tags(
    summary: pd.DataFrame, pairs: list[tuple[int, int, float, float]]
) -> list[str]:
    """Hints for the write-up, not new labels.

    The higher-luminosity member of a close-temperature pair is giant-like.
    The lower-luminosity member is dwarf-like.
    """
    luminosity = summary.set_index("cluster")["median_log_lum"]
    higher: set[int] = set()
    lower: set[int] = set()
    for left, right, _d_teff, _d_lum in pairs:
        if luminosity[left] >= luminosity[right]:
            higher.add(left)
            lower.add(right)
        else:
            higher.add(right)
            lower.add(left)

    tags = []
    for cluster in summary["cluster"]:
        cluster = int(cluster)
        is_high = cluster in higher
        is_low = cluster in lower
        if is_high and is_low:
            tags.append("mixed (both sides of a luminosity split)")
        elif is_high:
            tags.append("giant-like (higher L at similar Teff)")
        elif is_low:
            tags.append("dwarf-like (lower L at similar Teff)")
        else:
            tags.append("no luminosity twin")
    return tags


def row_fractions(
    y: np.ndarray, labels: np.ndarray, present: list[str]
) -> pd.DataFrame:
    """Share of each spectral type that fell in each cluster."""
    k = int(labels.max()) + 1
    counts = pd.crosstab(pd.Series(y), pd.Series(labels))
    counts = counts.reindex(index=present, columns=list(range(k)), fill_value=0)
    totals = counts.sum(axis=1)
    frac = counts.div(totals.replace(0, np.nan), axis=0)
    frac.index.name = "spectral_type"
    frac.columns.name = "cluster"
    return frac


def describe_type_spread(spread: pd.Series) -> None:
    print("[uns] clusters containing >= 10% of each spectral type:")
    for spectral_type, n_clusters in spread.items():
        print(f"[uns]   {spectral_type:<6} {int(n_clusters)}")
    early = spread.reindex(["O", "B"]).dropna()
    late = spread.reindex(["G", "K", "M"]).dropna()
    if early.empty or late.empty:
        return
    print(
        f"[uns] mean clusters per type: O/B={early.mean():.2f}  "
        f"G/K/M={late.mean():.2f}"
    )
    if late.mean() > early.mean():
        print("[uns] G/K/M occupy more clusters on average than O/B")
    else:
        print("[uns] G/K/M do not occupy more clusters on average than O/B")


def print_summary(summary: pd.DataFrame) -> None:
    show = pd.DataFrame(
        {
            "cluster": summary["cluster"].astype(int),
            "n": summary["n"].astype(int),
            "median_teff": summary["median_teff"].map(lambda v: f"{v:.0f}"),
            "median_lum": summary["median_lum"].map(lambda v: f"{v:.4g}"),
            "log_lum_p10": summary["log_lum_p10"].map(lambda v: f"{v:.2f}"),
            "log_lum_p90": summary["log_lum_p90"].map(lambda v: f"{v:.2f}"),
            "mode_type": summary["mode_type"],
            "mode_frac": summary["mode_frac"].map(lambda v: f"{v:.3f}"),
            "tag": summary["tag"],
        }
    )
    print("[uns] cluster summary:")
    print(show.to_string(index=False))


def print_splits(pairs: list[tuple[int, int, float, float]]) -> None:
    print(
        "[uns] luminosity-split pairs "
        f"(|d log10 Teff| < {TEFF_SPLIT_DEX}, |d log10 L| > {LUM_SPLIT_DEX}):"
    )
    if not pairs:
        print("[uns]   none")
        return
    for left, right, d_teff, d_lum in pairs:
        print(
            f"[uns]   cluster {left} vs {right}: "
            f"dlogT={d_teff:.3f}  dlogL={d_lum:.3f}"
        )


def analyze(
    name: str,
    df: pd.DataFrame,
    y: np.ndarray,
    labels_hr: np.ndarray,
    labels_teff: np.ndarray,
    present: list[str],
) -> tuple[pd.DataFrame, float, float]:
    """Print ARI, contingency, cluster summary, and luminosity splits."""
    ari_hr = float(adjusted_rand_score(y, labels_hr))
    ari_teff = float(adjusted_rand_score(y, labels_teff))
    print(f"\n[uns] === {name} ===")
    print(f"[uns] ARI hr={ari_hr:.3f}  ARI temperature-only={ari_teff:.3f}")
    if ari_teff > ari_hr:
        print("[uns] temperature-only ARI is higher than HR ARI")
    else:
        print("[uns] temperature-only ARI is not higher than HR ARI")

    frac = row_fractions(y, labels_hr, present)
    print("[uns] row-normalized contingency (share of each spectral type):")
    with pd.option_context("display.width", 120, "display.max_columns", 20):
        print(frac.round(3).to_string())

    summary = summarize(df, labels_hr)
    pairs = find_splits(summary)
    summary["tag"] = assign_tags(summary, pairs)
    print_summary(summary)
    print_splits(pairs)
    describe_type_spread((frac >= TYPE_SHARE).sum(axis=1))
    return frac, ari_hr, ari_teff


def save_figure(fig: matplotlib.figure.Figure, out: Path) -> None:
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[uns] saved -> {out}")


def plot_selection(
    kmeans_table: pd.DataFrame,
    gmm_table: pd.DataFrame,
    k_kmeans: int,
    k_gmm: int,
    out: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].plot(kmeans_table["k"], kmeans_table["inertia"], marker="o")
    axes[0].axvline(
        k_kmeans,
        color="black",
        ls="--",
        lw=1,
        label=f"selected k={k_kmeans} (silhouette)",
    )
    axes[0].set_title("KMeans inertia")
    axes[0].set_ylabel("Inertia")

    axes[1].plot(kmeans_table["k"], kmeans_table["silhouette"], marker="o")
    axes[1].axvline(
        k_kmeans, color="black", ls="--", lw=1, label=f"selected k={k_kmeans}"
    )
    axes[1].set_title("KMeans silhouette")
    axes[1].set_ylabel("Mean silhouette")

    axes[2].plot(gmm_table["k"], gmm_table["bic"], marker="o")
    axes[2].axvline(k_gmm, color="black", ls="--", lw=1, label=f"selected k={k_gmm}")
    axes[2].set_title("GMM BIC")
    axes[2].set_ylabel("BIC")

    for ax in axes:
        ax.set_xlabel("k")
        ax.set_xticks(list(K_RANGE))
        ax.legend(fontsize=8)
    save_figure(fig, out)


def scatter_discrete(ax, x, y, codes, names, title) -> None:
    n = len(names)
    cmap = matplotlib.colormaps["tab10"].resampled(n)
    palette = cmap(np.linspace(0, 1, n))
    ax.scatter(
        x,
        y,
        c=palette[np.asarray(codes, dtype=int)],
        s=2,
        alpha=0.3,
        linewidths=0,
        rasterized=True,
    )
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=palette[i],
            markeredgecolor="none",
            markersize=6,
            label=str(names[i]),
        )
        for i in range(n)
    ]
    ax.legend(
        handles=handles,
        loc="upper left",
        fontsize=6 if n > 6 else 7,
        framealpha=0.9,
        borderaxespad=0.2,
        ncol=2 if n > 6 else 1,
    )
    ax.set_title(title)
    ax.set_xlabel("log10 Teff [K]")
    ax.set_ylabel("log10 luminosity [L_sun]")


def plot_hr(
    df: pd.DataFrame,
    labels_kmeans: np.ndarray,
    labels_gmm: np.ndarray,
    present: list[str],
    out: Path,
) -> None:
    x = df["log_teff"].to_numpy()
    y = df["log_lum"].to_numpy()
    type_code = df[TARGET].map({name: i for i, name in enumerate(present)}).to_numpy()
    k_kmeans = int(labels_kmeans.max()) + 1
    k_gmm = int(labels_gmm.max()) + 1

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    scatter_discrete(axes[0], x, y, type_code, present, "Spectral type")
    scatter_discrete(
        axes[1],
        x,
        y,
        labels_kmeans,
        [str(i) for i in range(k_kmeans)],
        f"KMeans (k={k_kmeans})",
    )
    scatter_discrete(
        axes[2],
        x,
        y,
        labels_gmm,
        [str(i) for i in range(k_gmm)],
        f"GMM (k={k_gmm})",
    )
    # Hot stars on the left, matching the usual HR diagram.
    axes[0].invert_xaxis()
    save_figure(fig, out)


def plot_contingency(frac_kmeans: pd.DataFrame, frac_gmm: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    panels = (
        (axes[0], frac_kmeans, "KMeans"),
        (axes[1], frac_gmm, "GMM"),
    )
    for ax, frac, title in panels:
        sns.heatmap(
            frac,
            ax=ax,
            cmap="Blues",
            vmin=0,
            vmax=1,
            annot=True,
            fmt=".2f",
            annot_kws={"size": 7},
            cbar_kws={"label": "share of spectral type"},
        )
        ax.set_title(f"Row-normalized contingency - {title}")
        ax.set_xlabel("cluster")
        ax.set_ylabel("spectral type")
    save_figure(fig, out)


def print_hypothesis() -> None:
    print("[uns] HR features: log10(teff_gspphot), log10(lum_flame), standardized")
    print("[uns] temperature-only baseline: log10(teff_gspphot), same k and algorithm")
    print("[uns] label: spectraltype_esphs (comparison only, not a feature)")
    print(
        "[uns] Hypothesis to investigate: luminosity may separate stars of "
        "similar temperature. ARI and whole-cluster median differences are "
        "descriptive, not independent evidence of giant/dwarf populations."
    )


def print_caveats() -> None:
    print("\n[uns] caveats:")
    print(
        "[uns]   The rare-class top-up over-represents O, B, and CSTAR. "
        "Cluster sizes are not the true Galactic mix."
    )
    print(
        "[uns]   teff_gspphot, lum_flame, and spectraltype_esphs are Gaia "
        "model products. This phase clusters derived parameters, not raw photometry."
    )
    print(
        "[uns]   parallax_over_error > 10 and the requirement of luminosity "
        "and spectral type remove white dwarfs and many distant giants."
    )
    print(
        "[uns]   CSTAR is a chemical peculiarity, not an HR-diagram island. "
        "Failing to isolate it is expected."
    )


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print_hypothesis()

    df = prepare(load_clean())
    if len(df) <= max(K_RANGE):
        raise RuntimeError(
            f"Need more than {max(K_RANGE)} rows to search k through {max(K_RANGE)}."
        )

    # Column 0 is standardized log temperature on its own, the same scale
    # the HR clustering uses for that axis.
    X_hr = scale_hr(df)
    X_teff = X_hr[:, [0]]
    idx = silhouette_index(len(df))
    print(f"[uns] silhouette subsample: {len(idx):,} rows")

    kmeans_table, k_kmeans, labels_kmeans = search_kmeans(X_hr, idx)
    gmm_table, k_gmm, labels_gmm = search_gmm(X_hr)
    kmeans_table.to_csv(FIG_DIR / "kmeans_selection.csv", index=False)
    gmm_table.to_csv(FIG_DIR / "gmm_selection.csv", index=False)

    print("\n[uns] === temperature-only baselines ===")
    labels_kmeans_teff = fit_temperature_only(X_teff, k_kmeans, "kmeans")
    labels_gmm_teff = fit_temperature_only(X_teff, k_gmm, "gmm")

    log_teff = df["log_teff"].to_numpy()
    labels_kmeans = relabel_hot_to_cool(labels_kmeans, log_teff, k_kmeans, "KMeans")
    labels_gmm = relabel_hot_to_cool(labels_gmm, log_teff, k_gmm, "GMM")

    y = df[TARGET].to_numpy()
    present = [spectral_type for spectral_type in SPECTRAL_ORDER if spectral_type in set(y)]

    frac_kmeans, ari_kmeans_hr, ari_kmeans_teff = analyze(
        "KMeans", df, y, labels_kmeans, labels_kmeans_teff, present
    )
    frac_gmm, ari_gmm_hr, ari_gmm_teff = analyze(
        "GMM", df, y, labels_gmm, labels_gmm_teff, present
    )

    print("\n[uns] ARI recap (HR vs temperature-only):")
    print(f"[uns]   KMeans  {ari_kmeans_hr:.3f}  vs  {ari_kmeans_teff:.3f}")
    print(f"[uns]   GMM     {ari_gmm_hr:.3f}  vs  {ari_gmm_teff:.3f}")
    pd.DataFrame([
        {"model": "KMeans", "k": k_kmeans, "ari_hr": ari_kmeans_hr,
         "ari_temperature_only": ari_kmeans_teff},
        {"model": "GMM", "k": k_gmm, "ari_hr": ari_gmm_hr,
         "ari_temperature_only": ari_gmm_teff},
    ]).to_csv(FIG_DIR / "ari_summary.csv", index=False)

    plot_selection(
        kmeans_table,
        gmm_table,
        k_kmeans,
        k_gmm,
        FIG_DIR / "cluster_selection.png",
    )
    plot_hr(df, labels_kmeans, labels_gmm, present, FIG_DIR / "hr_comparison.png")
    plot_contingency(frac_kmeans, frac_gmm, FIG_DIR / "contingency.png")
    print_caveats()


if __name__ == "__main__":
    main()
