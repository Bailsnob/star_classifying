"""Generate a fixed-k=8 GMM experiment without replacing model-search results."""
from pathlib import Path
import sys

# Optional checkout-local dependencies used by the testing page.
local_dependencies = Path(__file__).resolve().parents[1] / ".testing-deps"
if local_dependencies.exists():
    sys.path.insert(0, str(local_dependencies))

import matplotlib
if __name__ == "__main__":
    matplotlib.use("Agg")
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from threadpoolctl import threadpool_limits
import unsupervised as u


def main():
    df = u.prepare(u.load_clean())
    X = u.scale_hr(df)
    y = df[u.TARGET].to_numpy()
    present = [name for name in u.SPECTRAL_ORDER if name in set(y)]
    with threadpool_limits(limits=4):
        km = KMeans(n_clusters=2, n_init=10, random_state=u.RANDOM_STATE)
        labels_km = km.fit_predict(X)
        print("Fitting fixed GMM k=8 ...", flush=True)
        model = GaussianMixture(n_components=8, covariance_type="full", n_init=5,
                                max_iter=u.GMM_MAX_ITER, random_state=u.RANDOM_STATE)
        labels = model.fit_predict(X)
        print(f"GMM converged={model.converged_}; iterations={model.n_iter_}; BIC={model.bic(X):.1f}", flush=True)
        if not model.converged_:
            raise RuntimeError("GMM did not converge; results were not saved.")
        baseline = u.fit_temperature_only(X[:, [0]], 8, "gmm")
    temperature = df["log_teff"].to_numpy()
    labels = u.relabel_hot_to_cool(labels, temperature, 8, "GMM")
    labels_km = u.relabel_hot_to_cool(labels_km, temperature, 2, "KMeans")
    df["cluster_kmeans"] = labels_km
    df["cluster_gmm"] = labels
    df.to_parquet(u.DATA_DIR / "gaia_testing_k8.parquet", index=False)
    frac, _, _ = u.analyze("GMM fixed k=8", df, y, labels, baseline, present)
    u.FIG_DIR.mkdir(exist_ok=True)
    u.plot_hr(df, labels_km, labels, present, u.FIG_DIR / "hr_comparison_k8.png")
    u.plot_contingency(u.row_fractions(y, labels_km, present), frac,
                       u.FIG_DIR / "contingency_k8.png")
    summary = u.summarize(df, labels)
    summary["tag"] = u.assign_tags(summary, u.find_splits(summary))
    summary.to_csv(u.FIG_DIR / "gmm_k8_summary.csv", index=False)


if __name__ == "__main__":
    main()
