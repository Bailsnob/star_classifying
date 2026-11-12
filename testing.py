"""One-window Python viewer for Gaia clustering. Run: python testing.py.

Use --save to render every view without opening a desktop window.
"""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parent
if (ROOT / ".testing-deps").exists():
    sys.path.insert(0, str(ROOT / ".testing-deps"))
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", action="store_true", help="Save all views without a GUI")
    args = parser.parse_args()
    import matplotlib
    if args.save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button
    import pandas as pd
    import unsupervised as u

    cache = ROOT / "data/gaia_testing_k8.parquet"
    source = ROOT / "data/gaia_dr3_clean.parquet"
    if not cache.exists() or (source.exists() and source.stat().st_mtime > cache.stat().st_mtime):
        from test_gmm_k8 import main as generate
        generate()
    df = pd.read_parquet(cache)
    present = [name for name in u.SPECTRAL_ORDER if name in set(df[u.TARGET])]
    fig = plt.figure(figsize=(15, 6.5))
    if fig.canvas.manager is not None:
        fig.canvas.manager.set_window_title("Gaia DR3 - Unsupervised testing")
    buttons = []
    content_axes = []

    def show(view):
        for ax in content_axes:
            ax.remove()
        content_axes.clear()
        fig.suptitle("Gaia DR3 | GMM fixed k=8 | KMeans k=2", fontsize=16)
        if view == "HR comparison":
            axes = fig.subplots(1, 3, sharex=True, sharey=True)
            content_axes.extend(axes)
            codes = df[u.TARGET].map({name: i for i, name in enumerate(present)})
            for ax, labels, names, title in zip(
                axes, [codes, df.cluster_kmeans, df.cluster_gmm],
                [present, range(2), range(8)], ["Spectral type", "KMeans (k=2)", "GMM (k=8)"]
            ):
                u.scatter_discrete(ax, df.log_teff, df.log_lum, labels, names, title)
            axes[0].invert_xaxis()
        elif view == "Density":
            ax = fig.add_subplot(111)
            content_axes.append(ax)
            density = ax.hexbin(df.log_teff, df.log_lum, gridsize=150,
                                bins="log", mincnt=1, cmap="inferno")
            colorbar = fig.colorbar(density, ax=ax)
            colorbar.set_label("Star count (logarithmic color scale)")
            content_axes.append(colorbar.ax)
            ax.invert_xaxis()
            ax.set(xlabel="log10 Teff [K]", ylabel="log10 luminosity [L_sun]",
                   title=f"All {len(df):,} stars - density")
        else:
            name = "contingency_k8.png" if view == "Agreement" else "cluster_selection.png"
            path = ROOT / "figures" / name
            ax = fig.add_subplot(111)
            content_axes.append(ax)
            if path.exists():
                ax.imshow(plt.imread(path))
            else:
                ax.text(.5, .5, f"Missing {name}. Run python src/unsupervised.py.",
                        ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")
            if view == "Original k search":
                ax.set_title("Original search: GMM selected k=10; this experiment fixes k=8")
        fig.subplots_adjust(left=.07, right=.95, bottom=.22, top=.84, wspace=.25)
        fig.canvas.draw_idle()

    views = ["HR comparison", "Density", "Agreement", "Original k search"]
    for i, view in enumerate(views):
        ax = fig.add_axes([.08 + i * .225, .055, .20, .055])
        button = Button(ax, view)
        button.on_clicked(lambda event, name=view: show(name))
        buttons.append(button)
    if args.save:
        for view in views:
            show(view)
            path = ROOT / "figures" / ("testing_" + view.lower().replace(" ", "_") + ".png")
            fig.savefig(path, dpi=150)
            print(f"Saved {path}")
        plt.close(fig)
    else:
        show(views[0])
        plt.show()


if __name__ == "__main__":
    main()
