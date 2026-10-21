"""
Phase 1 - Data acquisition from the Gaia DR3 archive.

No bulk file download is required: this script queries the Gaia Archive's TAP+
server directly via astroquery and caches the result locally.

We join gaiadr3.gaia_source (photometry, parallax) to
gaiadr3.astrophysical_parameters (spectral type, temperature, luminosity) by
source_id, keeping only rows that have a real spectral-type label and the
fields we need for both the supervised and unsupervised phases.

Why windowing? Joining the two full catalogs and asking for a large result set
makes the archive time out (HTTP 500) for anonymous users. Instead we restrict
gaia_source with its `random_index` column -- a uniform random shuffle of the
catalog -- so each query scans only a few million random rows. We walk through
fixed-size random_index windows and concatenate the results. This keeps every
individual query cheap while still producing an unbiased random sample.

Usage:
    python src/fetch_data.py                   # default: 3 main windows + rare top-up
    python src/fetch_data.py --windows 5       # larger sample
    python src/fetch_data.py --no-topup        # skip rare-class top-up
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from astroquery.gaia import Gaia

# Return every matching row, not astroquery's default cap.
Gaia.ROW_LIMIT = -1

# The eight real spectral-type classes (ESP-ELS also emits 'unknown', excluded).
SPECTRAL_CLASSES = ["O", "B", "A", "F", "G", "K", "M", "CSTAR"]

# Classes a uniform random sample under-represents; topped up separately.
RARE_CLASSES = ["O", "B", "CSTAR"]

# Size of one random_index window. ~3M keeps each query well under the
# server timeout while yielding ~130k labelled rows per window.
DEFAULT_STEP = 3_000_000

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

BASE_COLUMNS = """
  s.source_id, s.ra, s.dec,
  s.phot_g_mean_mag, s.phot_bp_mean_mag, s.phot_rp_mean_mag, s.bp_rp,
  s.parallax, s.parallax_over_error,
  ap.spectraltype_esphs, ap.flags_esphs,
  ap.teff_gspphot, ap.lum_flame
"""

# Shared quality cuts. parallax_over_error > 10 keeps distances (and therefore
# absolute magnitudes) reliable; the rest ensure the features exist and the
# label is a real spectral type.
QUALITY_WHERE = """
  ap.spectraltype_esphs IS NOT NULL
  AND ap.spectraltype_esphs != 'unknown'
  AND ap.lum_flame IS NOT NULL
  AND ap.teff_gspphot IS NOT NULL
  AND s.parallax IS NOT NULL
  AND s.parallax > 0
  AND s.parallax_over_error > 10
  AND s.bp_rp IS NOT NULL
  AND s.phot_g_mean_mag IS NOT NULL
"""


def _run(adql: str) -> pd.DataFrame:
    """Run an async ADQL job and return the results as a pandas DataFrame."""
    job = Gaia.launch_job_async(adql)
    return job.get_results().to_pandas()


def _window_clause(lo: int, hi: int) -> str:
    return f"s.random_index >= {lo} AND s.random_index < {hi}"


def fetch_window(lo: int, hi: int, class_filter: list[str] | None = None) -> pd.DataFrame:
    """Fetch one random_index window, optionally restricted to given classes."""
    where = f"{_window_clause(lo, hi)} AND {QUALITY_WHERE}"
    if class_filter:
        quoted = ", ".join(f"'{c}'" for c in class_filter)
        where += f"\n  AND ap.spectraltype_esphs IN ({quoted})"
    adql = f"""
    SELECT
    {BASE_COLUMNS}
    FROM gaiadr3.gaia_source AS s
    JOIN gaiadr3.astrophysical_parameters AS ap ON s.source_id = ap.source_id
    WHERE {where}
    """
    df = _run(adql)
    return df


def fetch_main(windows: int, step: int) -> pd.DataFrame:
    frames = []
    for i in range(windows):
        lo, hi = i * step, (i + 1) * step
        print(f"[fetch] main window {i + 1}/{windows}  random_index [{lo:,}, {hi:,}) ...")
        df = fetch_window(lo, hi)
        print(f"[fetch]   -> {len(df):,} rows")
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def fetch_rare_topup(start_window: int, windows: int, step: int) -> pd.DataFrame:
    """Extra windows (higher random_index range) restricted to rare classes."""
    frames = []
    for i in range(windows):
        lo = (start_window + i) * step
        hi = lo + step
        print(
            f"[fetch] rare top-up {i + 1}/{windows}  "
            f"random_index [{lo:,}, {hi:,})  classes={RARE_CLASSES} ..."
        )
        df = fetch_window(lo, hi, class_filter=RARE_CLASSES)
        print(f"[fetch]   -> {len(df):,} rows")
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch a Gaia DR3 sample.")
    parser.add_argument("--windows", type=int, default=3,
                        help="Number of random_index windows for the main sample (default: 3).")
    parser.add_argument("--rare-windows", type=int, default=5,
                        help="Extra windows restricted to rare classes (default: 5).")
    parser.add_argument("--step", type=int, default=DEFAULT_STEP,
                        help=f"random_index window size (default: {DEFAULT_STEP}).")
    parser.add_argument("--no-topup", action="store_true",
                        help="Skip the rare-class top-up queries.")
    parser.add_argument("--out", type=str, default=None,
                        help="Output path (default: data/gaia_dr3_sample.parquet).")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else DATA_DIR / "gaia_dr3_sample.parquet"

    frames = [fetch_main(args.windows, args.step)]

    if not args.no_topup and args.rare_windows > 0:
        frames.append(fetch_rare_topup(args.windows, args.rare_windows, args.step))

    df = pd.concat(frames, ignore_index=True)
    before = len(df)
    df = df.drop_duplicates(subset="source_id").reset_index(drop=True)
    print(f"[fetch] combined {before:,} rows -> {len(df):,} unique sources")

    print("[fetch] class distribution:")
    print(df["spectraltype_esphs"].value_counts().to_string())

    df.to_parquet(out_path, index=False)
    print(f"[fetch] saved -> {out_path}")


if __name__ == "__main__":
    main()
