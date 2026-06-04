"""
download_gbs.py
===============
Download the Gaia FGK Benchmark Stars v3 (Soubiran+ 2024) from VizieR/CDS.

What this downloads
-------------------
1. GBS v3 parameter catalogue  (J/A+A/682/A145) — Teff, logg, [Fe/H], HIP ID
2. GBS v3 spectral library FITS table (Jofré+ 2026, J/A+A companion paper)
   — 522 high-resolution (R≈42 000, S/N>100) normalised spectra, 480–680 nm

Output layout
-------------
data/gbs/
├── gbs_v3_params.fits          ← parameter catalogue (192+9 stars)
├── gbs_v3_params.csv           ← same, CSV convenience copy
└── spectra/
    └── gbs_v3_spectra.fits     ← spectral library FITS table

Usage
-----
    python download_gbs.py [--outdir data/gbs] [--no-spectra]
"""

import argparse
import os
import sys
import time
import urllib.request
import urllib.error

# ── VizieR TAP / FTP endpoints ────────────────────────────────────────────────
# Parameter catalogue: J/A+A/682/A145  (Soubiran+ 2024, GBS v3 Teff/logg)
VIZIER_TAP   = "https://vizier.cds.unistra.fr/viz-bin/votable"
CATALOGUE_ID = "J/A+A/682/A145"
TABLE_NAME   = "J/A+A/682/A145/catalog"   # main table inside the catalogue

# Spectral library FITS (Jofré+ 2026, companion paper J/A+A/…/table_spectra)
# CDS anonymous FTP mirror — direct FITS download
SPECTRA_URL  = (
    "https://cdsarc.cds.unistra.fr/ftp/J/A+A/682/A145/spectra.fits.gz"
)

# Fallback: if the spectral FITS is not yet at CDS, the IAG Würzburg mirror
SPECTRA_URL_MIRROR = (
    "https://cdsarc.u-strasbg.fr/ftp/J/A+A/682/A145/spectra.fits.gz"
)

CHUNK = 1 << 16   # 64 kB download chunks


# ── Helpers ───────────────────────────────────────────────────────────────────

def _progress(label: str, done: int, total: int | None) -> None:
    if total:
        pct = 100 * done // total
        bar = "#" * (pct // 5) + "-" * (20 - pct // 5)
        print(f"\r  {label}  [{bar}] {pct:3d}%  ({done/1e6:.1f}/{total/1e6:.1f} MB)",
              end="", flush=True)
    else:
        print(f"\r  {label}  {done/1e6:.1f} MB downloaded", end="", flush=True)


def _download(url: str, dest: str, label: str) -> bool:
    """Download *url* to *dest*; return True on success."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "GBS-downloader/1.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            total = int(resp.headers.get("Content-Length", 0)) or None
            done  = 0
            with open(dest, "wb") as fh:
                while True:
                    chunk = resp.read(CHUNK)
                    if not chunk:
                        break
                    fh.write(chunk)
                    done += len(chunk)
                    _progress(label, done, total)
        print()   # newline after progress bar
        return True
    except urllib.error.URLError as exc:
        print(f"\n  [WARN] {url} failed: {exc}")
        return False


def download_catalogue(outdir: str) -> str:
    """
    Query VizieR TAP for the GBS v3 parameter table and save as FITS + CSV.
    Returns the path to the saved FITS file.
    """
    import urllib.parse

    fits_path = os.path.join(outdir, "gbs_v3_params.fits")
    csv_path  = os.path.join(outdir, "gbs_v3_params.csv")

    if os.path.exists(fits_path):
        print(f"  [SKIP] {fits_path} already exists")
        return fits_path

    os.makedirs(outdir, exist_ok=True)

    # ---- ADQL query via VizieR TAP ----------------------------------------
    adql = (
        f'SELECT * FROM "{TABLE_NAME}"'
    )
    params = urllib.parse.urlencode({
        "REQUEST": "doQuery",
        "LANG":    "ADQL",
        "FORMAT":  "fits",
        "QUERY":   adql,
    })
    tap_url = f"https://tapvizier.cds.unistra.fr/TAPVizieR/tap/sync?{params}"

    print(f"\n[1/3] Downloading GBS v3 parameter catalogue from VizieR TAP …")
    print(f"      Table : {TABLE_NAME}")

    ok = _download(tap_url, fits_path, "catalogue")
    if not ok:
        # Fallback: wget-style direct URL from CDS FTP
        ftp_url = "https://cdsarc.cds.unistra.fr/ftp/J/A+A/682/A145/catalog.fits"
        print(f"  [RETRY] Trying direct CDS FTP …")
        ok = _download(ftp_url, fits_path, "catalogue (ftp)")
    if not ok:
        raise RuntimeError(
            "Could not download GBS v3 parameter catalogue.\n"
            "Check your internet connection or visit:\n"
            "  https://vizier.cds.unistra.fr/viz-bin/VizieR?-source=J/A+A/682/A145"
        )

    # ---- Also save CSV for quick inspection --------------------------------
    params_csv = urllib.parse.urlencode({
        "REQUEST": "doQuery",
        "LANG":    "ADQL",
        "FORMAT":  "csv",
        "QUERY":   adql,
    })
    csv_url = f"https://tapvizier.cds.unistra.fr/TAPVizieR/tap/sync?{params_csv}"
    print(f"\n[2/3] Saving CSV copy …")
    _download(csv_url, csv_path, "catalogue (csv)")

    print(f"  ✓  Catalogue saved → {fits_path}")
    return fits_path


def download_spectra(outdir: str) -> str | None:
    """
    Download the GBS v3 spectral library FITS.
    Returns path if successful, None if the file is unavailable.
    """
    spec_dir  = os.path.join(outdir, "spectra")
    gz_path   = os.path.join(spec_dir, "gbs_v3_spectra.fits.gz")
    fits_path = os.path.join(spec_dir, "gbs_v3_spectra.fits")

    if os.path.exists(fits_path):
        print(f"  [SKIP] {fits_path} already exists")
        return fits_path

    os.makedirs(spec_dir, exist_ok=True)

    print(f"\n[3/3] Downloading GBS v3 spectral library …")
    ok = _download(SPECTRA_URL, gz_path, "spectra")
    if not ok:
        print(f"  [RETRY] Trying mirror …")
        ok = _download(SPECTRA_URL_MIRROR, gz_path, "spectra (mirror)")
    if not ok:
        print(
            "  [WARN] Spectral library FITS could not be downloaded.\n"
            "         The file may not yet be publicly available at CDS.\n"
            "         You can request it directly from:\n"
            "           https://www.aanda.org/articles/aa/full_html/2026/01/aa55211-25/\n"
            "         Skipping spectral download — parameter catalogue was saved."
        )
        return None

    # Decompress
    import gzip, shutil
    print("  Decompressing …", end=" ", flush=True)
    with gzip.open(gz_path, "rb") as f_in, open(fits_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    os.remove(gz_path)
    print("done")
    print(f"  ✓  Spectra saved → {fits_path}")
    return fits_path


def inspect_catalogue(fits_path: str) -> None:
    """Print a brief summary of what was downloaded."""
    try:
        from astropy.io import fits as afits
        import numpy as np
    except ImportError:
        print("\n  [INFO] Install astropy to inspect the catalogue: pip install astropy")
        return

    print(f"\n{'='*60}")
    print("  GBS v3 Catalogue — Quick Inspection")
    print(f"{'='*60}")
    with afits.open(fits_path) as hdul:
        hdul.info()
        for ext in hdul[1:]:
            if ext.data is not None:
                tbl = ext.data
                print(f"\n  Extension : {ext.name}  ({len(tbl)} rows)")
                print(f"  Columns   : {tbl.dtype.names}")

                # Print parameter ranges if recognised columns exist
                for col, unit in [("Teff", "K"), ("logg", "dex"), ("__Fe_H_", "dex")]:
                    if col in tbl.dtype.names:
                        vals = tbl[col]
                        finite = vals[np.isfinite(vals)]
                        print(f"  {col:12s}: {finite.min():.1f} – {finite.max():.1f} {unit}  "
                              f"(n={len(finite)})")
                break
    print(f"{'='*60}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download Gaia FGK Benchmark Stars v3 from VizieR/CDS"
    )
    parser.add_argument(
        "--outdir", default="data/gbs",
        help="Root output directory (default: data/gbs)"
    )
    parser.add_argument(
        "--no-spectra", action="store_true",
        help="Skip spectral library download (catalogue only)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Gaia FGK Benchmark Stars v3 — Downloader")
    print("  Soubiran+ (2024)  J/A+A/682/A145")
    print("=" * 60)

    fits_path = download_catalogue(args.outdir)

    if not args.no_spectra:
        download_spectra(args.outdir)

    inspect_catalogue(fits_path)

    print("Download complete.")
    print(f"Output root : {os.path.abspath(args.outdir)}/")


if __name__ == "__main__":
    main()