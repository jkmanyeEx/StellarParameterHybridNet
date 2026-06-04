"""
download_gbs.py
===============
Download the Gaia FGK Benchmark Stars v3 (Soubiran+ 2024) from VizieR/CDS.

What this downloads
-------------------
1. GBS v3 parameter catalogue  (J/A+A/682/A145) — Teff, logg, [Fe/H], HIP ID
2. GBS v3 spectral library — individual R=42 000 normalised FITS spectra
   from the Blanco-Cuaresma benchmark-stars website (480–680 nm).

Output layout
-------------
data/gbs/
├── gbs_v3_params.fits          ← parameter catalogue (202 stars)
├── gbs_v3_params.csv           ← same, CSV convenience copy
└── spectra/
    ├── HIP101345_HARPS_1_R42KNorm.fits
    ├── HIP101345_NARVAL_1_R42KNorm.fits
    ├── ...                     ← one file per star × instrument observation
    └── manifest.csv            ← download manifest with star/instrument info

Usage
-----
    python download_gbs.py [--outdir data/gbs] [--no-spectra] [--resolution R42KNorm]
"""

import argparse
import csv
import gzip
import os
import re
import shutil
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

# ── Constants ────────────────────────────────────────────────────────────────

CATALOGUE_TABLE = "J/A+A/682/A145/catalog"

# GBS spectral library — hosted as individual files
GBS_LIBRARY_BASE = (
    "https://blancocuaresma.com/s/repository/"
    "gaia_benchmark_stars_library/2025/all/"
)
GBS_INDEX_PAGE = "https://blancocuaresma.com/s/benchmarkstars"

# Available resolutions (filename suffixes before .fits.gz):
#   Rmax       — original instrument resolution
#   RmaxNorm   — original resolution, continuum-normalised
#   R42K       — degraded to R = 42 000
#   R42KNorm   — degraded to R = 42 000, continuum-normalised  ← DEFAULT
DEFAULT_RESOLUTION = "R42KNorm"

CHUNK = 1 << 16   # 64 kB download chunks


# ── Helpers ──────────────────────────────────────────────────────────────────

def _progress(label: str, done: int, total: int | None) -> None:
    if total:
        pct = 100 * done // total
        bar = "#" * (pct // 5) + "-" * (20 - pct // 5)
        print(f"\r  {label}  [{bar}] {pct:3d}%  ({done/1e6:.1f}/{total/1e6:.1f} MB)",
              end="", flush=True)
    else:
        print(f"\r  {label}  {done/1e6:.1f} MB downloaded", end="", flush=True)


def _ssl_context():
    """Create an SSL context that skips certificate verification.
    Needed because macOS Python often lacks the CDS/university root CAs."""
    ctx = ssl._create_unverified_context()
    return ctx


def _download(url: str, dest: str, label: str, context=None) -> bool:
    """Download *url* to *dest*; return True on success."""
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "GBS-downloader/1.0"})
        try:
            resp = urllib.request.urlopen(req, timeout=120, context=context)
        except urllib.error.URLError as exc:
            err_str = str(exc)
            if "CERTIFICATE_VERIFY_FAILED" in err_str or "certificate verify failed" in err_str:
                resp = urllib.request.urlopen(req, timeout=120, context=_ssl_context())
            else:
                raise

        with resp:
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
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
        print(f"\n  [WARN] {url} failed: {exc}")
        return False


# ── Catalogue download ───────────────────────────────────────────────────────

def download_catalogue(outdir: str) -> str:
    """
    Query VizieR TAP for the GBS v3 parameter table and save as FITS + CSV.
    Returns the path to the saved FITS file.
    """
    fits_path = os.path.join(outdir, "gbs_v3_params.fits")
    csv_path  = os.path.join(outdir, "gbs_v3_params.csv")

    if os.path.exists(fits_path):
        print(f"  [SKIP] {fits_path} already exists")
        return fits_path

    os.makedirs(outdir, exist_ok=True)

    # ---- ADQL query via VizieR TAP ----------------------------------------
    adql = f'SELECT * FROM "{CATALOGUE_TABLE}"'
    params = urllib.parse.urlencode({
        "REQUEST": "doQuery",
        "LANG":    "ADQL",
        "FORMAT":  "fits",
        "QUERY":   adql,
    })
    tap_url = f"https://tapvizier.cds.unistra.fr/TAPVizieR/tap/sync?{params}"

    print(f"\n[1/3] Downloading GBS v3 parameter catalogue from VizieR TAP …")
    print(f"      Table : {CATALOGUE_TABLE}")

    ok = _download(tap_url, fits_path, "catalogue")
    if not ok:
        # Fallback: direct URL from CDS FTP
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


# ── Spectral library download ────────────────────────────────────────────────

def _scrape_spectrum_urls(resolution: str = DEFAULT_RESOLUTION) -> list[dict]:
    """
    Parse the GBS benchmark-stars website to extract individual spectrum
    download URLs matching the requested resolution.

    Returns a list of dicts:
        {"filename": str, "url": str, "hip": str, "instrument": str}
    """
    print(f"  Fetching GBS index page …")
    ctx = _ssl_context()
    req = urllib.request.Request(GBS_INDEX_PAGE,
                                headers={"User-Agent": "GBS-downloader/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=120, context=ctx) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except Exception as exc:
        print(f"  [ERROR] Could not fetch index page: {exc}")
        return []

    # Extract all links matching the pattern:
    #   repository/gaia_benchmark_stars_library/2025/all/<name>_<resolution>.fits.gz
    # Only match observed spectra (HIP*_INSTRUMENT_N_Resolution.fits.gz),
    # not synthetic spectra (which contain "synthetic" in the name).
    pattern = re.compile(
        r'href="((?:repository/)?gaia_benchmark_stars_library/2025/all/'
        r'(HIP\d+_[A-Z]+_\d+_' + re.escape(resolution) + r')\.fits\.gz)"',
        re.IGNORECASE
    )

    seen = set()
    results = []
    for match in pattern.finditer(html):
        rel_path = match.group(1)
        basename = match.group(2)

        if basename in seen:
            continue
        seen.add(basename)

        # Build absolute URL
        if rel_path.startswith("http"):
            url = rel_path
        else:
            url = f"https://blancocuaresma.com/s/{rel_path}"

        # Parse HIP ID and instrument from the filename
        parts = basename.split("_")
        hip = parts[0] if parts else ""
        instrument = parts[1] if len(parts) > 1 else ""

        results.append({
            "filename": f"{basename}.fits.gz",
            "url": url,
            "hip": hip,
            "instrument": instrument,
        })

    print(f"  Found {len(results)} {resolution} spectra on the GBS website")
    return results


def download_spectra(outdir: str, resolution: str = DEFAULT_RESOLUTION) -> str | None:
    """
    Download GBS v3 individual spectra from the Blanco-Cuaresma website.
    Returns the spectra directory path if successful, None otherwise.
    """
    spec_dir = os.path.join(outdir, "spectra")
    manifest_path = os.path.join(spec_dir, "manifest.csv")

    # If manifest already exists with entries, check for completeness
    existing = set()
    if os.path.exists(manifest_path):
        with open(manifest_path, newline="") as fh:
            for row in csv.DictReader(fh):
                fits_file = os.path.join(spec_dir, row.get("fits_file", ""))
                if os.path.exists(fits_file):
                    existing.add(row.get("filename", ""))

    print(f"\n[3/3] Downloading GBS v3 spectral library ({resolution}) …")

    spectra = _scrape_spectrum_urls(resolution)
    if not spectra:
        print("  [WARN] No spectra URLs found. The website may have changed.")
        print("         Visit: https://blancocuaresma.com/s/benchmarkstars")
        return None

    os.makedirs(spec_dir, exist_ok=True)

    # Use an unverified SSL context for the download server
    ctx = _ssl_context()

    downloaded, skipped, failed = 0, 0, 0
    records = []

    for i, spec in enumerate(spectra, 1):
        gz_name  = spec["filename"]
        fits_name = gz_name.replace(".fits.gz", ".fits")
        fits_path = os.path.join(spec_dir, fits_name)

        # Skip if already downloaded (either from manifest or filesystem)
        if gz_name in existing or os.path.exists(fits_path):
            skipped += 1
            records.append({
                "filename": gz_name,
                "fits_file": fits_name,
                "hip": spec["hip"],
                "instrument": spec["instrument"],
                "url": spec["url"],
                "status": "exists",
            })
            continue

        label = f"[{i}/{len(spectra)}] {fits_name}"
        gz_path = os.path.join(spec_dir, gz_name)

        ok = _download(spec["url"], gz_path, label, context=ctx)
        if ok:
            # Decompress .fits.gz → .fits
            try:
                with gzip.open(gz_path, "rb") as f_in, open(fits_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
                os.remove(gz_path)
                downloaded += 1
                records.append({
                    "filename": gz_name,
                    "fits_file": fits_name,
                    "hip": spec["hip"],
                    "instrument": spec["instrument"],
                    "url": spec["url"],
                    "status": "downloaded",
                })
            except Exception as exc:
                print(f"  [WARN] Decompression failed for {gz_name}: {exc}")
                if os.path.exists(gz_path):
                    os.remove(gz_path)
                failed += 1
                records.append({
                    "filename": gz_name,
                    "fits_file": fits_name,
                    "hip": spec["hip"],
                    "instrument": spec["instrument"],
                    "url": spec["url"],
                    "status": f"decompress_error: {exc}",
                })
        else:
            failed += 1
            records.append({
                "filename": gz_name,
                "fits_file": fits_name,
                "hip": spec["hip"],
                "instrument": spec["instrument"],
                "url": spec["url"],
                "status": "download_failed",
            })

        # Small delay to be polite to the server
        if downloaded % 10 == 0 and downloaded > 0:
            time.sleep(0.5)

    # Write manifest
    if records:
        with open(manifest_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)

    # Summary
    unique_stars = len(set(r["hip"] for r in records if r["status"] != "download_failed"))
    print(f"\n  ────────────────────────────────────────────")
    print(f"  Spectral library download summary:")
    print(f"    Total files found : {len(spectra)}")
    print(f"    Downloaded (new)  : {downloaded}")
    print(f"    Skipped (exists)  : {skipped}")
    print(f"    Failed            : {failed}")
    print(f"    Unique stars      : {unique_stars}")
    print(f"    Resolution        : {resolution}")
    print(f"    Manifest          : {manifest_path}")
    print(f"  ────────────────────────────────────────────")

    if downloaded + skipped > 0:
        print(f"  ✓  Spectra saved → {spec_dir}/")
        return spec_dir
    else:
        print("  [WARN] No spectra were downloaded successfully.")
        return None


# ── Catalogue inspection ─────────────────────────────────────────────────────

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


# ── Entry point ──────────────────────────────────────────────────────────────

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
    parser.add_argument(
        "--resolution", default=DEFAULT_RESOLUTION,
        choices=["Rmax", "RmaxNorm", "R42K", "R42KNorm"],
        help="Spectral resolution variant to download (default: R42KNorm)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Gaia FGK Benchmark Stars v3 — Downloader")
    print("  Soubiran+ (2024)  J/A+A/682/A145")
    print("  Spectral library: Blanco-Cuaresma+ (2014, 2025)")
    print("=" * 60)

    fits_path = download_catalogue(args.outdir)

    if not args.no_spectra:
        download_spectra(args.outdir, resolution=args.resolution)

    inspect_catalogue(fits_path)

    print("Download complete.")
    print(f"Output root : {os.path.abspath(args.outdir)}/")


if __name__ == "__main__":
    main()