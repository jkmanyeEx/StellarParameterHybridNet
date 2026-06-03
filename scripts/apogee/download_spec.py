"""
APOGEE DR17 spectrum downloader.

Downloads aspcapStar FITS files from the SDSS Science Archive Server (SAS):
  https://data.sdss.org/sas/dr17/apogee/spectro/aspcap/dr17/synspec_rev1/
    {telescope}/{field}/aspcapStar-dr17-{apogee_id}.fits

Each file is a single FITS containing the full APOGEE spectrum.
The pipeline uses HDU[1] (normalised flux, 3 chips combined).

Metadata is queried from SDSS SkyServer DR17 via SQL:
  https://skyserver.sdss.org/dr17/SkyServerWS/SearchTools/SqlSearch

Quality filters applied:
  - ASPCAPFLAG = 0       : no major analysis issues
  - SNR > 100            : sufficient signal-to-noise
  - Teff 3500-7000 K, logg 0-5, [Fe/H] -2.5 to 0.5

FETCH_LIMIT set to 50,000 to match GALAH download target and exceed
StarNet training set size (41,000 stars).

Existing valid files are always skipped (idempotent re-runs).
"""

import os
import sys
import csv
import requests
import concurrent.futures
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

BASE_DIR     = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
RAW_DIR      = os.path.join(BASE_DIR, "data", "apogee", "raw")
OUT_DIR      = os.path.join(RAW_DIR,  "spectra")
CSV_PATH     = os.path.join(RAW_DIR,  "allStar-dr17.csv")

SAS_BASE     = ("https://data.sdss.org/sas/dr17/apogee/spectro/aspcap"
                "/dr17/synspec_rev1")
SKYSERVER_URL = ("https://skyserver.sdss.org/dr17/SkyServerWS"
                 "/SearchTools/SqlSearch")

FETCH_LIMIT   = 50_000
MAX_WORKERS   = 16
MIN_FITS_SIZE = 100_000   # bytes — aspcapStar files are typically 300 KB+


# ── Metadata ──────────────────────────────────────────────────────────────────

def fetch_metadata():
    """
    Query SDSS SkyServer for the top FETCH_LIMIT APOGEE DR17 stars
    ordered by SNR.

    Quality filters:
      ASPCAPFLAG = 0  : clean spectroscopic analysis
      SNR > 100       : high signal-to-noise
      Teff / logg / [Fe/H] within physical parameter bounds
    """
    print(f"[Download] Querying SDSS SkyServer — requesting up to {FETCH_LIMIT} stars...")
    print( "[Download] Quality filter: ASPCAPFLAG=0, SNR>100")

    query = f"""
    SELECT TOP {FETCH_LIMIT}
        a.apogee_id, s.telescope, s.field,
        a.teff, a.logg, a.fe_h
    FROM aspcapStar AS a
    JOIN apogeeStar AS s ON a.apogee_id = s.apogee_id
    WHERE a.aspcapflag  =  0
      AND s.snr         > 100
      AND a.teff   BETWEEN 3500 AND 7000
      AND a.logg   BETWEEN 0.0  AND 5.0
      AND a.fe_h   BETWEEN -2.5 AND 0.5
    ORDER BY s.snr DESC
    """

    r = requests.get(SKYSERVER_URL,
                     params={"cmd": query, "format": "csv"},
                     timeout=120)
    if r.status_code != 200:
        raise RuntimeError(
            f"SDSS SkyServer query failed: HTTP {r.status_code}\n{r.text[:300]}"
        )

    # SkyServer prepends "#Table1" comment lines — strip them
    lines       = r.text.splitlines()
    clean_lines = [l for l in lines if not l.startswith("#")]

    os.makedirs(RAW_DIR, exist_ok=True)
    with open(CSV_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(clean_lines))

    n = len(clean_lines) - 1   # subtract header
    print(f"   [Download] {n} stars written to: {CSV_PATH}")
    return n


def read_stars_from_csv():
    stars = []
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            apogee_id = row.get("apogee_id", "").strip()
            telescope = row.get("telescope", "").strip()
            field     = row.get("field",     "").strip()
            if apogee_id and telescope and field:
                stars.append({
                    "apogee_id": apogee_id,
                    "telescope": telescope,
                    "field":     field,
                })
    return stars


# ── Per-star download ─────────────────────────────────────────────────────────

def download_star_spectrum(star):
    """
    Download one aspcapStar FITS file from SDSS SAS.

    Skip logic (idempotent):
      - File exists AND size >= MIN_FITS_SIZE AND starts with FITS magic bytes
        -> return (apogee_id, True, "skipped")
      - Otherwise attempt download.
    """
    apogee_id = star["apogee_id"]
    telescope = star["telescope"]
    field     = star["field"]

    filename  = f"{apogee_id}.fits"
    dest      = os.path.join(OUT_DIR, filename)

    # ── Skip if already a valid FITS file ────────────────────────────────────
    if os.path.exists(dest) and os.path.getsize(dest) >= MIN_FITS_SIZE:
        try:
            with open(dest, "rb") as fh:
                magic = fh.read(8)
            if magic.startswith(b"SIMPLE  "):
                return apogee_id, True, "skipped"
        except OSError:
            pass   # fall through to re-download

    url = (f"{SAS_BASE}/{telescope}/{field}"
           f"/aspcapStar-dr17-{apogee_id}.fits")
    try:
        r = requests.get(url, timeout=60)

        if r.status_code != 200:
            return apogee_id, False, f"HTTP {r.status_code}"

        if len(r.content) < MIN_FITS_SIZE:
            return apogee_id, False, (
                f"response too small ({len(r.content)} B) "
                "— file may not exist at this path"
            )

        if not r.content.startswith(b"SIMPLE  "):
            return apogee_id, False, "response is not a FITS file (bad magic bytes)"

        with open(dest, "wb") as fh:
            fh.write(r.content)
        return apogee_id, True, "downloaded"

    except requests.Timeout:
        return apogee_id, False, "timeout"
    except Exception as e:
        return apogee_id, False, str(e)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Count already-downloaded valid files
    existing = sum(
        1 for f in os.listdir(OUT_DIR)
        if f.endswith(".fits")
        and os.path.getsize(os.path.join(OUT_DIR, f)) >= MIN_FITS_SIZE
    )
    print(f"[Download] Valid FITS files already present: {existing} "
          f"— these will be skipped.")

    # Refresh metadata CSV if it has fewer entries than FETCH_LIMIT
    existing_csv_lines = 0
    if os.path.exists(CSV_PATH):
        with open(CSV_PATH, encoding="utf-8") as f:
            existing_csv_lines = sum(1 for _ in f) - 1

    if existing_csv_lines < FETCH_LIMIT * 0.9:
        print(f"[Download] CSV has {existing_csv_lines} entries "
              f"(target {FETCH_LIMIT}) — refreshing metadata...")
        try:
            fetch_metadata()
        except Exception as e:
            raise RuntimeError(f"Metadata fetch failed: {e}")
    else:
        print(f"[Download] CSV already has {existing_csv_lines} entries "
              "— skipping re-fetch.")

    stars = read_stars_from_csv()
    n     = len(stars)
    print(f"\n[Download] Target  : {n} stars")
    print(f"[Download] Output  : {OUT_DIR}")
    print(f"[Download] Workers : {MAX_WORKERS}")
    print(f"[Download] Already present files will be skipped automatically.\n")

    downloaded, skipped, failed = 0, 0, 0
    failed_list = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(download_star_spectrum, star): star
                   for star in stars}
        with tqdm(total=n, desc="APOGEE DR17 spectra", unit="star") as pbar:
            for fut in concurrent.futures.as_completed(futures):
                apogee_id, ok, reason = fut.result()
                if ok and reason == "downloaded":
                    downloaded += 1
                elif ok and reason == "skipped":
                    skipped += 1
                else:
                    failed += 1
                    failed_list.append((apogee_id, reason))
                pbar.update(1)

    print(f"\n{'='*65}")
    print("  APOGEE DR17 Download Summary")
    print(f"{'='*65}")
    print(f"  Stars in catalog            : {n}")
    print(f"  Files newly downloaded      : {downloaded}")
    print(f"  Files already present       : {skipped}")
    print(f"  Files failed                : {failed}")
    print(f"  Output directory            : {OUT_DIR}")
    print(f"{'='*65}")

    if failed > 0:
        print(f"\n  [NOTE] {failed} file(s) failed. "
              "Re-running this script will retry only the missing files.")
        print(f"\n  First {min(10, len(failed_list))} failures:")
        for apogee_id, reason in failed_list[:10]:
            print(f"    apogee_id={apogee_id}  reason={reason}")


if __name__ == "__main__":
    main()
