"""
GALAH DR4 spectrum downloader.

Downloads 4-arm reduced spectra from AAO DataCentral using the official
slink API documented at https://www.galah-survey.org/dr4/the_spectra/

URL pattern (confirmed from official docs):
  https://datacentral.org.au/vo/slink/links
    ?ID={sobject_id}&DR=galah_dr3&IDX=0&FILT={B|G|R|I}&RESPONSEFORMAT=fits

Saved filename convention used by this pipeline:
  {sobject_id}{ccd_number}.fits   (e.g. 1401130047013951.fits for CCD1/Blue)

CCD mapping:
  B (Blue)  -> CCD1 -> suffix 1 -> wavelength ~4713-4903 A
  G (Green) -> CCD2 -> suffix 2 -> wavelength ~5648-5873 A
  R (Red)   -> CCD3 -> suffix 3 -> wavelength ~6478-6737 A
  I (NIR)   -> CCD4 -> suffix 4 -> wavelength ~7585-7887 A

FITS extensions inside each file:
  [0] Primary     : raw flux (sky-subtracted, not normalised)
  [1] normalized  : normalised flux  <-- used by preprocess_flux.py
  [2] relative_error
  ...

Metadata (sobject_id + stellar parameters) is fetched via DataCentral TAP
with quality flags:  flag_sp = 0, flag_fe_h = 0, SNR > 30 on all CCDs.
"""

import os
import sys
import csv
import requests
import concurrent.futures
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
RAW_DIR  = os.path.join(BASE_DIR, "data", "galah", "raw")
OUT_DIR  = os.path.join(RAW_DIR, "spectra")
CSV_PATH = os.path.join(RAW_DIR, "galah_dr4_allstar.csv")

TAP_URL  = "https://datacentral.org.au/vo/tap/sync"
SLINK_URL = "https://datacentral.org.au/vo/slink/links"

# Arm letter -> CCD number suffix in saved filename
ARM_TO_CCD = {"B": "1", "G": "2", "R": "3", "I": "4"}

FETCH_LIMIT  = 15000
MAX_WORKERS  = 8
MIN_FITS_SIZE = 20_000   # bytes; VOTable stubs are ~1-2 KB, real FITS can be as small as ~40 KB


# ── Metadata ──────────────────────────────────────────────────────────────────

def fetch_metadata():
    """
    Query DataCentral TAP for quality-filtered GALAH DR4 stars and write CSV.
    Filters applied: flag_sp=0, flag_fe_h=0, SNR>30 on all 4 CCDs,
                     Teff 4000-7000 K, logg 1-5, [Fe/H] -3 to 0.5.
    """
    print("[Download] Querying DataCentral TAP for GALAH DR4 metadata...")
    query = f"""
    SELECT TOP {FETCH_LIMIT}
        sobject_id, teff, logg, fe_h,
        snr_px_ccd1, snr_px_ccd2, snr_px_ccd3, snr_px_ccd4
    FROM galah_dr4.mainstartable
    WHERE snr_px_ccd1 BETWEEN 30 AND 250
      AND snr_px_ccd2 BETWEEN 30 AND 250
      AND snr_px_ccd3 BETWEEN 30 AND 250
      AND snr_px_ccd4 BETWEEN 30 AND 250
      AND flag_sp   = 0
      AND flag_fe_h = 0
      AND teff  BETWEEN 4000 AND 7000
      AND logg  BETWEEN 1.0  AND 5.0
      AND fe_h  BETWEEN -3.0 AND 0.5
    ORDER BY snr_px_ccd1 DESC
    """
    r = requests.post(
        TAP_URL,
        data={"REQUEST": "doQuery", "LANG": "ADQL",
              "FORMAT": "csv", "QUERY": query},
        timeout=120,
    )
    if r.status_code != 200:
        raise RuntimeError(
            f"TAP query failed: HTTP {r.status_code}\n{r.text[:300]}"
        )
    os.makedirs(RAW_DIR, exist_ok=True)
    with open(CSV_PATH, "w", encoding="utf-8") as f:
        f.write(r.text)
    n = len(r.text.strip().splitlines()) - 1
    print(f"   [Download] {n} stars written to: {CSV_PATH}")


def read_stars_from_csv():
    stars = []
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            stars.append(row["sobject_id"].strip())
    return stars


# ── Per-arm download ───────────────────────────────────────────────────────────

def download_single_arm(sobject_id, arm_letter):
    """
    Download one CCD arm and save as {sobject_id}{ccd_num}.fits.

    Uses the official DataCentral slink endpoint:
      GET /vo/slink/links?ID=<sobject_id>&DR=galah_dr3&IDX=0&FILT=<arm>&RESPONSEFORMAT=fits

    Returns (ok: bool, reason: str).
    """
    ccd_num  = ARM_TO_CCD[arm_letter]
    filename = f"{sobject_id}{ccd_num}.fits"
    dest     = os.path.join(OUT_DIR, filename)

    if os.path.exists(dest) and os.path.getsize(dest) >= MIN_FITS_SIZE:
        return True, "skipped"

    params = {
        "ID":             sobject_id,
        "DR":             "galah_dr4",   # Query the official GALAH DR4 spectrum archive
        "IDX":            "0",
        "FILT":           arm_letter,
        "RESPONSEFORMAT": "fits",
    }
    try:
        r = requests.get(SLINK_URL, params=params, timeout=60)
        if r.status_code != 200:
            return False, f"HTTP {r.status_code}"
        if len(r.content) < MIN_FITS_SIZE:
            # DataCentral returned a VOTable stub or error page instead of FITS
            return False, f"response too small ({len(r.content)} bytes, likely VOTable stub)"
        if not r.content.startswith(b"SIMPLE  ="):
            return False, "Not a valid FITS file (starts with non-FITS header signature)"
        with open(dest, "wb") as f:
            f.write(r.content)
        return True, "downloaded"
    except Exception as e:
        return False, str(e)


def download_star(sobject_id):
    return sobject_id, {arm: download_single_arm(sobject_id, arm)
                        for arm in ["B", "G", "R", "I"]}


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    # Remove any VOTable stub files left from previous download attempts
    if os.path.isdir(OUT_DIR):
        stubs = [
            f for f in os.listdir(OUT_DIR)
            if f.endswith(".fits")
            and os.path.getsize(os.path.join(OUT_DIR, f)) < MIN_FITS_SIZE
        ]
        if stubs:
            print(f"[Download] Removing {len(stubs)} stub/incomplete file(s) "
                  f"from previous run...")
            for s in stubs:
                os.remove(os.path.join(OUT_DIR, s))

    if not os.path.exists(CSV_PATH):
        try:
            fetch_metadata()
        except Exception as e:
            raise RuntimeError(f"Metadata fetch failed: {e}")

    stars = read_stars_from_csv()
    n = len(stars)
    print(f"\n[Download] Target  : {n} stars ({n * 4} arm files)")
    print(f"[Download] Output  : {OUT_DIR}")
    print(f"[Download] Workers : {MAX_WORKERS}\n")
    os.makedirs(OUT_DIR, exist_ok=True)

    downloaded, skipped, failed = 0, 0, 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(download_star, sid): sid for sid in stars}
        with tqdm(total=n, desc="Downloading GALAH DR4 spectra", unit="star") as pbar:
            for fut in concurrent.futures.as_completed(futures):
                sid, results = fut.result()
                for arm, (ok, reason) in results.items():
                    if ok and reason == "downloaded":
                        downloaded += 1
                    elif ok and reason == "skipped":
                        skipped += 1
                    else:
                        failed += 1
                pbar.update(1)

    print(f"\n{'='*60}")
    print("  GALAH DR4 Download Summary")
    print(f"{'='*60}")
    print(f"  Stars targeted              : {n}")
    print(f"  Arm files downloaded        : {downloaded}")
    print(f"  Arm files already present   : {skipped}")
    print(f"  Arm files failed            : {failed}")
    print(f"  Stars with all 4 arms ~     : {(downloaded + skipped) // 4}")
    print(f"  Output directory            : {OUT_DIR}")
    print(f"{'='*60}")
    if failed > 0:
        print(
            f"\n  [NOTE] {failed} arm file(s) failed. "
            "Re-running this script will retry only the missing files."
        )


if __name__ == "__main__":
    main()
