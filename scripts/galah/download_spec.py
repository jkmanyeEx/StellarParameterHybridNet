"""
GALAH DR4 spectrum downloader.

Downloads 4-arm reduced spectra from AAO DataCentral using the official
slink API:
  https://datacentral.org.au/vo/slink/links
    ?ID={sobject_id}&DR=galah_dr4&IDX=0&FILT={B|G|R|I}&RESPONSEFORMAT=fits

CCD mapping:
  B (Blue)  -> suffix 1 -> ~4713-4903 A
  G (Green) -> suffix 2 -> ~5648-5873 A
  R (Red)   -> suffix 3 -> ~6478-6737 A
  I (NIR)   -> suffix 4 -> ~7585-7887 A

Saved filename: {sobject_id}{ccd_number}.fits  (e.g. 1712070036012021.fits)

FITS layout (DataCentral SSA single-HDU format):
  HDU[0] : normalised flux  (CRVAL1 + PC1_1 wavelength solution)

Quality filter changes vs. previous version:
  - flag_fe_h filter REMOVED (DR4 best practices: flag_fe_h is not
    meaningful in DR4 due to a calculation error — galah-survey.org/dr4/using_the_data)
  - flag_sp = 0 retained
  - snr_px_ccd3 > 30 retained (CCD3 / Red arm is the primary quality indicator)
  - FETCH_LIMIT raised to 50,000 to expand training coverage
  - Existing valid files are always skipped (idempotent re-runs)
"""

import os
import sys
import csv
import requests
import concurrent.futures
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

BASE_DIR  = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
RAW_DIR   = os.path.join(BASE_DIR, "data", "galah", "raw")
OUT_DIR   = os.path.join(RAW_DIR, "spectra")
CSV_PATH  = os.path.join(RAW_DIR, "galah_dr4_allstar.csv")

TAP_URL   = "https://datacentral.org.au/vo/tap/sync"
SLINK_URL = "https://datacentral.org.au/vo/slink/links"

ARM_TO_CCD   = {"B": "1", "G": "2", "R": "3", "I": "4"}
FETCH_LIMIT  = 50_000
MAX_WORKERS  = 8
MIN_FITS_SIZE = 20_000   # bytes — real FITS ~40 KB+, VOTable stubs ~2 KB


# ── Metadata ──────────────────────────────────────────────────────────────────

def fetch_metadata():
    """
    Query DataCentral TAP for GALAH DR4 stars.

    Quality filters (per galah-survey.org/dr4/using_the_data best practices):
      - flag_sp   = 0     : spectroscopic quality flag
      - flag_fe_h removed : not meaningful in DR4 (calculation error in catalog)
      - snr_px_ccd3 > 30  : CCD3 (Red arm) SNR is the primary quality indicator
      - Teff 4000–7000 K, logg 1–5, [Fe/H] –3 to 0.5
    """
    print(f"[Download] Querying DataCentral TAP — requesting up to {FETCH_LIMIT} stars...")
    print( "[Download] Quality filter: flag_sp=0, snr_px_ccd3>30 "
           "(flag_fe_h removed per DR4 best practices)")

    query = f"""
    SELECT TOP {FETCH_LIMIT}
        sobject_id, teff, logg, fe_h,
        snr_px_ccd1, snr_px_ccd2, snr_px_ccd3, snr_px_ccd4
    FROM galah_dr4.mainstartable
    WHERE flag_sp      =  0
      AND snr_px_ccd3  >  30
      AND teff  BETWEEN 4000 AND 7000
      AND logg  BETWEEN 1.0  AND 5.0
      AND fe_h  BETWEEN -3.0 AND 0.5
    ORDER BY snr_px_ccd3 DESC
    """

    r = requests.post(
        TAP_URL,
        data={"REQUEST": "doQuery", "LANG": "ADQL",
              "FORMAT": "csv", "QUERY": query},
        timeout=180,
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
    return n


def read_stars_from_csv():
    stars = []
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sid = row.get("sobject_id", "").strip()
            if sid:
                stars.append(sid)
    return stars


# ── Per-arm download ──────────────────────────────────────────────────────────

def download_single_arm(sobject_id, arm_letter):
    """
    Download one CCD arm FITS file.

    Skip logic (idempotent):
      - File exists AND size >= MIN_FITS_SIZE AND starts with FITS magic bytes
        → return (True, "skipped")
      - Otherwise attempt download.

    DR parameter: galah_dr4  (confirmed working; galah_DR4 was incorrect)
    """
    ccd_num  = ARM_TO_CCD[arm_letter]
    filename = f"{sobject_id}{ccd_num}.fits"
    dest     = os.path.join(OUT_DIR, filename)

    # ── Skip if already a valid FITS file ────────────────────────────────────
    if os.path.exists(dest) and os.path.getsize(dest) >= MIN_FITS_SIZE:
        try:
            with open(dest, "rb") as fh:
                magic = fh.read(8)
            if magic.startswith(b"SIMPLE  "):
                return True, "skipped"
        except OSError:
            pass   # fall through to re-download

    params = {
        "ID":             sobject_id,
        "DR":             "galah_dr4",
        "IDX":            "0",
        "FILT":           arm_letter,
        "RESPONSEFORMAT": "fits",
    }
    try:
        r = requests.get(SLINK_URL, params=params, timeout=60)

        if r.status_code != 200:
            return False, f"HTTP {r.status_code}"

        if len(r.content) < MIN_FITS_SIZE:
            return False, (f"response too small ({len(r.content)} B) "
                           "— likely VOTable stub")

        if not r.content.startswith(b"SIMPLE  "):
            return False, "response is not a FITS file (bad magic bytes)"

        with open(dest, "wb") as fh:
            fh.write(r.content)
        return True, "downloaded"

    except requests.Timeout:
        return False, "timeout"
    except Exception as e:
        return False, str(e)


def download_star(sobject_id):
    return sobject_id, {
        arm: download_single_arm(sobject_id, arm)
        for arm in ["B", "G", "R", "I"]
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Remove VOTable stubs from any previous failed run
    stubs = [
        f for f in os.listdir(OUT_DIR)
        if f.endswith(".fits")
        and os.path.getsize(os.path.join(OUT_DIR, f)) < MIN_FITS_SIZE
    ]
    if stubs:
        print(f"[Download] Removing {len(stubs)} stub file(s) from previous run...")
        for s in stubs:
            os.remove(os.path.join(OUT_DIR, s))

    # Count already-downloaded complete files (any run, including current data)
    existing = sum(
        1 for f in os.listdir(OUT_DIR)
        if f.endswith(".fits")
        and os.path.getsize(os.path.join(OUT_DIR, f)) >= MIN_FITS_SIZE
    )
    print(f"[Download] Valid FITS files already present: {existing} arm files "
          f"(~{existing // 4} complete stars) — these will be skipped.")

    # Fetch or refresh metadata CSV
    # Always re-fetch when FETCH_LIMIT changed to 50,000
    # so the CSV reflects the larger target set.
    existing_csv_lines = 0
    if os.path.exists(CSV_PATH):
        with open(CSV_PATH, encoding="utf-8") as f:
            existing_csv_lines = sum(1 for _ in f) - 1   # subtract header

    if existing_csv_lines < FETCH_LIMIT * 0.9:
        print(f"[Download] CSV has {existing_csv_lines} entries "
              f"(target {FETCH_LIMIT}) — refreshing metadata...")
        try:
            fetch_metadata()
        except Exception as e:
            raise RuntimeError(f"Metadata fetch failed: {e}")
    else:
        print(f"[Download] CSV already has {existing_csv_lines} entries — skipping re-fetch.")

    stars = read_stars_from_csv()
    n     = len(stars)
    print(f"\n[Download] Target  : {n} stars ({n * 4} arm files total)")
    print(f"[Download] Output  : {OUT_DIR}")
    print(f"[Download] Workers : {MAX_WORKERS}")
    print(f"[Download] Already present arm files will be skipped automatically.\n")

    downloaded, skipped, failed = 0, 0, 0
    failed_list = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(download_star, sid): sid for sid in stars}
        with tqdm(total=n, desc="GALAH DR4 spectra", unit="star") as pbar:
            for fut in concurrent.futures.as_completed(futures):
                sid, results = fut.result()
                for arm, (ok, reason) in results.items():
                    if ok and reason == "downloaded":
                        downloaded += 1
                    elif ok and reason == "skipped":
                        skipped += 1
                    else:
                        failed += 1
                        failed_list.append((sid, arm, reason))
                pbar.update(1)

    complete_stars = (downloaded + skipped) // 4

    print(f"\n{'='*65}")
    print("  GALAH DR4 Download Summary")
    print(f"{'='*65}")
    print(f"  Stars in catalog            : {n}")
    print(f"  Arm files newly downloaded  : {downloaded}")
    print(f"  Arm files already present   : {skipped}")
    print(f"  Arm files failed            : {failed}")
    print(f"  Stars with all 4 arms ~     : {complete_stars}")
    print(f"  Output directory            : {OUT_DIR}")
    print(f"{'='*65}")

    if failed > 0:
        print(f"\n  [NOTE] {failed} arm file(s) failed. Re-running this script "
              "will retry only the missing files.")
        # Show first 10 failures for diagnosis
        print(f"\n  First {min(10, len(failed_list))} failures:")
        for sid, arm, reason in failed_list[:10]:
            print(f"    sobject_id={sid}  arm={arm}  reason={reason}")


if __name__ == "__main__":
    main()
