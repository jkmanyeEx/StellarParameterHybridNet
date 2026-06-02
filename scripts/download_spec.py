import os
import csv
import time
import requests

# Resolve paths relative to project root
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CSV_PATH  = os.path.join(BASE_DIR, "data", "validation_dataset", "Skyserver_SQL6_1_2026 10_51_26 PM.csv")
OUT_DIR   = os.path.join(BASE_DIR, "data", "validation_dataset")
BASE_URL  = "https://dr17.sdss.org/sas/dr17/sdss/spectro/redux/26/spectra"
ALT_URL   = "https://dr17.sdss.org/sas/dr17/sdss/spectro/redux/v5_13_2/spectra"
RETRY_MAX = 3
DELAY_SEC = 0.5


def build_filename(plate, mjd, fiberid):
    return f"spec-{plate}-{mjd}-{fiberid:04d}.fits"


def build_url(base, plate, fname):
    return f"{base}/{plate}/{fname}"


def download_file(url, dest_path, retries=RETRY_MAX):
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, timeout=60, stream=True)
            if r.status_code == 200:
                with open(dest_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 64):
                        f.write(chunk)
                return True
            elif r.status_code == 404:
                return False
            else:
                print(f"      HTTP {r.status_code} — attempt {attempt}/{retries}")
        except requests.RequestException as e:
            print(f"      Network error attempt {attempt}/{retries}: {e}")
        time.sleep(1.5 * attempt)
    return False


def read_csv(csv_path):
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        lines = [l for l in f if not l.startswith("#")]
    for row in csv.DictReader(lines):
        rows.append({
            "plate":   int(row["plate"]),
            "mjd":     int(row["mjd"]),
            "fiberid": int(row["fiberid"]),
            "teff":    float(row["teffadop"]),
            "logg":    float(row["loggadop"]),
            "feh":     float(row["fehadop"]),
        })
    return rows


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    if not os.path.exists(CSV_PATH):
        print(f"[Error] Catalog CSV not found at: {CSV_PATH}")
        return
        
    rows  = read_csv(CSV_PATH)
    total = len(rows)
    print(f"Found {total} entries in CSV.")
    print(f"Saving to: {os.path.abspath(OUT_DIR)}\n")

    ok_count   = 0
    skip_count = 0
    fail_count = 0

    for i, row in enumerate(rows, 1):
        plate, mjd, fiberid = row["plate"], row["mjd"], row["fiberid"]
        fname = build_filename(plate, mjd, fiberid)
        dest  = os.path.join(OUT_DIR, fname)

        if os.path.exists(dest) and os.path.getsize(dest) > 10_000:
            print(f"[{i:>3}/{total}] SKIP (exists)  {fname}")
            skip_count += 1
            continue

        if os.path.exists(dest):
            os.remove(dest)

        print(f"[{i:>3}/{total}] {fname}  "
              f"T={row['teff']:.0f}K  logg={row['logg']:.2f}  "
              f"[Fe/H]={row['feh']:.2f}")

        success = False
        for base in (BASE_URL, ALT_URL):
            url = build_url(base, plate, fname)
            if download_file(url, dest):
                size_kb = os.path.getsize(dest) / 1024
                print(f"      ✓  {size_kb:.0f} KB  ({url.split('/redux/')[1].split('/')[0]})")
                ok_count += 1
                success = True
                break
            else:
                print(f"      ✗  404 at {url.split('/redux/')[1].split('/')[0]}, trying next...")

        if not success:
            print(f"      ✗  All URLs failed — skipping {fname}")
            fail_count += 1

        time.sleep(DELAY_SEC)

    print("\n" + "=" * 55)
    print(f"Downloaded : {ok_count}")
    print(f"Skipped    : {skip_count}  (already existed)")
    print(f"Failed     : {fail_count}")
    print(f"Saved to   : {os.path.abspath(OUT_DIR)}")


if __name__ == "__main__":
    main()
