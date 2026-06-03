import os
import sys
import csv
import requests
import concurrent.futures
from tqdm import tqdm

# Add project root to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
RAW_DIR = os.path.join(BASE_DIR, "data", "apogee", "raw")
OUT_DIR = os.path.join(RAW_DIR, "spectra")
CSV_PATH = os.path.join(RAW_DIR, "allStar-dr17.csv")
MAX_WORKERS = 16

def fetch_metadata(limit=15000):
    print("🛰️  Querying SDSS SkyServer online for best APOGEE DR17 stars...")
    query = f"""
    SELECT TOP {limit}
        a.apogee_id, s.telescope, s.field, a.teff, a.logg, a.fe_h
    FROM aspcapStar AS a
    JOIN apogeeStar AS s ON a.apogee_id = s.apogee_id
    WHERE a.teff BETWEEN 3500 AND 7000
      AND a.logg BETWEEN 0.0 AND 5.0
      AND a.fe_h BETWEEN -2.5 AND 0.5
    ORDER BY s.snr DESC
    """
    url = "https://skyserver.sdss.org/dr17/SkyServerWS/SearchTools/SqlSearch"
    params = {
        "cmd": query,
        "format": "csv"
    }
    r = requests.get(url, params=params, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"Failed to query SDSS: HTTP {r.status_code}\n{r.text[:200]}")
    
    # SkyServer returns the CSV output wrapped in table structures occasionally,
    # but the API endpoint SkyServerWS/SearchTools/SqlSearch returns clean CSV with #Table1 on line 1.
    lines = r.text.splitlines()
    clean_lines = [l for l in lines if not l.startswith("#")]
    
    os.makedirs(RAW_DIR, exist_ok=True)
    with open(CSV_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(clean_lines))
    
    print(f"   > Metadata for {limit} stars written to {CSV_PATH}")

def read_stars_from_csv():
    stars = []
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stars.append({
                "apogee_id": row["apogee_id"].strip(),
                "telescope": row["telescope"].strip(),
                "field": row["field"].strip(),
                "teff": float(row["teff"]),
                "logg": float(row["logg"]),
                "fe_h": float(row["fe_h"])
            })
    return stars

def download_star_spectrum(star):
    apogee_id = star["apogee_id"]
    telescope = star["telescope"]
    field = star["field"]
    
    filename = f"{apogee_id}.fits"
    dest_path = os.path.join(OUT_DIR, filename)
    
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 10000:
        return apogee_id, True, "skipped"
        
    url = f"https://data.sdss.org/sas/dr17/apogee/spectro/aspcap/dr17/synspec_rev1/{telescope}/{field}/aspcapStar-dr17-{apogee_id}.fits"
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            with open(dest_path, "wb") as f:
                f.write(r.content)
            return apogee_id, True, "downloaded"
        else:
            return apogee_id, False, f"HTTP {r.status_code}"
    except Exception as e:
        return apogee_id, False, str(e)

def main():
    if not os.path.exists(CSV_PATH):
        try:
            fetch_metadata()
        except Exception as e:
            print(f"[Error] Failed to fetch metadata: {e}")
            return
            
    stars = read_stars_from_csv()
    num_stars = len(stars)
    print(f"Total stars to download: {num_stars} (1 FITS file per star)")
    
    os.makedirs(OUT_DIR, exist_ok=True)
    
    downloaded_count = 0
    skipped_count = 0
    failed_count = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(download_star_spectrum, star): star for star in stars}
        
        with tqdm(total=num_stars, desc="Downloading APOGEE Spectra", unit="star") as pbar:
            for future in concurrent.futures.as_completed(futures):
                apogee_id, success, reason = future.result()
                if success:
                    if reason == "downloaded":
                        downloaded_count += 1
                    elif reason == "skipped":
                        skipped_count += 1
                else:
                    failed_count += 1
                pbar.update(1)
                
    print("\n" + "=" * 55)
    print("APOGEE DOWNLOAD SUMMARY")
    print("=" * 55)
    print(f"Total Stars Target : {num_stars}")
    print(f"Files Downloaded   : {downloaded_count}")
    print(f"Files Skipped      : {skipped_count}")
    print(f"Files Failed       : {failed_count}")
    print(f"Local Storage Dir  : {OUT_DIR}")
    print("=" * 55)

if __name__ == "__main__":
    main()
