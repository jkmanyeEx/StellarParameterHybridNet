import os
import sys
import csv
import requests
import concurrent.futures
from tqdm import tqdm

# Add project root to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
RAW_DIR = os.path.join(BASE_DIR, "data", "galah", "raw")
OUT_DIR = os.path.join(RAW_DIR, "spectra")
CSV_PATH = os.path.join(RAW_DIR, "galah_dr4_allstar.csv")
MAX_WORKERS = 16

def fetch_metadata(limit=15000):
    print("🛰️  Querying DataCentral online for best GALAH DR4 stars...")
    query = f"""
    SELECT TOP {limit}
        sobject_id, teff, logg, fe_h, snr_px_ccd1, snr_px_ccd2, snr_px_ccd3, snr_px_ccd4
    FROM galah_dr4.mainstartable
    WHERE snr_px_ccd1 > 30 AND snr_px_ccd2 > 30 AND snr_px_ccd3 > 30 AND snr_px_ccd4 > 30
      AND flag_sp = 0 AND flag_fe_h = 0
      AND teff BETWEEN 4000 AND 7000
      AND logg BETWEEN 1.0 AND 5.0
      AND fe_h BETWEEN -3.0 AND 0.5
    ORDER BY snr_px_ccd1 DESC
    """
    url = "https://datacentral.org.au/vo/tap/sync"
    data = {
        "REQUEST": "doQuery",
        "LANG": "ADQL",
        "FORMAT": "csv",
        "QUERY": query
    }
    r = requests.post(url, data=data, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"Failed to query DataCentral: HTTP {r.status_code}\n{r.text[:200]}")
    
    os.makedirs(RAW_DIR, exist_ok=True)
    with open(CSV_PATH, "w", encoding="utf-8") as f:
        f.write(r.text)
    
    print(f"   > Metadata for {limit} stars written to {CSV_PATH}")

def read_stars_from_csv():
    stars = []
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stars.append({
                "sobject_id": row["sobject_id"].strip(),
                "teff": float(row["teff"]),
                "logg": float(row["logg"]),
                "fe_h": float(row["fe_h"])
            })
    return stars

def download_single_arm(sobject_id, arm_filt):
    filename = f"{sobject_id}_{arm_filt}.fits"
    dest_path = os.path.join(OUT_DIR, filename)
    
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 10000:
        return True, "skipped"
        
    url = "https://datacentral.org.au/vo/slink/links"
    params = {
        "ID": sobject_id,
        "DR": "galah_dr4",
        "FILT": arm_filt,
        "RESPONSEFORMAT": "fits"
    }
    try:
        r = requests.get(url, params=params, timeout=30)
        if r.status_code == 200:
            with open(dest_path, "wb") as f:
                f.write(r.content)
            return True, "downloaded"
        else:
            return False, f"HTTP {r.status_code}"
    except Exception as e:
        return False, str(e)

def download_star_spectra(star):
    sobject_id = star["sobject_id"]
    results = []
    for filt in ["B", "G", "R", "I"]:
        success, reason = download_single_arm(sobject_id, filt)
        results.append((filt, success, reason))
    return sobject_id, results

def main():
    if not os.path.exists(CSV_PATH):
        try:
            fetch_metadata()
        except Exception as e:
            print(f"[Error] Failed to fetch metadata: {e}")
            return
            
    stars = read_stars_from_csv()
    num_stars = len(stars)
    print(f"Total stars to download: {num_stars} (4 arms per star = {num_stars * 4} files)")
    
    os.makedirs(OUT_DIR, exist_ok=True)
    
    downloaded_count = 0
    skipped_count = 0
    failed_count = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(download_star_spectra, star): star for star in stars}
        
        with tqdm(total=num_stars, desc="Downloading GALAH Spectra", unit="star") as pbar:
            for future in concurrent.futures.as_completed(futures):
                sobject_id, results = future.result()
                all_ok = True
                for filt, success, reason in results:
                    if success:
                        if reason == "downloaded":
                            downloaded_count += 1
                        elif reason == "skipped":
                            skipped_count += 1
                    else:
                        all_ok = False
                        failed_count += 1
                        
                pbar.update(1)
                
    print("\n" + "=" * 55)
    print("GALAH DOWNLOAD SUMMARY")
    print("=" * 55)
    print(f"Total Stars Target : {num_stars}")
    print(f"Files Downloaded   : {downloaded_count}")
    print(f"Files Skipped      : {skipped_count}")
    print(f"Files Failed       : {failed_count}")
    print(f"Local Storage Dir  : {OUT_DIR}")
    print("=" * 55)

if __name__ == "__main__":
    main()
