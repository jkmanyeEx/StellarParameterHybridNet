import numpy as np
import os

def main():
    print("Aligning APOGEE Catalog Labels with Spectrum ID sequence...")
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    processed_dir = os.path.join(base_dir, "data", "apogee", "processed")
    
    catalog_csv   = os.path.join(base_dir, "data", "apogee", "raw", "allStar-dr17.csv")
    catalog_path  = os.path.join(base_dir, "data", "apogee", "raw", "allStar-dr17.fits")
    star_ids_path = os.path.join(processed_dir, "star_ids.npy")
    save_path     = os.path.join(processed_dir, "Y_labels.npy")

    if not os.path.exists(star_ids_path):
        print(f"Error: '{star_ids_path}' not found. Run preprocess_flux.py first.")
        return

    star_ids = np.load(star_ids_path, allow_pickle=True)
    num_stars = len(star_ids)

    # 1. Check if CSV catalog exists (downloaded online)
    if os.path.exists(catalog_csv):
        import csv
        print(f"   > Reading APOGEE catalog from CSV: {catalog_csv}")
        cat_map = {}
        with open(catalog_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sid = row["apogee_id"].strip()
                cat_map[sid] = {
                    "TEFF": float(row["teff"]),
                    "LOGG": float(row["logg"]),
                    "FEH":  float(row["fe_h"])
                }
        
        aligned_teff, aligned_logg, aligned_feh = [], [], []
        vac_hit, null_count = 0, 0
        for sid in star_ids:
            spec_id = str(sid).strip()
            if spec_id in cat_map:
                aligned_teff.append(cat_map[spec_id]["TEFF"])
                aligned_logg.append(cat_map[spec_id]["LOGG"])
                aligned_feh.append(cat_map[spec_id]["FEH"])
                vac_hit += 1
            else:
                aligned_teff.append(-999.0)
                aligned_logg.append(-999.0)
                aligned_feh.append(-999.0)
                null_count += 1
        
        Y_labels = np.column_stack((aligned_teff, aligned_logg, aligned_feh))
        print(f"   > Real catalog alignment complete. Hits: {vac_hit}, Misses: {null_count}")

    # 2. Fallback to FITS catalog
    elif os.path.exists(catalog_path):
        import astropy.io.fits as fits
        print(f"   > Reading APOGEE catalog from FITS: {catalog_path}")
        with fits.open(catalog_path) as hdul:
            data = hdul[1].data
            cat_map = {}
            for row in data:
                sid = str(row["APOGEE_ID"]).strip()
                cat_map[sid] = {
                    "TEFF": float(row["TEFF"]),
                    "LOGG": float(row["LOGG"]),
                    "FEH":  float(row["FE_H"])
                }
        
        aligned_teff, aligned_logg, aligned_feh = [], [], []
        vac_hit, null_count = 0, 0
        for sid in star_ids:
            spec_id = str(sid).strip()
            if spec_id in cat_map:
                aligned_teff.append(cat_map[spec_id]["TEFF"])
                aligned_logg.append(cat_map[spec_id]["LOGG"])
                aligned_feh.append(cat_map[spec_id]["FEH"])
                vac_hit += 1
            else:
                aligned_teff.append(-999.0)
                aligned_logg.append(-999.0)
                aligned_feh.append(-999.0)
                null_count += 1
        
        Y_labels = np.column_stack((aligned_teff, aligned_logg, aligned_feh))
        print(f"   > Real catalog alignment complete. Hits: {vac_hit}, Misses: {null_count}")

    else:
        print("⚠️  APOGEE Catalog CSV/FITS not found. Generating synthetic labels...")
        np.random.seed(42)
        # APOGEE stars are mostly red giants: Teff 3000 ~ 5500 K, logg 0.5 ~ 4.0 dex
        teff = np.random.uniform(3000, 5500, num_stars)
        logg = np.random.uniform(0.5, 4.0, num_stars)
        feh = np.random.uniform(-2.5, 0.5, num_stars)
        
        Y_labels = np.column_stack((teff, logg, feh))
        print(f"   > Generated {num_stars} synthetic labels.")

    np.save(save_path, Y_labels)
    print(f"[Success] Shape: {Y_labels.shape} → {save_path}")


if __name__ == "__main__":
    main()
