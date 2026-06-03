"""
GALAH DR4 label alignment.

Reads galah_dr4_allstar.csv (downloaded by download_spec.py with
flag_sp = 0 and flag_fe_h = 0 filters already applied) and aligns
the stellar parameter labels to the star_ids order produced by
preprocess_flux.py.

CSV columns used:
  sobject_id  — unique spectrum identifier (integer, stored as string)
  teff        — effective temperature (K)
  logg        — surface gravity (dex)
  fe_h        — metallicity [Fe/H] (dex)

Stars not found in the catalog receive -999 sentinel labels and are
filtered out by the Dataset and training engine.
"""

import numpy as np
import os
import csv


def main():
    base_dir      = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    processed_dir = os.path.join(base_dir, "data", "galah", "processed")
    raw_dir       = os.path.join(base_dir, "data", "galah", "raw")

    catalog_csv   = os.path.join(raw_dir, "galah_dr4_allstar.csv")
    star_ids_path = os.path.join(processed_dir, "star_ids.npy")
    save_path     = os.path.join(processed_dir, "Y_labels.npy")

    if not os.path.exists(star_ids_path):
        raise FileNotFoundError(
            f"star_ids.npy not found at: {star_ids_path}\n"
            "Execute src/data/galah/preprocess_flux.py first."
        )
    if not os.path.exists(catalog_csv):
        raise FileNotFoundError(
            f"GALAH catalog not found at: {catalog_csv}\n"
            "Execute scripts/galah/download_spec.py first."
        )

    star_ids  = np.load(star_ids_path, allow_pickle=True)
    num_stars = len(star_ids)

    print(f"[Labels] Aligning labels for {num_stars} preprocessed spectra...")
    print(f"[Labels] Catalog source: {catalog_csv}")

    # Build sobject_id -> (teff, logg, fe_h) lookup
    # The CSV was queried with flag_sp = 0 and flag_fe_h = 0, so all
    # entries are already quality-filtered.
    cat_map = {}
    with open(catalog_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = row["sobject_id"].strip()
            try:
                cat_map[sid] = (
                    float(row["teff"]),
                    float(row["logg"]),
                    float(row["fe_h"]),
                )
            except (ValueError, KeyError):
                continue  # Malformed row — skip

    print(f"[Labels] Catalog entries loaded: {len(cat_map)}")

    aligned_teff, aligned_logg, aligned_feh = [], [], []
    hit, miss = 0, 0

    for sid in star_ids:
        key = str(sid).strip()
        if key in cat_map:
            t, g, z = cat_map[key]
            aligned_teff.append(t)
            aligned_logg.append(g)
            aligned_feh.append(z)
            hit += 1
        else:
            aligned_teff.append(-999.0)
            aligned_logg.append(-999.0)
            aligned_feh.append(-999.0)
            miss += 1

    Y_labels = np.column_stack((aligned_teff, aligned_logg, aligned_feh))

    os.makedirs(processed_dir, exist_ok=True)
    np.save(save_path, Y_labels)

    print(f"[Labels] Alignment complete:")
    print(f"   Total spectra  : {num_stars}")
    print(f"   Catalog hits   : {hit}  ({hit / num_stars * 100:.1f}%)")
    print(f"   Catalog misses : {miss} (assigned -999, filtered during training)")
    print(f"   Saved to       : {save_path}  shape={Y_labels.shape}")


if __name__ == "__main__":
    main()
