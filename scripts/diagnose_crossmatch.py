"""
Cross-match Diagnostic
======================
Diagnoses why so few stars matched between GALAH and APOGEE.
Prints sample IDs from both sides to identify format issues.

Usage:
    python scripts/diagnose_crossmatch.py
"""

import os
import csv

BASE_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GALAH_CSV  = os.path.join(BASE_DIR, "data", "galah",  "raw", "galah_dr4_allstar.csv")
APOGEE_CSV = os.path.join(BASE_DIR, "data", "apogee", "raw", "allStar-dr17.csv")
MATCH_CSV  = os.path.join(BASE_DIR, "data", "crossmatch_galah_apogee.csv")

GALAH_IDS_NPY  = os.path.join(BASE_DIR, "data", "galah",  "processed", "star_ids.npy")
APOGEE_IDS_NPY = os.path.join(BASE_DIR, "data", "apogee", "processed", "star_ids.npy")


def _norm_tmass(raw):
    s = str(raw).strip()
    for prefix in ("2MASS J", "2MASS", "2M"):
        if s.upper().startswith(prefix):
            s = s[len(prefix):]
    return s.strip().upper()


def main():
    sep = "=" * 65
    print(sep)
    print("  Cross-match Diagnostic")
    print(sep)

    # ── 1. APOGEE CSV row count vs processed star_ids ─────────────────────────
    print("\n[1] APOGEE CSV vs processed star_ids")
    apogee_csv_rows = 0
    apogee_sample   = []
    with open(APOGEE_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            apogee_csv_rows += 1
            if len(apogee_sample) < 5:
                apogee_sample.append(row.get("apogee_id", ""))

    import numpy as np
    apogee_ids_npy = np.load(APOGEE_IDS_NPY, allow_pickle=True)

    print(f"  allStar-dr17.csv rows      : {apogee_csv_rows:,}")
    print(f"  star_ids.npy entries       : {len(apogee_ids_npy):,}")
    print(f"  Sample apogee_id (CSV)     : {apogee_sample}")
    print(f"  Sample apogee_id (npy)     : {[str(x) for x in apogee_ids_npy[:5]]}")

    # ── 2. GALAH tmass_id format samples ─────────────────────────────────────
    print("\n[2] GALAH tmass_id format")
    galah_tmass_sample = []
    galah_tmass_norm_sample = []
    with open(GALAH_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tm = row.get("tmass_id", "").strip()
            if tm and tm not in ("", "None", "nan") and len(galah_tmass_sample) < 10:
                galah_tmass_sample.append(tm)
                galah_tmass_norm_sample.append(_norm_tmass(tm))

    print(f"  Raw tmass_id samples  : {galah_tmass_sample}")
    print(f"  Normalised            : {galah_tmass_norm_sample}")

    # ── 3. APOGEE apogee_id format samples ───────────────────────────────────
    print("\n[3] APOGEE apogee_id format")
    apogee_norm_sample = [_norm_tmass(x) for x in apogee_sample]
    print(f"  Raw apogee_id samples : {apogee_sample}")
    print(f"  Normalised            : {apogee_norm_sample}")

    # ── 4. Manual intersection attempt on first 200 of each ──────────────────
    print("\n[4] Manual intersection test (first 500 GALAH tmass vs all APOGEE apogee_id)")

    galah_norm_set = set()
    with open(GALAH_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= 500:
                break
            tm = row.get("tmass_id", "").strip()
            if tm and tm not in ("", "None", "nan"):
                galah_norm_set.add(_norm_tmass(tm))

    apogee_norm_set = set()
    with open(APOGEE_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            aid = row.get("apogee_id", "").strip()
            if aid:
                apogee_norm_set.add(_norm_tmass(aid))

    overlap = galah_norm_set & apogee_norm_set
    print(f"  GALAH  first-500 normalised tmass IDs  : {len(galah_norm_set)}")
    print(f"  APOGEE all normalised apogee IDs       : {len(apogee_norm_set)}")
    print(f"  Overlap                                : {len(overlap)}")
    if overlap:
        print(f"  Sample overlapping IDs : {list(overlap)[:5]}")
    else:
        print("  NO OVERLAP — format mismatch or disjoint sky coverage")

        # Show side-by-side comparison
        g_sample = sorted(galah_norm_set)[:3]
        a_sample = sorted(apogee_norm_set)[:3]
        print(f"\n  GALAH  normalised sample  : {g_sample}")
        print(f"  APOGEE normalised sample  : {a_sample}")

    # ── 5. Check if apogee_ids in npy vs csv differ ───────────────────────────
    print("\n[5] APOGEE npy vs CSV ID alignment")
    csv_ids = set()
    with open(APOGEE_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            csv_ids.add(row.get("apogee_id", "").strip())
    npy_ids = set(str(x).strip() for x in apogee_ids_npy)
    only_in_npy = npy_ids - csv_ids
    only_in_csv = csv_ids - npy_ids
    print(f"  IDs in npy only (not in CSV) : {len(only_in_npy):,}")
    print(f"  IDs in CSV only (not in npy) : {len(only_in_csv):,}")
    if only_in_npy:
        print(f"  Sample npy-only IDs : {list(only_in_npy)[:3]}")

    # ── 6. Existing match sample ──────────────────────────────────────────────
    if os.path.exists(MATCH_CSV):
        print("\n[6] Sample from existing crossmatch_galah_apogee.csv")
        with open(MATCH_CSV, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for r in rows[:5]:
            print(f"  {r}")

    print(f"\n{sep}")


if __name__ == "__main__":
    main()
