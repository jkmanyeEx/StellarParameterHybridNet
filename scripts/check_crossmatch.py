"""
GALAH DR4 x APOGEE DR17 Cross-match Counter
============================================
Finds how many stars in your LOCAL trained data exist in both surveys,
using Gaia DR3 source_id as the common identifier.

Usage:
    python scripts/check_crossmatch.py
"""

import os
import sys
import csv
from datetime import datetime

import numpy as np

# ── Project root (one level up from scripts/) ─────────────────────────────────
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# ── Paths ─────────────────────────────────────────────────────────────────────
GALAH_CSV       = os.path.join(BASE_DIR, "data", "galah",  "raw", "galah_dr4_allstar.csv")
GALAH_IDS_NPY   = os.path.join(BASE_DIR, "data", "galah",  "processed", "star_ids.npy")
GALAH_TEST_NPY  = os.path.join(BASE_DIR, "data", "galah",  "processed", "test_indices.npy")

APOGEE_CSV      = os.path.join(BASE_DIR, "data", "apogee", "raw", "allStar-dr17.csv")
APOGEE_IDS_NPY  = os.path.join(BASE_DIR, "data", "apogee", "processed", "star_ids.npy")
APOGEE_TEST_NPY = os.path.join(BASE_DIR, "data", "apogee", "processed", "test_indices.npy")

OUT_ALL         = os.path.join(BASE_DIR, "data", "crossmatch_galah_apogee.csv")
OUT_CV          = os.path.join(BASE_DIR, "data", "crossmatch_cv_set.csv")
OUT_REPORT      = os.path.join(BASE_DIR, "report", "crossmatch", "crossmatch_report.txt")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _check_file(path, label):
    if not os.path.exists(path):
        print(f"  ERROR: {label} not found at:\n    {path}")
        return False
    return True


def _pick_col(headers, *candidates):
    hl = {h.lower(): h for h in headers}
    for c in candidates:
        if c.lower() in hl:
            return hl[c.lower()]
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — GALAH: sobject_id -> gaia_source_id
# ─────────────────────────────────────────────────────────────────────────────

def load_galah_gaia_map(trained_sobject_ids):
    print(f"\n[GALAH] Catalog : {GALAH_CSV}")
    if not _check_file(GALAH_CSV, "GALAH catalog CSV"):
        return {}, None, None

    trained_set = set(int(x) for x in trained_sobject_ids)
    mapping     = {}
    no_gaia     = 0

    with open(GALAH_CSV, newline="", encoding="utf-8") as f:
        reader  = csv.DictReader(f)
        headers = reader.fieldnames or []
        print(f"  Columns (first 12): {headers[:12]}")

        gaia_col = _pick_col(
            headers,
            "source_id",
            "gaia_source_id",
            "gaiadr3_source_id",
            "dr3_source_id",
            "gaia_dr3_source_id",
        )
        print(f"  Gaia column selected : '{gaia_col}'")

        if gaia_col is None:
            print("  WARNING: No Gaia source_id column found in GALAH catalog.")
            print(f"  All available columns: {headers}")
            return {}, gaia_col, headers

        for row in reader:
            try:
                sid = int(row["sobject_id"])
            except (KeyError, ValueError):
                continue
            if sid not in trained_set:
                continue
            raw = row.get(gaia_col, "").strip()
            if not raw or raw in ("", "None", "nan", "0", "0.0"):
                no_gaia += 1
                continue
            try:
                mapping[sid] = int(float(raw))
            except ValueError:
                no_gaia += 1

    print(f"  Trained stars          : {len(trained_set):,}")
    print(f"  With Gaia source_id    : {len(mapping):,}")
    print(f"  Missing Gaia source_id : {no_gaia:,}")
    return mapping, gaia_col, headers


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — APOGEE: apogee_id -> gaia_source_id
# ─────────────────────────────────────────────────────────────────────────────

def load_apogee_gaia_map(trained_apogee_ids):
    print(f"\n[APOGEE] Catalog : {APOGEE_CSV}")
    if not _check_file(APOGEE_CSV, "APOGEE catalog CSV"):
        return {}, None, None

    trained_set = set(str(x).strip() for x in trained_apogee_ids)
    mapping     = {}
    no_gaia     = 0

    with open(APOGEE_CSV, newline="", encoding="utf-8") as f:
        reader  = csv.DictReader(f)
        headers = reader.fieldnames or []
        print(f"  Columns (first 12): {headers[:12]}")

        gaia_col = _pick_col(
            headers,
            "gaia_source_id",
            "gaiadr3_source_id",
            "gaiaedr3_source_id",
            "source_id",
            "gaia_dr3_source_id",
            "gaia_edr3_source_id",
        )
        print(f"  Gaia column selected : '{gaia_col}'")

        if gaia_col is None:
            print("  WARNING: No Gaia source_id column in APOGEE catalog.")
            print("  Re-query SkyServer and include gaia_source_id in SELECT.")
            return {}, gaia_col, headers

        for row in reader:
            aid = row.get("apogee_id", "").strip()
            if not aid or aid not in trained_set:
                continue
            raw = row.get(gaia_col, "").strip()
            if not raw or raw in ("", "None", "nan", "0", "0.0"):
                no_gaia += 1
                continue
            try:
                mapping[aid] = int(float(raw))
            except ValueError:
                no_gaia += 1

    print(f"  Trained stars          : {len(trained_set):,}")
    print(f"  With Gaia source_id    : {len(mapping):,}")
    print(f"  Missing Gaia source_id : {no_gaia:,}")
    return mapping, gaia_col, headers


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Intersect on Gaia source_id
# ─────────────────────────────────────────────────────────────────────────────

def find_intersection(galah_map, apogee_map):
    apogee_inv = {gsid: aid for aid, gsid in apogee_map.items()}
    matches = []
    for sobject_id, gsid in galah_map.items():
        if gsid in apogee_inv:
            matches.append((sobject_id, apogee_inv[gsid], gsid))
    return matches


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Split by sealed test sets
# ─────────────────────────────────────────────────────────────────────────────

def split_by_test_sets(matches, galah_ids, apogee_ids,
                       galah_test_idx, apogee_test_idx):
    galah_test_set  = set(int(galah_ids[i])  for i in galah_test_idx)
    apogee_test_set = set(str(apogee_ids[i]) for i in apogee_test_idx)

    both, g_only, a_only, neither = [], [], [], []
    for sobject_id, apogee_id, gaia_id in matches:
        in_g = int(sobject_id) in galah_test_set
        in_a = str(apogee_id)  in apogee_test_set
        if   in_g and in_a: both.append((sobject_id, apogee_id, gaia_id))
        elif in_g:           g_only.append((sobject_id, apogee_id, gaia_id))
        elif in_a:           a_only.append((sobject_id, apogee_id, gaia_id))
        else:                neither.append((sobject_id, apogee_id, gaia_id))

    return both, g_only, a_only, neither


# ─────────────────────────────────────────────────────────────────────────────
# Report writer
# ─────────────────────────────────────────────────────────────────────────────

def write_report(path, lines):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n  Report saved          : {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    sep = "=" * 65
    print(sep)
    print("  GALAH DR4 x APOGEE DR17  —  Cross-match Analysis")
    print(sep)

    report_lines = [
        sep,
        "  GALAH DR4 x APOGEE DR17 Cross-match Report",
        sep,
        f"  Generated : {datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}",
        "",
    ]

    # ── Verify processed IDs ──────────────────────────────────────────────────
    for path, label in [
        (GALAH_IDS_NPY,  "GALAH star_ids.npy"),
        (APOGEE_IDS_NPY, "APOGEE star_ids.npy"),
    ]:
        if not _check_file(path, label):
            print("  Run the respective training pipeline first.")
            sys.exit(1)

    galah_ids  = np.load(GALAH_IDS_NPY,  allow_pickle=True)
    apogee_ids = np.load(APOGEE_IDS_NPY, allow_pickle=True)

    print(f"\n  GALAH  trained stars : {len(galah_ids):,}")
    print(f"  APOGEE trained stars : {len(apogee_ids):,}")

    report_lines += [
        "▶ [SECTION 1] Trained Dataset Sizes",
        f"   GALAH  trained stars : {len(galah_ids):,}",
        f"   APOGEE trained stars : {len(apogee_ids):,}",
        "",
    ]

    # ── Build Gaia maps ───────────────────────────────────────────────────────
    galah_map,  galah_gaia_col,  galah_headers  = load_galah_gaia_map(galah_ids)
    apogee_map, apogee_gaia_col, apogee_headers = load_apogee_gaia_map(apogee_ids)

    report_lines += [
        "▶ [SECTION 2] Catalog Column Detection",
        f"   GALAH  Gaia column   : '{galah_gaia_col}'",
        f"   APOGEE Gaia column   : '{apogee_gaia_col}'",
        f"   GALAH  stars w/ Gaia : {len(galah_map):,}  / {len(galah_ids):,}",
        f"   APOGEE stars w/ Gaia : {len(apogee_map):,} / {len(apogee_ids):,}",
        "",
    ]

    if not galah_map or not apogee_map:
        msg = "Cannot proceed without Gaia source_id in both catalogs."
        print(f"\n  {msg}")
        report_lines += [f"  ERROR: {msg}", ""]
        write_report(OUT_REPORT, report_lines)
        sys.exit(1)

    # ── Intersection ──────────────────────────────────────────────────────────
    matches = find_intersection(galah_map, apogee_map)

    print(f"\n{sep}")
    print(f"  CROSS-MATCH RESULT")
    print(f"{sep}")
    print(f"  Stars in BOTH trained sets : {len(matches):,}")

    report_lines += [
        "▶ [SECTION 3] Cross-match Result",
        f"   Stars in BOTH trained sets (Gaia source_id match) : {len(matches):,}",
        "",
    ]

    if len(matches) == 0:
        msg = "No matches found."
        print(f"\n  {msg}")
        report_lines += [
            f"  {msg}",
            "  Possible reasons:",
            "    1. Quality filters select non-overlapping sky regions",
            "    2. Gaia source_id column name not detected — check warnings",
            "",
        ]
        write_report(OUT_REPORT, report_lines)
        sys.exit(0)

    # Save full match list
    os.makedirs(os.path.dirname(OUT_ALL), exist_ok=True)
    with open(OUT_ALL, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sobject_id", "apogee_id", "gaia_source_id"])
        w.writerows(matches)
    print(f"\n  Full match list saved : {OUT_ALL}")
    report_lines.append(f"   Full match list       : {OUT_ALL}")

    # ── Test set breakdown ────────────────────────────────────────────────────
    galah_test_ok  = os.path.exists(GALAH_TEST_NPY)
    apogee_test_ok = os.path.exists(APOGEE_TEST_NPY)

    if galah_test_ok and apogee_test_ok:
        galah_test_idx  = np.load(GALAH_TEST_NPY)
        apogee_test_idx = np.load(APOGEE_TEST_NPY)

        both, g_only, a_only, neither = split_by_test_sets(
            matches, galah_ids, apogee_ids,
            galah_test_idx, apogee_test_idx,
        )

        print(f"\n  Split by sealed test sets:")
        print(f"  ┌─────────────────────────────────────────────────┐")
        print(f"  │  In BOTH test sets     : {len(both):>6,}  ← use for CV  │")
        print(f"  │  GALAH test set only   : {len(g_only):>6,}               │")
        print(f"  │  APOGEE test set only  : {len(a_only):>6,}               │")
        print(f"  │  In neither (train/val): {len(neither):>6,}               │")
        print(f"  └─────────────────────────────────────────────────┘")

        report_lines += [
            "",
            "▶ [SECTION 4] Test Set Breakdown",
            f"   GALAH  sealed test size  : {len(galah_test_idx):,}  (10% of trained)",
            f"   APOGEE sealed test size  : {len(apogee_test_idx):,}  (10% of trained)",
            "",
            f"   In BOTH test sets        : {len(both):,}  <- safe for cross-validation",
            f"   In GALAH test set only   : {len(g_only):,}",
            f"   In APOGEE test set only  : {len(a_only):,}",
            f"   In neither (train/val)   : {len(neither):,}",
            "",
        ]

        if len(both) == 0:
            print("\n  No stars in both sealed test sets.")
            print("  Falling back: using 'in neither' stars for CV.")
            print(f"  ({len(neither):,} stars were in training/val of both — not ideal but usable)")
            report_lines += [
                "  NOTE: No stars in both sealed test sets.",
                f"  Fallback: {len(neither):,} stars in neither training set saved as CV candidates.",
                "  These may have been in training. Use with caution.",
                "",
            ]
            with open(OUT_CV, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["sobject_id", "apogee_id", "gaia_source_id"])
                w.writerows(neither)
            print(f"  Fallback CV set saved : {OUT_CV}")
            report_lines.append(f"   Fallback CV set saved : {OUT_CV}")
        else:
            print(f"\n  These {len(both):,} stars were NEVER seen by either model.")
            print(f"  Safe to use as cross-survey validation set.")
            report_lines += [
                f"  These {len(both):,} stars were never seen by either model during training.",
                "  They are safe to use as an independent cross-survey validation set.",
                "",
            ]
            with open(OUT_CV, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["sobject_id", "apogee_id", "gaia_source_id"])
                w.writerows(both)
            print(f"  CV set saved          : {OUT_CV}")
            report_lines.append(f"   CV set saved          : {OUT_CV}")

    else:
        missing = []
        if not galah_test_ok:  missing.append("GALAH test_indices.npy")
        if not apogee_test_ok: missing.append("APOGEE test_indices.npy")
        print(f"\n  NOTE: {', '.join(missing)} not found.")
        print("  Re-train both models to generate sealed test sets.")
        report_lines += [
            "",
            "▶ [SECTION 4] Test Set Breakdown",
            f"   SKIPPED — missing: {', '.join(missing)}",
            "   Re-train both models to generate test_indices.npy,",
            "   then re-run this script.",
            "",
        ]

    # ── Write report ──────────────────────────────────────────────────────────
    report_lines += [sep, "  END OF REPORT", sep]
    write_report(OUT_REPORT, report_lines)

    print(f"\n{sep}")


if __name__ == "__main__":
    main()
