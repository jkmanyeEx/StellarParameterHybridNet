"""
GALAH DR4 x APOGEE DR17 Cross-match Counter
============================================
Finds how many stars in your LOCAL trained data exist in both surveys.

Matching strategy (tried in order):
  1. Gaia DR4 source_id  — most precise, if present in both CSVs
  2. 2MASS ID            — GALAH tmass_id vs APOGEE apogee_id (2M... format)

Usage:
    python scripts/check_crossmatch.py

Run  scripts/enrich_crossmatch_ids.py  first if GALAH CSV lacks tmass_id/source_id.
"""

import os
import sys
import csv
from datetime import datetime

import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

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


def _norm_tmass(raw):
    """
    Normalise a 2MASS ID to bare digits for comparison.
    GALAH:  '08261116-4811012'  or  '2MASS J08261116-4811012'
    APOGEE: '2M08261116-4811012'
    -> strip leading '2MASS', '2M', spaces; lowercase
    """
    s = str(raw).strip()
    for prefix in ("2MASS J", "2MASS", "2M"):
        if s.upper().startswith(prefix):
            s = s[len(prefix):]
    return s.strip().upper()


# ─────────────────────────────────────────────────────────────────────────────
# GALAH catalog loader
# ─────────────────────────────────────────────────────────────────────────────

def load_galah_catalog(trained_sobject_ids):
    """
    Returns:
        gaia_map   : sobject_id (int) -> gaia source_id (int)   [may be empty]
        tmass_map  : sobject_id (int) -> normalised 2MASS ID (str)
        gaia_col, tmass_col  : detected column names (for report)
    """
    print(f"\n[GALAH] Catalog : {GALAH_CSV}")
    if not _check_file(GALAH_CSV, "GALAH catalog CSV"):
        return {}, {}, None, None

    trained_set = set(int(x) for x in trained_sobject_ids)
    gaia_map    = {}
    tmass_map   = {}

    with open(GALAH_CSV, newline="", encoding="utf-8") as f:
        reader  = csv.DictReader(f)
        headers = reader.fieldnames or []
        print(f"  Columns : {headers}")

        gaia_col  = _pick_col(headers,
                              "source_id", "gaia_source_id",
                              "gaiaDR4_source_id", "gaia_DR4_source_id")
        tmass_col = _pick_col(headers,
                              "tmass_id", "2mass_id",
                              "tmassid",  "twomass_id")

        print(f"  Gaia column  : '{gaia_col}'")
        print(f"  2MASS column : '{tmass_col}'")

        for row in reader:
            try:
                sid = int(row["sobject_id"])
            except (KeyError, ValueError):
                continue
            if sid not in trained_set:
                continue

            if gaia_col:
                raw = row.get(gaia_col, "").strip()
                if raw and raw not in ("", "None", "nan", "0", "0.0"):
                    try:
                        gaia_map[sid] = int(float(raw))
                    except ValueError:
                        pass

            if tmass_col:
                raw = row.get(tmass_col, "").strip()
                if raw and raw not in ("", "None", "nan"):
                    norm = _norm_tmass(raw)
                    if norm:
                        tmass_map[sid] = norm

    print(f"  Trained stars          : {len(trained_set):,}")
    print(f"  With Gaia source_id    : {len(gaia_map):,}")
    print(f"  With 2MASS ID          : {len(tmass_map):,}")
    return gaia_map, tmass_map, gaia_col, tmass_col


# ─────────────────────────────────────────────────────────────────────────────
# APOGEE catalog loader
# ─────────────────────────────────────────────────────────────────────────────

def load_apogee_catalog(trained_apogee_ids):
    """
    Returns:
        gaia_map   : apogee_id (str) -> gaia source_id (int)    [may be empty]
        tmass_map  : apogee_id (str) -> normalised 2MASS ID (str)
        gaia_col   : detected column name (for report)

    Note: APOGEE apogee_id IS the 2MASS ID in '2M...' format.
    So tmass_map is trivially built from apogee_id itself — no extra column needed.
    """
    print(f"\n[APOGEE] Catalog : {APOGEE_CSV}")
    if not _check_file(APOGEE_CSV, "APOGEE catalog CSV"):
        return {}, {}, None

    trained_set = set(str(x).strip() for x in trained_apogee_ids)
    gaia_map    = {}
    tmass_map   = {}

    with open(APOGEE_CSV, newline="", encoding="utf-8") as f:
        reader  = csv.DictReader(f)
        headers = reader.fieldnames or []
        print(f"  Columns : {headers}")

        gaia_col = _pick_col(headers,
                             "gaia_source_id", "gaiaDR4_source_id",
                             "gaiaeDR4_source_id", "source_id",
                             "gaia_DR4_source_id", "gaia_eDR4_source_id")
        print(f"  Gaia column  : '{gaia_col}'")
        print(f"  2MASS column : 'apogee_id' (built-in — apogee_id IS the 2MASS ID)")

        for row in reader:
            aid = row.get("apogee_id", "").strip()
            if not aid or aid not in trained_set:
                continue

            # 2MASS from apogee_id directly
            norm = _norm_tmass(aid)
            if norm:
                tmass_map[aid] = norm

            # Gaia if column present
            if gaia_col:
                raw = row.get(gaia_col, "").strip()
                if raw and raw not in ("", "None", "nan", "0", "0.0"):
                    try:
                        gaia_map[aid] = int(float(raw))
                    except ValueError:
                        pass

    print(f"  Trained stars          : {len(trained_set):,}")
    print(f"  With Gaia source_id    : {len(gaia_map):,}")
    print(f"  With 2MASS ID          : {len(tmass_map):,}")
    return gaia_map, tmass_map, gaia_col


# ─────────────────────────────────────────────────────────────────────────────
# Intersection (Gaia preferred, 2MASS fallback)
# ─────────────────────────────────────────────────────────────────────────────

def find_intersection(g_gaia, g_tmass, a_gaia, a_tmass):
    """
    Returns list of (sobject_id, apogee_id, match_key, method)
    method: 'gaia' or '2mass'
    """
    matches = []
    used_apogee = set()

    # ── Pass 1: Gaia source_id ────────────────────────────────────────────────
    if g_gaia and a_gaia:
        apogee_gaia_inv = {}
        for aid, gsid in a_gaia.items():
            apogee_gaia_inv[gsid] = aid

        for sid, gsid in g_gaia.items():
            if gsid in apogee_gaia_inv:
                aid = apogee_gaia_inv[gsid]
                matches.append((sid, aid, str(gsid), "gaia"))
                used_apogee.add(aid)

    gaia_count = len(matches)

    # ── Pass 2: 2MASS ID ──────────────────────────────────────────────────────
    # Build apogee 2MASS -> apogee_id, excluding already matched
    apogee_tmass_inv = {}
    for aid, tm in a_tmass.items():
        if aid not in used_apogee and tm:
            apogee_tmass_inv[tm] = aid

    # Build already-matched GALAH sobject_ids to avoid double-counting
    matched_galah = {m[0] for m in matches}

    for sid, tm in g_tmass.items():
        if sid in matched_galah:
            continue
        if tm in apogee_tmass_inv:
            aid = apogee_tmass_inv[tm]
            matches.append((sid, aid, tm, "2mass"))

    tmass_count = len(matches) - gaia_count
    print(f"\n  Matched via Gaia source_id : {gaia_count:,}")
    print(f"  Matched via 2MASS ID       : {tmass_count:,}")
    print(f"  Total matches              : {len(matches):,}")
    return matches


# ─────────────────────────────────────────────────────────────────────────────
# Test set split
# ─────────────────────────────────────────────────────────────────────────────

def split_by_test_sets(matches, galah_ids, apogee_ids,
                       galah_test_idx, apogee_test_idx):
    galah_test_set  = set(int(galah_ids[i])  for i in galah_test_idx)
    apogee_test_set = set(str(apogee_ids[i]) for i in apogee_test_idx)

    both, g_only, a_only, neither = [], [], [], []
    for sobject_id, apogee_id, key, method in matches:
        in_g = int(sobject_id) in galah_test_set
        in_a = str(apogee_id)  in apogee_test_set
        tup  = (sobject_id, apogee_id, key, method)
        if   in_g and in_a: both.append(tup)
        elif in_g:           g_only.append(tup)
        elif in_a:           a_only.append(tup)
        else:                neither.append(tup)

    return both, g_only, a_only, neither


# ─────────────────────────────────────────────────────────────────────────────
# Report writer
# ─────────────────────────────────────────────────────────────────────────────

def write_report(path, lines):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n  Report saved : {path}")


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

    # ── Load catalogs ─────────────────────────────────────────────────────────
    g_gaia, g_tmass, galah_gaia_col, galah_tmass_col = load_galah_catalog(galah_ids)
    a_gaia, a_tmass, apogee_gaia_col                 = load_apogee_catalog(apogee_ids)

    report_lines += [
        "▶ [SECTION 2] Catalog Column Detection",
        f"   GALAH  Gaia column   : '{galah_gaia_col}'",
        f"   GALAH  2MASS column  : '{galah_tmass_col}'",
        f"   APOGEE Gaia column   : '{apogee_gaia_col}'",
        f"   APOGEE 2MASS column  : 'apogee_id' (built-in)",
        "",
        f"   GALAH  stars w/ Gaia   : {len(g_gaia):,}  / {len(galah_ids):,}",
        f"   GALAH  stars w/ 2MASS  : {len(g_tmass):,}  / {len(galah_ids):,}",
        f"   APOGEE stars w/ Gaia   : {len(a_gaia):,} / {len(apogee_ids):,}",
        f"   APOGEE stars w/ 2MASS  : {len(a_tmass):,} / {len(apogee_ids):,}",
        "",
    ]

    # ── Check if we have anything to match on ─────────────────────────────────
    has_gaia  = bool(g_gaia  and a_gaia)
    has_tmass = bool(g_tmass and a_tmass)

    if not has_gaia and not has_tmass:
        msg = ("No common identifier available.\n"
               "  Run  python scripts/enrich_crossmatch_ids.py  to add "
               "tmass_id to the GALAH CSV.")
        print(f"\n  ERROR: {msg}")
        report_lines += [f"  ERROR: {msg}", ""]
        write_report(OUT_REPORT, report_lines)
        sys.exit(1)

    if not has_tmass and not has_gaia:
        print("\n  WARNING: No 2MASS IDs in GALAH CSV.")
        print("  Run  python scripts/enrich_crossmatch_ids.py  first.")

    # ── Intersection ──────────────────────────────────────────────────────────
    matches = find_intersection(g_gaia, g_tmass, a_gaia, a_tmass)

    gaia_count  = sum(1 for m in matches if m[3] == "gaia")
    tmass_count = sum(1 for m in matches if m[3] == "2mass")

    print(f"\n{sep}")
    print(f"  CROSS-MATCH RESULT")
    print(f"{sep}")
    print(f"  Total matched : {len(matches):,}")

    report_lines += [
        "▶ [SECTION 3] Cross-match Result",
        f"   Matched via Gaia source_id : {gaia_count:,}",
        f"   Matched via 2MASS ID       : {tmass_count:,}",
        f"   Total                      : {len(matches):,}",
        "",
    ]

    if len(matches) == 0:
        print("\n  No matches found.")
        print("  If GALAH CSV has no tmass_id, run enrich_crossmatch_ids.py first.")
        report_lines += [
            "  No matches found.",
            "  If GALAH CSV lacks tmass_id, run enrich_crossmatch_ids.py.",
            "",
        ]
        write_report(OUT_REPORT, report_lines)
        sys.exit(0)

    # Save full match list
    os.makedirs(os.path.dirname(OUT_ALL), exist_ok=True)
    with open(OUT_ALL, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sobject_id", "apogee_id", "match_key", "match_method"])
        w.writerows(matches)
    print(f"\n  Full match list saved : {OUT_ALL}")
    report_lines.append(f"   Full match list : {OUT_ALL}")

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
            f"   GALAH  sealed test size  : {len(galah_test_idx):,}",
            f"   APOGEE sealed test size  : {len(apogee_test_idx):,}",
            "",
            f"   In BOTH test sets        : {len(both):,}  <- safe for cross-validation",
            f"   In GALAH test set only   : {len(g_only):,}",
            f"   In APOGEE test set only  : {len(a_only):,}",
            f"   In neither (train/val)   : {len(neither):,}",
            "",
        ]

        cv_stars = both if both else neither
        cv_label = "both-test" if both else "neither (fallback)"

        if not both:
            print(f"\n  No stars in both test sets.")
            print(f"  Fallback: using {len(neither):,} 'in-neither' stars for CV.")
            report_lines += [
                f"  NOTE: Fallback CV set used ({cv_label}).",
                f"  These may have been in training. Use with caution.",
                "",
            ]
        else:
            print(f"\n  {len(both):,} stars safe for cross-survey validation.")

        with open(OUT_CV, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["sobject_id", "apogee_id", "match_key", "match_method"])
            w.writerows(cv_stars)
        print(f"  CV set saved ({cv_label}) : {OUT_CV}")
        report_lines.append(f"   CV set saved ({cv_label}) : {OUT_CV}")

    else:
        missing = []
        if not galah_test_ok:  missing.append("GALAH test_indices.npy")
        if not apogee_test_ok: missing.append("APOGEE test_indices.npy")
        print(f"\n  NOTE: {', '.join(missing)} not found.")
        report_lines += [
            "",
            "▶ [SECTION 4] Test Set Breakdown",
            f"   SKIPPED — missing: {', '.join(missing)}",
            "",
        ]

    # ── Write report ──────────────────────────────────────────────────────────
    report_lines += [sep, "  END OF REPORT", sep]
    write_report(OUT_REPORT, report_lines)
    print(f"\n{sep}")


if __name__ == "__main__":
    main()
