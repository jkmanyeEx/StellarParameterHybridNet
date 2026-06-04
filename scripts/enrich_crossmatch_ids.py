"""
Cross-match ID Enrichment
=========================
Fetches tmass_id and source_id (Gaia DR4) for the GALAH stars already
in galah_dr4_allstar.csv, using DataCentral TAP.

Strategy: query the full DR4 table once (no IN clause) and join locally.
This avoids HTTP 400 from oversized IN(...) clauses.

Usage:
    python scripts/enrich_crossmatch_ids.py
"""

import os
import sys
import csv
import time
import io
import requests
from datetime import datetime

BASE_DIR  = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GALAH_CSV = os.path.join(BASE_DIR, "data", "galah", "raw", "galah_dr4_allstar.csv")
BACKUP_CSV= os.path.join(BASE_DIR, "data", "galah", "raw", "galah_dr4_allstar_backup.csv")
TAP_URL   = "https://datacentral.org.au/vo/tap/sync"


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — probe: check what columns exist in the table
# ─────────────────────────────────────────────────────────────────────────────

def probe_columns():
    """Fetch one row to see available column names."""
    query = "SELECT TOP 1 * FROM galah_dr4.mainstartable"
    r = requests.post(
        TAP_URL,
        data={"REQUEST": "doQuery", "LANG": "ADQL",
              "FORMAT": "csv", "QUERY": query},
        timeout=60,
    )
    if r.status_code != 200:
        raise RuntimeError(f"Probe failed HTTP {r.status_code}:\n{r.text[:400]}")
    lines = [l for l in r.text.strip().splitlines() if l.strip()]
    if not lines:
        raise RuntimeError("Probe returned empty result")
    headers = lines[0].split(",")
    print(f"  Available columns: {headers}")
    return headers


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — fetch id columns for ALL stars in batches by sobject_id range
# ─────────────────────────────────────────────────────────────────────────────

def fetch_all_ids(sobject_ids, tmass_col, source_col):
    """
    Fetches tmass_col and source_col from TAP by splitting sobject_ids
    into BETWEEN ranges (much shorter query than IN).

    Returns dict: sobject_id (int) -> {"tmass_id": str, "source_id": str}
    """
    if not tmass_col and not source_col:
        print("  Neither tmass nor source_id column found. Nothing to fetch.")
        return {}

    cols = "sobject_id"
    if tmass_col:  cols += f", {tmass_col}"
    if source_col: cols += f", {source_col}"

    sorted_ids = sorted(sobject_ids)
    BATCH      = 5000   # BETWEEN range covers 5000 IDs at a time
    result     = {}
    total      = (len(sorted_ids) + BATCH - 1) // BATCH

    print(f"  Fetching {len(sorted_ids):,} IDs in {total} range-batches of ~{BATCH}...")

    for i in range(0, len(sorted_ids), BATCH):
        chunk     = sorted_ids[i : i + BATCH]
        id_min    = chunk[0]
        id_max    = chunk[-1]
        batch_num = i // BATCH + 1

        query = (
            f"SELECT {cols} "
            f"FROM galah_dr4.mainstartable "
            f"WHERE sobject_id BETWEEN {id_min} AND {id_max}"
        )

        for attempt in range(1, 4):
            try:
                r = requests.post(
                    TAP_URL,
                    data={"REQUEST": "doQuery", "LANG": "ADQL",
                          "FORMAT": "csv", "QUERY": query},
                    timeout=90,
                )
                if r.status_code != 200:
                    raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")

                lines = [l for l in r.text.strip().splitlines() if l.strip()]
                if len(lines) < 2:
                    break   # empty range

                reader = csv.DictReader(lines)
                for row in reader:
                    sid_raw = row.get("sobject_id", "").strip()
                    if not sid_raw:
                        continue
                    try:
                        sid = int(float(sid_raw))
                    except ValueError:
                        continue
                    result[sid] = {
                        "tmass_id":  row.get(tmass_col,  "").strip() if tmass_col  else "",
                        "source_id": row.get(source_col, "").strip() if source_col else "",
                    }
                break   # success

            except Exception as e:
                if attempt < 3:
                    print(f"    Batch {batch_num} attempt {attempt} failed ({e}), retrying...")
                    time.sleep(5)
                else:
                    print(f"    Batch {batch_num} failed permanently: {e}")

        print(f"  Batch {batch_num:>4}/{total}  "
              f"(sobject_id {id_min}–{id_max})  "
              f"→ {len(result):,} total fetched so far")
        time.sleep(0.5)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    sep = "=" * 65
    print(sep)
    print("  GALAH ID Enrichment  (tmass_id + source_id)")
    print(sep)
    print(f"  Generated : {datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}")

    if not os.path.exists(GALAH_CSV):
        print(f"\n  ERROR: {GALAH_CSV} not found.")
        sys.exit(1)

    # ── Read CSV ──────────────────────────────────────────────────────────────
    with open(GALAH_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    existing_cols = list(rows[0].keys()) if rows else []

    print(f"\n  Stars in CSV     : {len(rows):,}")
    print(f"  Existing columns : {existing_cols}")

    # Short-circuit if already enriched
    if "tmass_id" in existing_cols and "source_id" in existing_cols:
        already = sum(
            1 for r in rows
            if r.get("tmass_id", "").strip() not in ("", "None", "nan")
        )
        print(f"\n  tmass_id already present ({already:,} non-empty).")
        if already > len(rows) * 0.8:
            print("  CSV appears fully enriched. Nothing to do.")
            sys.exit(0)
        print("  Partially enriched — refetching gaps.")

    # ── Backup ────────────────────────────────────────────────────────────────
    import shutil
    shutil.copy2(GALAH_CSV, BACKUP_CSV)
    print(f"\n  Backup saved : {BACKUP_CSV}")

    # ── Probe TAP to find real column names ───────────────────────────────────
    print("\n  Probing DataCentral TAP for column names...")
    try:
        all_cols = probe_columns()
    except Exception as e:
        print(f"\n  ERROR probing TAP: {e}")
        print("  Check network connection.")
        sys.exit(1)

    # Detect 2MASS and Gaia columns
    all_cols_lower = [c.lower() for c in all_cols]

    tmass_col = None
    for candidate in ("tmass_id", "2mass_id", "tmassid", "twomass_id"):
        if candidate in all_cols_lower:
            tmass_col = all_cols[all_cols_lower.index(candidate)]
            break

    source_col = None
    for candidate in ("source_id", "gaia_source_id", "gaiaDR4_source_id",
                      "DR4_source_id", "gaia_DR4_source_id"):
        if candidate in all_cols_lower:
            source_col = all_cols[all_cols_lower.index(candidate)]
            break

    print(f"  Using 2MASS column  : '{tmass_col}'")
    print(f"  Using Gaia column   : '{source_col}'")

    if not tmass_col and not source_col:
        print("\n  ERROR: Neither tmass_id nor source_id found in DR4 table.")
        print(f"  Available columns: {all_cols}")
        sys.exit(1)

    # ── Collect sobject_ids ───────────────────────────────────────────────────
    sobject_ids = []
    for row in rows:
        try:
            sobject_ids.append(int(float(row["sobject_id"])))
        except (KeyError, ValueError):
            pass

    # ── Fetch from TAP ────────────────────────────────────────────────────────
    fetched = fetch_all_ids(sobject_ids, tmass_col, source_col)

    with_tmass  = sum(1 for v in fetched.values() if v["tmass_id"]  not in ("", "None", "nan"))
    with_source = sum(1 for v in fetched.values() if v["source_id"] not in ("", "None", "nan"))
    print(f"\n  Fetched entries    : {len(fetched):,}")
    print(f"  With tmass_id      : {with_tmass:,}")
    print(f"  With source_id     : {with_source:,}")

    # ── Write enriched CSV ────────────────────────────────────────────────────
    new_cols = [c for c in existing_cols if c not in ("tmass_id", "source_id")]
    new_cols += ["tmass_id", "source_id"]

    with open(GALAH_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=new_cols)
        writer.writeheader()
        for row in rows:
            try:
                sid = int(float(row["sobject_id"]))
            except (KeyError, ValueError):
                sid = None

            ids = fetched.get(sid, {"tmass_id": "", "source_id": ""}) \
                  if sid is not None else {"tmass_id": "", "source_id": ""}

            new_row = {c: row.get(c, "") for c in new_cols}
            new_row["tmass_id"]  = ids["tmass_id"]
            new_row["source_id"] = ids["source_id"]
            writer.writerow(new_row)

    print(f"\n  Enriched CSV saved : {GALAH_CSV}")
    print(f"\n  Next step: python scripts/check_crossmatch.py")
    print(f"\n{sep}")


if __name__ == "__main__":
    main()
