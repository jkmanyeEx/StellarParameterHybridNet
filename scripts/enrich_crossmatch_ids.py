"""
Cross-match ID Enrichment
=========================
Fetches tmass_id and source_id (Gaia DR3) for the GALAH stars already
in galah_dr4_allstar.csv, using DataCentral TAP in batches.

APOGEE needs no enrichment — apogee_id IS the 2MASS ID (2M{RA}{Dec}).
GALAH-APOGEE matching is then done via 2MASS ID.

Usage:
    python scripts/enrich_crossmatch_ids.py

Output:
    data/galah/raw/galah_dr4_allstar.csv   — original CSV + tmass_id + source_id columns added
    (original is backed up to galah_dr4_allstar_backup.csv before modification)
"""

import os
import sys
import csv
import time
import requests
from datetime import datetime

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

GALAH_CSV    = os.path.join(BASE_DIR, "data", "galah", "raw", "galah_dr4_allstar.csv")
BACKUP_CSV   = os.path.join(BASE_DIR, "data", "galah", "raw", "galah_dr4_allstar_backup.csv")
TAP_URL      = "https://datacentral.org.au/vo/tap/sync"
BATCH_SIZE   = 1000    # TAP IN-clause limit; keep ≤ 1000 to avoid query size issues
RETRY_MAX    = 3
RETRY_DELAY  = 5       # seconds between retries


# ─────────────────────────────────────────────────────────────────────────────
# TAP batch query
# ─────────────────────────────────────────────────────────────────────────────

def fetch_ids_batch(sobject_ids):
    """
    Query DataCentral TAP for tmass_id and source_id for a batch of sobject_ids.
    Returns dict: sobject_id (int) -> {"tmass_id": str, "source_id": str}
    """
    id_list = ", ".join(str(s) for s in sobject_ids)
    query = f"""
    SELECT sobject_id, tmass_id, source_id
    FROM galah_dr4.mainstartable
    WHERE sobject_id IN ({id_list})
    """

    for attempt in range(1, RETRY_MAX + 1):
        try:
            r = requests.post(
                TAP_URL,
                data={"REQUEST": "doQuery", "LANG": "ADQL",
                      "FORMAT": "csv", "QUERY": query},
                timeout=60,
            )
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")

            lines = [l for l in r.text.strip().splitlines() if l.strip()]
            if len(lines) < 2:
                return {}   # empty result

            reader = csv.DictReader(lines)
            result = {}
            for row in reader:
                sid = row.get("sobject_id", "").strip()
                if not sid:
                    continue
                result[int(float(sid))] = {
                    "tmass_id":  row.get("tmass_id",  "").strip(),
                    "source_id": row.get("source_id", "").strip(),
                }
            return result

        except Exception as e:
            if attempt < RETRY_MAX:
                print(f"    Attempt {attempt} failed ({e}), retrying in {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)
            else:
                print(f"    Batch failed after {RETRY_MAX} attempts: {e}")
                return {}


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

    # ── Read existing CSV ─────────────────────────────────────────────────────
    with open(GALAH_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    existing_cols = list(rows[0].keys()) if rows else []

    print(f"\n  Stars in CSV       : {len(rows):,}")
    print(f"  Existing columns   : {existing_cols}")

    # Check if already enriched
    if "tmass_id" in existing_cols and "source_id" in existing_cols:
        already = sum(
            1 for r in rows
            if r.get("tmass_id", "").strip() not in ("", "None")
        )
        print(f"\n  tmass_id already present in CSV ({already:,} non-empty).")
        if already > len(rows) * 0.8:
            print("  CSV appears fully enriched. Nothing to do.")
            sys.exit(0)
        else:
            print("  Only partially enriched — continuing to fill gaps.")

    # ── Backup ────────────────────────────────────────────────────────────────
    import shutil
    shutil.copy2(GALAH_CSV, BACKUP_CSV)
    print(f"\n  Backup saved       : {BACKUP_CSV}")

    # ── Collect sobject_ids that need enrichment ───────────────────────────────
    needs_enrichment = []
    existing_map = {}   # sobject_id (int) -> {"tmass_id", "source_id"}

    for row in rows:
        sid_raw = row.get("sobject_id", "").strip()
        if not sid_raw:
            continue
        try:
            sid = int(float(sid_raw))
        except ValueError:
            continue

        tmass  = row.get("tmass_id",  "").strip()
        source = row.get("source_id", "").strip()

        if tmass in ("", "None", "nan") or source in ("", "None", "nan"):
            needs_enrichment.append(sid)
        else:
            existing_map[sid] = {"tmass_id": tmass, "source_id": source}

    print(f"  Already have IDs   : {len(existing_map):,}")
    print(f"  Need fetching      : {len(needs_enrichment):,}")

    if not needs_enrichment:
        print("\n  All stars already have tmass_id and source_id. Nothing to do.")
        sys.exit(0)

    # ── Batch query TAP ───────────────────────────────────────────────────────
    print(f"\n  Fetching in batches of {BATCH_SIZE}...")
    fetched_map = {}
    total_batches = (len(needs_enrichment) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(needs_enrichment), BATCH_SIZE):
        batch = needs_enrichment[i : i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        print(f"  Batch {batch_num:>4}/{total_batches}  ({len(batch)} stars)...", end="", flush=True)
        result = fetch_ids_batch(batch)
        fetched_map.update(result)
        print(f"  got {len(result)} results")
        time.sleep(0.3)   # polite rate limit

    # Merge
    full_map = {**existing_map, **fetched_map}

    # Stats
    with_tmass  = sum(1 for v in full_map.values() if v["tmass_id"]  not in ("", "None", "nan", ""))
    with_source = sum(1 for v in full_map.values() if v["source_id"] not in ("", "None", "nan", ""))
    print(f"\n  Stars with tmass_id  : {with_tmass:,} / {len(rows):,}")
    print(f"  Stars with source_id : {with_source:,} / {len(rows):,}")

    # ── Write enriched CSV ────────────────────────────────────────────────────
    new_cols = [c for c in existing_cols if c not in ("tmass_id", "source_id")]
    new_cols += ["tmass_id", "source_id"]

    with open(GALAH_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=new_cols)
        writer.writeheader()
        for row in rows:
            sid_raw = row.get("sobject_id", "").strip()
            try:
                sid = int(float(sid_raw))
            except ValueError:
                sid = None

            ids = full_map.get(sid, {"tmass_id": "", "source_id": ""}) if sid else \
                  {"tmass_id": row.get("tmass_id", ""), "source_id": row.get("source_id", "")}

            new_row = {c: row.get(c, "") for c in new_cols}
            new_row["tmass_id"]  = ids["tmass_id"]
            new_row["source_id"] = ids["source_id"]
            writer.writerow(new_row)

    print(f"\n  Enriched CSV saved : {GALAH_CSV}")
    print(f"\n  Next step: python scripts/check_crossmatch.py")
    print(f"\n{sep}")


if __name__ == "__main__":
    main()
