"""
GALAH x APOGEE Cross-Survey Validation
=======================================
For each star in the CV set (crossmatch_cv_set.csv), runs BOTH models
independently and compares their predictions against each survey's own
ground-truth labels.

What this measures:
  - GALAH model accuracy on cross-matched stars   (vs GALAH labels)
  - APOGEE model accuracy on cross-matched stars  (vs APOGEE labels)
  - Inter-model agreement: how consistently the two models predict
    T_eff / log g / [Fe/H] for the SAME physical star

Usage:
    python scripts/run_crossvalidation.py

Prerequisites:
    1. scripts/check_crossmatch.py must have been run
       → produces data/crossmatch_cv_set.csv
    2. Both GALAH and APOGEE training pipelines must be complete
       → weights/galah/stellar_hybrid_model.pth
       → weights/apogee/stellar_hybrid_model.pth
       → data/galah/processed/  (X_flux_clean.npy, X_features_physical.npy,
                                  Y_labels.npy, star_ids.npy, label_stats.npy,
                                  feature_stats.npy, standard_wave.npy)
       → data/apogee/processed/ (same set)
"""

import os
import sys
import csv
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm

# ── Project root ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

from src.models.galah.hybrid_net  import StellarParameterHybridNet as GalahNet
from src.models.apogee.hybrid_net import StellarParameterHybridNet as ApogeeNet
from src.data.galah.extract_features  import extract_45d_features_single_star
from src.data.apogee.extract_features import extract_30d_features_single_star
from src.utils.galah.config  import DEVICE as GALAH_DEVICE
from src.utils.apogee.config import DEVICE as APOGEE_DEVICE

# ── Paths ─────────────────────────────────────────────────────────────────────
CV_SET_CSV   = os.path.join(BASE_DIR, "data", "crossmatch_cv_set.csv")

GALAH_PROC   = os.path.join(BASE_DIR, "data", "galah",  "processed")
APOGEE_PROC  = os.path.join(BASE_DIR, "data", "apogee", "processed")

GALAH_WEIGHTS  = os.path.join(BASE_DIR, "weights", "galah",  "stellar_hybrid_model.pth")
APOGEE_WEIGHTS = os.path.join(BASE_DIR, "weights", "apogee", "stellar_hybrid_model.pth")

REPORT_DIR   = os.path.join(BASE_DIR, "report", "crossmatch")
REPORT_PATH  = os.path.join(REPORT_DIR, "crossvalidation_report.txt")

PARAMETERS   = ["T_eff",  "log g",  "[Fe/H]"]
UNITS        = ["K",      "dex",    "dex"]


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred):
    """Returns mae, rmse, r2, rel_err arrays of shape (3,)."""
    mae  = np.mean(np.abs(y_true - y_pred), axis=0)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))

    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    ss_tot = np.sum((y_true - np.mean(y_true, axis=0)) ** 2, axis=0)
    r2     = 1.0 - ss_res / (ss_tot + 1e-8)

    rel = np.zeros(3)
    # T_eff
    rel[0] = np.mean(np.abs(y_true[:, 0] - y_pred[:, 0])
                     / (np.abs(y_true[:, 0]) + 1e-8)) * 100
    # log g — only where |true| > 0.1
    mask_g = np.abs(y_true[:, 1]) > 0.1
    if mask_g.any():
        rel[1] = np.mean(np.abs(y_true[mask_g, 1] - y_pred[mask_g, 1])
                         / np.abs(y_true[mask_g, 1])) * 100
    # [Fe/H] — only where |true| > 0.01
    mask_f = np.abs(y_true[:, 2]) > 0.01
    if mask_f.any():
        rel[2] = np.mean(np.abs(y_true[mask_f, 2] - y_pred[mask_f, 2])
                         / np.abs(y_true[mask_f, 2])) * 100

    return mae, rmse, r2, rel


def compute_agreement(preds_a, preds_b):
    """
    Inter-model agreement between two prediction arrays (N, 3).
    Returns mae, rmse arrays of shape (3,) — treating one model's
    predictions as 'truth' for the other.
    """
    diff = preds_a - preds_b
    mae  = np.mean(np.abs(diff), axis=0)
    rmse = np.sqrt(np.mean(diff ** 2, axis=0))
    return mae, rmse


# ─────────────────────────────────────────────────────────────────────────────
# Model loader
# ─────────────────────────────────────────────────────────────────────────────

def _load_model(net_class, weights_path, device):
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Weights not found: {weights_path}\n"
            "Run the training pipeline first."
        )
    model = net_class(use_features=True).to(device)
    ckpt  = torch.load(weights_path, map_location=device)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Per-survey data loader  (returns index lookup by star_id)
# ─────────────────────────────────────────────────────────────────────────────

def _load_survey_data(proc_dir):
    """
    Returns a dict with everything needed to run inference for one survey:
        flux        : np.ndarray  (N, arms, pixels)
        features    : np.ndarray  (N, D)
        labels      : np.ndarray  (N, 3)  — raw (un-normalised)
        star_ids    : np.ndarray  (N,)
        id_to_idx   : dict  star_id (str) -> row index
        label_mean  : np.ndarray (3,)
        label_std   : np.ndarray (3,)
        feat_mean   : np.ndarray (D,)
        feat_std    : np.ndarray (D,)
        wave        : np.ndarray  — standard_wave for this survey
    """
    def _req(fname):
        p = os.path.join(proc_dir, fname)
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Required file missing: {p}\n"
                "Run the survey data pipeline first."
            )
        return p

    flux     = np.load(_req("X_flux_clean.npy"),       mmap_mode="r")
    features = np.load(_req("X_features_physical.npy"), mmap_mode="r")
    labels   = np.load(_req("Y_labels.npy"))
    star_ids = np.load(_req("star_ids.npy"), allow_pickle=True)
    wave     = np.load(_req("standard_wave.npy"))

    ls = np.load(_req("label_stats.npy"))
    fs = np.load(_req("feature_stats.npy"))

    id_to_idx = {str(sid).strip(): i for i, sid in enumerate(star_ids)}

    return dict(
        flux       = flux,
        features   = features,
        labels     = labels,
        star_ids   = star_ids,
        id_to_idx  = id_to_idx,
        label_mean = ls[0].astype(np.float32),
        label_std  = ls[1].astype(np.float32),
        feat_mean  = fs[0].astype(np.float32),
        feat_std   = fs[1].astype(np.float32),
        wave       = wave,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Single-star inference
# ─────────────────────────────────────────────────────────────────────────────

def _predict_one(model, raw_flux, raw_feat, feat_mean, feat_std,
                 label_mean, label_std, device):
    """
    Runs one forward pass.
    raw_flux : (arms, pixels) — un-normalised
    raw_feat : (D,)           — un-normalised
    Returns real-space prediction np.ndarray (3,)
    """
    # Z-score flux per arm
    f_mean = np.mean(raw_flux, axis=1, keepdims=True)
    f_std  = np.std(raw_flux,  axis=1, keepdims=True) + 1e-8
    norm_flux = np.clip((raw_flux - f_mean) / f_std, -3.0, 3.0)

    # Normalise features
    norm_feat = (raw_feat - feat_mean) / (feat_std + 1e-8)

    t_flux = torch.from_numpy(norm_flux).float().unsqueeze(0).to(device)
    t_feat = torch.from_numpy(norm_feat).float().unsqueeze(0).to(device)

    with torch.no_grad():
        norm_pred = model(t_flux, t_feat).cpu().numpy()[0]

    return norm_pred * label_std + label_mean


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    sep = "=" * 70

    print(sep)
    print("  GALAH x APOGEE  —  Cross-Survey Validation")
    print(sep)

    # ── Check CV set ──────────────────────────────────────────────────────────
    if not os.path.exists(CV_SET_CSV):
        print(f"\n  ERROR: CV set not found at:\n    {CV_SET_CSV}")
        print("  Run  python scripts/check_crossmatch.py  first.")
        sys.exit(1)

    cv_rows = []
    with open(CV_SET_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cv_rows.append(row)

    if len(cv_rows) == 0:
        print("\n  CV set is empty. Re-run check_crossmatch.py.")
        sys.exit(1)

    print(f"\n  CV set stars : {len(cv_rows):,}")

    # ── Load survey data ──────────────────────────────────────────────────────
    print("\n  Loading GALAH processed data...")
    galah_data  = _load_survey_data(GALAH_PROC)
    print(f"  GALAH  flux shape : {galah_data['flux'].shape}")

    print("\n  Loading APOGEE processed data...")
    apogee_data = _load_survey_data(APOGEE_PROC)
    print(f"  APOGEE flux shape : {apogee_data['flux'].shape}")

    # ── Load models ───────────────────────────────────────────────────────────
    print("\n  Loading GALAH model ...")
    galah_model  = _load_model(GalahNet,  GALAH_WEIGHTS,  GALAH_DEVICE)
    print(f"  GALAH  model loaded  ({GALAH_WEIGHTS})")

    print("  Loading APOGEE model ...")
    apogee_model = _load_model(ApogeeNet, APOGEE_WEIGHTS, APOGEE_DEVICE)
    print(f"  APOGEE model loaded  ({APOGEE_WEIGHTS})")

    # ── Run inference ─────────────────────────────────────────────────────────
    galah_preds,  galah_truths  = [], []
    apogee_preds, apogee_truths = [], []
    skipped = []

    print(f"\n  Running inference on {len(cv_rows):,} cross-matched stars...")

    for row in tqdm(cv_rows, desc="Cross-validation", unit="star"):
        sobject_id = row["sobject_id"].strip()
        apogee_id  = row["apogee_id"].strip()

        # ── GALAH ──
        g_idx = galah_data["id_to_idx"].get(sobject_id)
        if g_idx is None:
            skipped.append((sobject_id, apogee_id, "sobject_id not in GALAH processed"))
            continue

        g_label = galah_data["labels"][g_idx]
        if np.any(g_label < -900):
            skipped.append((sobject_id, apogee_id, "GALAH label sentinel (-999)"))
            continue

        # ── APOGEE ──
        a_idx = apogee_data["id_to_idx"].get(apogee_id)
        if a_idx is None:
            skipped.append((sobject_id, apogee_id, "apogee_id not in APOGEE processed"))
            continue

        a_label = apogee_data["labels"][a_idx]
        if np.any(a_label < -900):
            skipped.append((sobject_id, apogee_id, "APOGEE label sentinel (-999)"))
            continue

        # ── GALAH prediction ──
        g_flux = np.array(galah_data["flux"][g_idx])      # (4, 4000)
        g_feat = np.array(galah_data["features"][g_idx])  # (45,)
        g_pred = _predict_one(
            galah_model, g_flux, g_feat,
            galah_data["feat_mean"], galah_data["feat_std"],
            galah_data["label_mean"], galah_data["label_std"],
            GALAH_DEVICE,
        )

        # ── APOGEE prediction ──
        a_flux = np.array(apogee_data["flux"][a_idx])      # (3, 2800)
        a_feat = np.array(apogee_data["features"][a_idx])  # (30,)
        a_pred = _predict_one(
            apogee_model, a_flux, a_feat,
            apogee_data["feat_mean"], apogee_data["feat_std"],
            apogee_data["label_mean"], apogee_data["label_std"],
            APOGEE_DEVICE,
        )

        galah_preds.append(g_pred)
        galah_truths.append(g_label)
        apogee_preds.append(a_pred)
        apogee_truths.append(a_label)

    n_valid = len(galah_preds)
    print(f"\n  Valid stars evaluated : {n_valid:,}")
    print(f"  Skipped               : {len(skipped):,}")

    if n_valid == 0:
        print("\n  No valid stars to evaluate. Check star_ids alignment.")
        sys.exit(1)

    galah_preds   = np.array(galah_preds)
    galah_truths  = np.array(galah_truths)
    apogee_preds  = np.array(apogee_preds)
    apogee_truths = np.array(apogee_truths)

    # ── Metrics ───────────────────────────────────────────────────────────────
    g_mae,  g_rmse,  g_r2,  g_rel  = compute_metrics(galah_truths,  galah_preds)
    a_mae,  a_rmse,  a_r2,  a_rel  = compute_metrics(apogee_truths, apogee_preds)
    ag_mae, ag_rmse = compute_agreement(galah_preds, apogee_preds)

    # ── Console print ─────────────────────────────────────────────────────────
    def _print_table(title, mae, rmse, r2, rel):
        print(f"\n  {title}")
        print(f"  {'-'*66}")
        print(f"  {'Parameter':<14} {'MAE':>10} {'RMSE':>10} {'Rel.Err':>10} {'R²':>10}")
        print(f"  {'-'*66}")
        for i in range(3):
            print(f"  {PARAMETERS[i]+' ('+UNITS[i]+')':14} "
                  f"{mae[i]:>10.3f} {rmse[i]:>10.3f} "
                  f"{rel[i]:>9.2f}% {r2[i]:>10.4f}")

    print(f"\n{sep}")
    print("  CROSS-SURVEY VALIDATION RESULTS")
    print(sep)
    _print_table("GALAH  model  vs  GALAH  labels  (in-survey, cross-matched stars)",
                 g_mae, g_rmse, g_r2, g_rel)
    _print_table("APOGEE model  vs  APOGEE labels  (in-survey, cross-matched stars)",
                 a_mae, a_rmse, a_r2, a_rel)

    print(f"\n  Inter-Model Agreement  (GALAH pred vs APOGEE pred, same physical star)")
    print(f"  {'-'*46}")
    print(f"  {'Parameter':<14} {'MAE':>10} {'RMSE':>10}")
    print(f"  {'-'*46}")
    for i in range(3):
        print(f"  {PARAMETERS[i]+' ('+UNITS[i]+')':14} "
              f"{ag_mae[i]:>10.3f} {ag_rmse[i]:>10.3f}")
    print(f"\n  (lower = two models agree more on the same star)")

    # ── Write report ──────────────────────────────────────────────────────────
    os.makedirs(REPORT_DIR, exist_ok=True)

    def _fmt_section(title, mae, rmse, r2, rel):
        lines = [f"   {title}", ""]
        for i in range(3):
            lines += [
                f"   * {PARAMETERS[i]} ({UNITS[i]}):",
                f"     MAE            : {mae[i]:.4f} {UNITS[i]}",
                f"     RMSE           : {rmse[i]:.4f} {UNITS[i]}",
                f"     Relative Error : {rel[i]:.2f}%",
                f"     R2 Score       : {r2[i]:.4f}",
                "",
            ]
        return lines

    report = [
        sep,
        "  GALAH x APOGEE  Cross-Survey Validation Report",
        sep,
        f"  Generated      : {datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}",
        f"  CV set size    : {len(cv_rows):,} stars",
        f"  Valid evaluated: {n_valid:,} stars",
        f"  Skipped        : {len(skipped):,} stars",
        "",
        "▶ [SECTION 1] GALAH Model — In-Survey Performance (cross-matched subset)",
        *_fmt_section("GALAH model vs GALAH labels", g_mae, g_rmse, g_r2, g_rel),
        "▶ [SECTION 2] APOGEE Model — In-Survey Performance (cross-matched subset)",
        *_fmt_section("APOGEE model vs APOGEE labels", a_mae, a_rmse, a_r2, a_rel),
        "▶ [SECTION 3] Inter-Model Agreement  (GALAH pred vs APOGEE pred)",
        "   Measures how consistently both models estimate parameters",
        "   for the SAME physical star observed in different wavelength regimes.",
        "",
    ]
    for i in range(3):
        report += [
            f"   * {PARAMETERS[i]} ({UNITS[i]}):",
            f"     Agreement MAE  : {ag_mae[i]:.4f} {UNITS[i]}",
            f"     Agreement RMSE : {ag_rmse[i]:.4f} {UNITS[i]}",
            "",
        ]

    if skipped:
        report += ["▶ [SECTION 4] Skipped Stars", ""]
        for sobject_id, apogee_id, reason in skipped[:20]:
            report.append(f"   sobject={sobject_id}  apogee={apogee_id}  reason={reason}")
        if len(skipped) > 20:
            report.append(f"   ... and {len(skipped) - 20} more")
        report.append("")

    report += [sep, "  END OF REPORT", sep]

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")

    print(f"\n  Report saved : {REPORT_PATH}")
    print(f"\n{sep}")


if __name__ == "__main__":
    main()
