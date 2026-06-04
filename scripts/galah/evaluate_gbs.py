"""
evaluate_gbs_galah.py
=====================
Evaluate the GALAH HybridNet (StellarParameterHybridNet) on
Gaia FGK Benchmark Stars v3 (Soubiran+ 2024).

The GBS provides Teff and logg through fundamental relations (angular diameter
+ parallax + bolometric flux) — fully independent of spectroscopy — making
them the most trustworthy external anchors available for validation.

What this script does
---------------------
1. Load GBS v3 parameter catalogue  (data/gbs/gbs_v3_params.fits)
2. For each GBS star that has a GALAH DR4 match (via HIP / 2MASS cross-ID):
   a. Load the corresponding GALAH FITS spectrum from the 4-arm CCD files
   b. Continuum-normalise and resample onto the model's internal wavelength grid
   c. Extract 45D physical line-feature vector (same pipeline used in training)
   d. Run GALAH HybridNet inference
3. Compare predictions to GBS fundamental Teff / logg and spectroscopic [Fe/H]
4. Print a structured report and save results/gbs_galah_eval.csv

Usage
-----
    python evaluate_gbs_galah.py \\
        --weights  weights/galah/stellar_hybrid_model_n41386.pth \\
        --gbs-cat  data/gbs/gbs_v3_params.fits \\
        --galah-dir data/galah/raw/spectra \\
        --outdir   results/gbs

Directory layout expected
-------------------------
data/galah/processed/
    label_stats.npy      shape (2, 3)  — [[Tmean,gm,Fm],[Tstd,gstd,Fstd]]
    feature_stats.npy    shape (2, 45) — [[mean45],[std45]]

The GALAH spectrum FITS files are assumed to follow the standard DR4 naming:
    <galah-dir>/DR4/<ccd>/<sobject_id>.fits   (ccd = 1,2,3,4)
OR the flat layout used in some local mirrors:
    <galah-dir>/<sobject_id><ccd>.fits
"""

import argparse
import csv
import os
import sys
import warnings
from datetime import datetime

import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)

# ── Project root on sys.path (adjust depth to match your layout) ──────────────
_script_dir  = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_script_dir, os.pardir, os.pardir))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# ── Survey-specific constants ─────────────────────────────────────────────────
# GALAH HERMES 4-arm wavelength ranges (Å)
GALAH_ARM_RANGES = [
    (4713, 4903),   # CCD1 — blue (Hβ region)
    (5648, 5873),   # CCD2 — green (Mg-b region)
    (6478, 6737),   # CCD3 — red  (Hα region)
    (7585, 7887),   # CCD4 — NIR  (Ca II triplet edge)
]
GALAH_NPIX_PER_ARM = 4000        # internal model grid
GALAH_N_ARMS       = 4
GALAH_N_FEATURES   = 45          # 15 lines × 3 descriptors

# Absorption lines used for 45D feature extraction (Å, CCD index 0-based)
# Each entry: (central_wavelength, arm_index, label)
GALAH_LINE_TABLE = [
    (4861.3, 0, "H_beta"),
    (4921.9, 0, "Fe_I_4882"),    # approx; adjust to your extractor's list
    (4703.0, 0, "Mg_I_4703"),
    (5711.0, 1, "Mg_I_5711"),
    (5782.0, 1, "Fe_I_5782"),
    (5862.0, 1, "Fe_I_5862"),
    (5662.0, 1, "Fe_I_5662"),
    (6562.8, 2, "H_alpha"),
    (6495.0, 2, "Fe_I_6495"),
    (6499.0, 2, "Ca_I_6499"),
    (7699.0, 3, "K_I_7699"),
    (7772.0, 3, "O_I_7772"),
    (7748.0, 3, "Fe_I_7748"),
    (6708.0, 2, "Li_I_6708"),
    (4897.0, 0, "Ba_II_4897"),
]

PARAM_NAMES  = ["T_eff (K)", "log g (dex)", "[Fe/H] (dex)"]
PARAM_UNITS  = ["K",         "dex",          "dex"]


# ── Model import ─────────────────────────────────────────────────────────────

def _load_model(weights_path: str, device):
    """Import and load the GALAH HybridNet from the project source tree."""
    try:
        from src.models.galah.hybrid_net import StellarParameterHybridNet
    except ImportError:
        raise ImportError(
            "Could not import StellarParameterHybridNet from src.models.galah.hybrid_net.\n"
            "Ensure this script is run from the project root and src/ is on sys.path."
        )
    import torch
    model = StellarParameterHybridNet()
    ckpt  = torch.load(weights_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt.get("model_state", ckpt))
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    print(f"  ✓  GALAH model loaded from {weights_path}")
    return model


# ── Normalisation stats ───────────────────────────────────────────────────────

def _load_norm_stats(proc_dir: str):
    label_path   = os.path.join(proc_dir, "label_stats.npy")
    feature_path = os.path.join(proc_dir, "feature_stats.npy")
    for p in [label_path, feature_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Normalisation stats not found: {p}\n"
                "Run the GALAH preprocessing pipeline first."
            )
    ls = np.load(label_path)      # (2, 3)
    fs = np.load(feature_path)    # (2, 45)
    label_mean,   label_std   = ls[0].astype(np.float32), ls[1].astype(np.float32)
    feature_mean, feature_std = fs[0].astype(np.float32), fs[1].astype(np.float32)
    return label_mean, label_std, feature_mean, feature_std


# ── GBS catalogue loader ──────────────────────────────────────────────────────

def load_gbs_catalogue(fits_path: str) -> list[dict]:
    """
    Load GBS v3 from FITS; return list of dicts with keys:
        hip_id, teff_gbs, logg_gbs, feh_gbs, teff_err, logg_err
    Stars with missing Teff or logg are excluded.
    """
    from astropy.io import fits as afits
    import numpy as np

    stars = []
    with afits.open(fits_path) as hdul:
        # Find the data extension (usually index 1)
        for ext in hdul[1:]:
            if ext.data is not None:
                tbl = ext.data
                break
        else:
            raise ValueError("No data extension found in GBS FITS file.")

        names = [n.upper() for n in tbl.dtype.names]

        def _col(candidates):
            for c in candidates:
                if c.upper() in names:
                    return tbl[tbl.dtype.names[names.index(c.upper())]]
            return None

        hip    = _col(["HIP", "HIPID", "HIP_ID"])
        gaia   = _col(["GAIADR3", "GAIA_DR3", "SOURCE_ID"])
        teff   = _col(["TEFF", "T_EFF", "TEFF_GBS"])
        logg   = _col(["LOGG", "LOG_G", "LOGG_GBS"])
        feh    = _col(["__FE_H_", "FEH", "FE_H", "[FE/H]", "MET"])
        e_teff = _col(["E_TEFF", "ETEFF", "ERR_TEFF"])
        e_logg = _col(["E_LOGG", "ELOGG", "ERR_LOGG"])

        for i in range(len(tbl)):
            t = float(teff[i])   if teff  is not None else np.nan
            g = float(logg[i])   if logg  is not None else np.nan
            f = float(feh[i])    if feh   is not None else np.nan
            try:
                h = int("".join(c for c in str(hip[i]) if c.isdigit())) if hip is not None else -1
            except (ValueError, TypeError):
                h = -1
            try:
                gaia_id = int(str(gaia[i]).strip()) if gaia is not None else -1
            except (ValueError, TypeError):
                gaia_id = -1

            if not (np.isfinite(t) and np.isfinite(g)):
                continue   # skip stars without fundamental params
            stars.append({
                "hip_id":    h,
                "gaia_dr3":  gaia_id,
                "teff_gbs":  t,
                "logg_gbs":  g,
                "feh_gbs":   f,
                "teff_err":  float(e_teff[i]) if e_teff is not None else np.nan,
                "logg_err":  float(e_logg[i]) if e_logg is not None else np.nan,
            })

    print(f"  ✓  GBS catalogue loaded — {len(stars)} stars with Teff & logg")
    return stars


# ── GALAH spectrum loader ─────────────────────────────────────────────────────

def _find_gbs_fits(gbs_dir: str, hip_id: int) -> str | None:
    """
    Locate the GBS spectrum for a given HIP ID (e.g., HIP101345_HARPS_1_R42KNorm.fits).
    Returns the first matching path, or None.
    """
    import glob
    pattern = os.path.join(gbs_dir, f"HIP{hip_id}_*_R42KNorm.fits")
    matches = glob.glob(pattern)
    if matches:
        return matches[0] # Just use the first available instrument for now
    return None

def load_gbs_spectrum_as_galah(fits_path: str) -> np.ndarray | None:
    """
    Load a GBS spectrum (R42KNorm) and resample it onto the GALAH 4-arm model grid.
    GBS spectra cover 4800-6800 A.
    CCD1 (4713-4903) will be partially padded with 1.0.
    CCD4 (7585-7887) will be entirely filled with 1.0.
    Returns array of shape (4, GALAH_NPIX_PER_ARM).
    """
    from astropy.io import fits as afits
    from scipy.interpolate import interp1d

    try:
        with afits.open(fits_path) as hdul:
            ext = hdul[0] if hdul[0].data is not None else hdul[1]
            flux = ext.data.astype(np.float32).ravel()
            hdr = ext.header
            crval = hdr.get("CRVAL1")
            cdelt = hdr.get("CDELT1", hdr.get("CD1_1"))
            if crval is None or cdelt is None:
                return None
            wave_obs = crval + cdelt * np.arange(len(flux))
    except Exception:
        return None

    result = np.ones((GALAH_N_ARMS, GALAH_NPIX_PER_ARM), dtype=np.float32)

    for arm_idx, (wmin, wmax) in enumerate(GALAH_ARM_RANGES):
        wave_grid = np.linspace(wmin, wmax, GALAH_NPIX_PER_ARM)
        
        # Check if arm has any overlap with GBS range (approx 4800-6800)
        overlap_min = max(wmin, wave_obs[0])
        overlap_max = min(wmax, wave_obs[-1])
        
        if overlap_min < overlap_max:
            interp = interp1d(wave_obs, flux, kind="linear",
                              bounds_error=False, fill_value=1.0)
            result[arm_idx] = interp(wave_grid).astype(np.float32)
            
    return result


# ── Feature extraction ────────────────────────────────────────────────────────

def _gaussian(x, amp, mu, sigma):
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def _fit_line(wave_arm: np.ndarray, flux_arm: np.ndarray,
              center: float, half_width: float = 8.0) -> tuple[float, float, float]:
    """
    Fit a Gaussian absorption profile to a single spectral line.
    Returns (equivalent_width, fwhm, depth).
    """
    from scipy.optimize import curve_fit

    mask = np.abs(wave_arm - center) < half_width
    if mask.sum() < 5:
        return 0.0, 0.0, 0.0

    w_seg = wave_arm[mask]
    f_seg = flux_arm[mask]
    depth = max(0.0, 1.0 - float(np.nanmin(f_seg)))
    sigma_init = half_width / 3.0

    try:
        popt, _ = curve_fit(
            lambda x, amp, mu, sig: 1.0 - _gaussian(x, amp, mu, sig),
            w_seg, f_seg,
            p0=[depth, center, sigma_init],
            bounds=([0, center - half_width, 0.5],
                    [2.0, center + half_width, half_width]),
            maxfev=400,
        )
        amp, mu, sig = popt
        fwhm = 2.355 * abs(sig)
        ew   = amp * abs(sig) * np.sqrt(2 * np.pi)
        return float(ew), float(fwhm), float(amp)
    except Exception:
        return 0.0, 0.0, float(depth)


def extract_galah_features(flux_4arm: np.ndarray) -> np.ndarray:
    """
    Extract 45D feature vector from a (4, GALAH_NPIX_PER_ARM) flux array.
    Features: [ew, fwhm, depth] × 15 lines = 45D
    """
    features = np.zeros(GALAH_N_FEATURES, dtype=np.float32)
    for line_idx, (center, arm_idx, _) in enumerate(GALAH_LINE_TABLE):
        wmin, wmax = GALAH_ARM_RANGES[arm_idx]
        wave_arm   = np.linspace(wmin, wmax, GALAH_NPIX_PER_ARM)
        flux_arm   = flux_4arm[arm_idx]
        ew, fwhm, depth = _fit_line(wave_arm, flux_arm, center)
        features[line_idx * 3]     = ew
        features[line_idx * 3 + 1] = fwhm
        features[line_idx * 3 + 2] = depth
    return features


# ── GBS × GALAH cross-match ───────────────────────────────────────────────────

def crossmatch_gbs_galah(gbs_stars: list[dict], galah_cat: str) -> list[dict]:
    """
    Cross-match GBS stars against GALAH DR4 catalogue via Gaia DR3 source_id.
    Adds 'sobject_id' to each matched GBS entry.
    Returns only matched entries.
    """
    if not os.path.exists(galah_cat):
        print(f"  [WARN] GALAH catalogue not found at {galah_cat}.")
        print("         Cannot cross-match.")
        return []

    ext = os.path.splitext(galah_cat)[1].lower()
    gaia_to_sobject: dict[int, str] = {}   # gaiadr3_source_id → sobject_id

    if ext in (".fits", ".fit"):
        from astropy.io import fits as afits
        print("  Loading GALAH catalogue (FITS) …")
        with afits.open(galah_cat, memmap=True) as hdul:
            for ext_obj in hdul[1:]:
                if ext_obj.data is None:
                    continue
                col_names = ext_obj.data.dtype.names
                gaia_col = next((n for n in col_names
                                 if "gaia" in n.lower() and "source" in n.lower()), None)
                sob_col  = next((n for n in col_names
                                 if "sobject" in n.lower()), None)
                if gaia_col and sob_col:
                    for row in ext_obj.data:
                        try:
                            gid = int(row[gaia_col])
                            if gid > 0:
                                gaia_to_sobject[gid] = str(row[sob_col]).strip()
                        except Exception:
                            pass
                break

    elif ext == ".csv":
        print("  Loading GALAH catalogue (CSV) …")
        with open(galah_cat, newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                gaia_key = next((k for k in row if "gaia" in k.lower() and "source" in k.lower()), None)
                sob_key  = next((k for k in row if "sobject" in k.lower()), None)
                if gaia_key and sob_key:
                    try:
                        gid = int(float(row[gaia_key]))
                        if gid > 0:
                            gaia_to_sobject[gid] = row[sob_key].strip()
                    except (ValueError, TypeError):
                        pass

    print(f"  GALAH catalogue: {len(gaia_to_sobject)} entries with Gaia DR3 IDs")

    matched = []
    for star in gbs_stars:
        gaia_id = star.get("gaia_dr3", -1)
        sid = gaia_to_sobject.get(gaia_id)
        if sid:
            matched.append({**star, "sobject_id": sid})

    print(f"  ✓  Cross-match: {len(matched)} / {len(gbs_stars)} GBS stars found in GALAH")
    return matched


# ── Metrics ───────────────────────────────────────────────────────────────────

def _metrics(y_true: np.ndarray, y_pred: np.ndarray):
    diff = y_true - y_pred
    mae  = np.mean(np.abs(diff), axis=0)
    rmse = np.sqrt(np.mean(diff ** 2, axis=0))
    ss_res = np.sum(diff ** 2, axis=0)
    ss_tot = np.sum((y_true - y_true.mean(axis=0)) ** 2, axis=0)
    r2   = 1.0 - ss_res / (ss_tot + 1e-8)

    rel = np.zeros(3)
    for i, safe_thresh in enumerate([0.0, 0.1, 0.01]):
        mask = np.abs(y_true[:, i]) > safe_thresh
        if mask.sum() > 0:
            rel[i] = np.mean(np.abs(diff[mask, i]) / (np.abs(y_true[mask, i]) + 1e-8)) * 100

    return mae, rmse, r2, rel


# ── Main evaluation loop ──────────────────────────────────────────────────────

def run_evaluation(args):
    import torch

    device = torch.device(
        "mps"  if torch.backends.mps.is_available()  else
        "cuda" if torch.cuda.is_available()           else "cpu"
    )

    print("\n" + "=" * 70)
    print("  GBS v3 Evaluation — GALAH HybridNet")
    print(f"  Generated : {datetime.now():%Y-%m-%d  %H:%M:%S}")
    print("=" * 70)
    print(f"  Device    : {device}")

    # -- Load normalisation stats
    proc_dir = os.path.join(args.galah_processed)
    label_mean, label_std, feature_mean, feature_std = _load_norm_stats(proc_dir)

    # -- Load model
    model = _load_model(args.weights, device)

    # -- Load GBS catalogue
    gbs_stars = load_gbs_catalogue(args.gbs_cat)

    print(f"\n  Running inference on {len(gbs_stars)} GBS stars directly …")

    records      = []
    preds_list   = []
    labels_list  = []
    skipped      = 0

    for star in gbs_stars:
        fits_path = _find_gbs_fits(args.gbs_spectra, star["hip_id"])
        if fits_path is None:
            skipped += 1
            continue

        flux_4arm = load_gbs_spectrum_as_galah(fits_path)
        if flux_4arm is None:
            skipped += 1
            continue

        # Feature extraction
        try:
            features = extract_galah_features(flux_4arm)
        except Exception:
            features = np.zeros(GALAH_N_FEATURES, dtype=np.float32)

        # Normalise
        flux_t    = torch.tensor(flux_4arm, dtype=torch.float32).unsqueeze(0).to(device)
        feat_norm = (features - feature_mean) / (feature_std + 1e-8)
        feat_t    = torch.tensor(feat_norm, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            pred_norm = model(flux_t, feat_t).cpu().numpy()[0]   # (3,)

        pred = pred_norm * label_std + label_mean   # denormalise

        # Ground truth (GBS fundamental + spectroscopic [Fe/H])
        gbs_label = np.array([
            star["teff_gbs"],
            star["logg_gbs"],
            star.get("feh_gbs", np.nan),
        ], dtype=np.float32)

        preds_list.append(pred)
        labels_list.append(gbs_label)
        records.append({
            "hip_id":       star["hip_id"],
            "teff_gbs":     star["teff_gbs"],
            "logg_gbs":     star["logg_gbs"],
            "feh_gbs":      star.get("feh_gbs", np.nan),
            "teff_pred":    float(pred[0]),
            "logg_pred":    float(pred[1]),
            "feh_pred":     float(pred[2]),
            "teff_err_gbs": star.get("teff_err", np.nan),
            "logg_err_gbs": star.get("logg_err", np.nan),
        })

    if not records:
        print("\n  [ERROR] No stars could be evaluated. Check spectrum file paths.")
        sys.exit(1)

    preds  = np.array(preds_list,  dtype=np.float32)
    labels = np.array(labels_list, dtype=np.float32)

    # -- Metrics (skip [Fe/H] rows where GBS value is NaN)
    feh_valid = np.isfinite(labels[:, 2])
    mae, rmse, r2, rel = _metrics(labels, preds)

    # -- Report
    print(f"\n{'=' * 70}")
    print("  GBS v3 Evaluation Report — GALAH HybridNet")
    print(f"  Generated      : {datetime.now():%Y-%m-%d  %H:%M:%S}")
    print(f"  Weights        : {args.weights}")
    print(f"  Stars evaluated: {len(records)}   (skipped: {skipped})")
    print(f"{'=' * 70}")

    for i, (name, unit) in enumerate(zip(PARAM_NAMES, PARAM_UNITS)):
        if i == 2:
            n = int(feh_valid.sum())
            mae_i  = float(np.mean(np.abs(labels[feh_valid, 2] - preds[feh_valid, 2])))
            rmse_i = float(np.sqrt(np.mean((labels[feh_valid, 2] - preds[feh_valid, 2]) ** 2)))
            r2_i   = float(1.0 - np.sum((labels[feh_valid,2]-preds[feh_valid,2])**2) /
                           (np.sum((labels[feh_valid,2]-labels[feh_valid,2].mean())**2)+1e-8))
        else:
            n, mae_i, rmse_i, r2_i = len(records), float(mae[i]), float(rmse[i]), float(r2[i])

        print(f"\n   {name}  (n={n}):")
        print(f"     MAE            : {mae_i:.4f} {unit}")
        print(f"     RMSE           : {rmse_i:.4f} {unit}")
        print(f"     Relative Error : {float(rel[i]):.2f}%")
        print(f"     R2 Score       : {r2_i:.4f}")

    print(f"\n   Note: Teff and logg from GBS are FUNDAMENTAL (spectroscopy-independent).")
    print(f"         [Fe/H] is spectroscopic (from high-res optical spectra).")

    print(f"\n{'=' * 70}\n")

    # -- Save CSV
    os.makedirs(args.outdir, exist_ok=True)
    csv_path = os.path.join(args.outdir, "gbs_galah_eval.csv")
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    print(f"  Results saved → {csv_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate GALAH HybridNet on Gaia FGK Benchmark Stars v3"
    )
    parser.add_argument("--weights",
        default="weights/galah/stellar_hybrid_model_n41386.pth",
        help="Path to GALAH model weights (.pth)")
    parser.add_argument("--gbs-cat",
        default="data/gbs/gbs_v3_params.fits",
        help="GBS v3 parameter catalogue FITS (from download_gbs.py)")
    parser.add_argument("--gbs-spectra",
        default="data/gbs/spectra",
        help="Root directory containing GBS spectra (e.g. data/gbs/spectra)")
    parser.add_argument("--galah-processed",
        default="data/galah/processed",
        help="Directory containing label_stats.npy and feature_stats.npy")
    parser.add_argument("--outdir",
        default="results/gbs",
        help="Output directory for evaluation CSV")
    args = parser.parse_args()

    for p, label in [
        (args.weights,        "GALAH weights"),
        (args.gbs_cat,        "GBS catalogue"),
        (args.gbs_spectra,    "GBS spectra dir"),
        (args.galah_processed,"GALAH processed dir"),
    ]:
        if not os.path.exists(p):
            print(f"[ERROR] {label} not found: {p}")
            sys.exit(1)

    run_evaluation(args)


if __name__ == "__main__":
    main()