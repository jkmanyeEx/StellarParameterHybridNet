"""
evaluate_gbs_apogee.py
======================
Evaluate the APOGEE HybridNet (StellarParameterHybridNet) on
Gaia FGK Benchmark Stars v3 (Soubiran+ 2024).

Because APOGEE operates in the near-infrared H-band (1.51–1.70 µm), while GBS
spectra are optical (480–680 nm), this script uses the APOGEE DR17 allStar
catalogue to cross-match GBS stars by 2MASS ID (embedded in APOGEE_ID as
"2M<2MASS>"), then loads the corresponding APOGEE combined spectrum FITS from
your local DR17 mirror.

What this script does
---------------------
1. Load GBS v3 parameter catalogue
2. Cross-match GBS stars to APOGEE DR17 via 2MASS ID
3. For each matched star:
   a. Load APOGEE combined spectrum (apStar / aspcapStar FITS)
   b. Continuum-normalise and resample onto 3-arm model grid
   c. Extract 30D physical feature vector (NIR H-band lines)
   d. Run APOGEE HybridNet inference
4. Compare to GBS fundamental Teff / logg and ASPCAP [Fe/H]
5. Print a structured report and save results/gbs_apogee_eval.csv

APOGEE arm grid (H-band, 3 pseudo-arms)
----------------------------------------
Arm 0: 15140–15800 Å
Arm 1: 15860–16430 Å
Arm 2: 16490–16960 Å
(Boundaries match the APOGEE chip gaps; adjust if your model uses different
 segmentation.)

Usage
-----
    python evaluate_gbs_apogee.py \\
        --weights  weights/apogee/stellar_hybrid_model.pth \\
        --gbs-cat  data/gbs/gbs_v3_params.fits \\
        --apogee-dir data/apogee/raw/spectra \\
        --apogee-catalogue data/apogee/raw/allStar-dr17.fits \\
        --outdir   results/gbs
"""

import argparse
import csv
import os
import sys
import warnings
from datetime import datetime

import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)

_script_dir   = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_script_dir, os.pardir, os.pardir))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# ── Survey-specific constants ─────────────────────────────────────────────────
# APOGEE H-band split into 3 pseudo-arms (chip boundaries)
APOGEE_ARM_RANGES = [
    (15140, 15800),   # chip a (blue)
    (15860, 16430),   # chip b (green)
    (16490, 16960),   # chip c (red)
]
APOGEE_NPIX_PER_ARM = 4000   # model internal grid (adjust if your model differs)
APOGEE_N_ARMS       = 3
APOGEE_N_FEATURES   = 30     # 10 NIR lines × 3 descriptors

# NIR absorption lines in the APOGEE H-band
# (central_wavelength Å, arm_index 0-based, label)
APOGEE_LINE_TABLE = [
    (15272.0, 0, "Fe_I_15272"),
    (15335.0, 0, "Fe_I_15335"),
    (15395.0, 0, "Mg_I_15395"),
    (15504.0, 0, "Al_I_15504"),
    (15560.0, 0, "CO_15560"),
    (16114.0, 1, "Fe_I_16114"),
    (16155.0, 1, "Mg_I_16155"),
    (16200.0, 1, "Si_I_16200"),
    (16710.0, 2, "Fe_I_16710"),
    (16770.0, 2, "OH_16770"),
]

PARAM_NAMES = ["T_eff (K)", "log g (dex)", "[Fe/H] (dex)"]
PARAM_UNITS = ["K",         "dex",          "dex"]


# ── Model import ─────────────────────────────────────────────────────────────

def _load_model(weights_path: str, device):
    try:
        from src.models.apogee.hybrid_net import StellarParameterHybridNet
    except ImportError:
        raise ImportError(
            "Could not import StellarParameterHybridNet from src.models.apogee.hybrid_net.\n"
            "Ensure this script is run from the project root."
        )
    import torch
    model = StellarParameterHybridNet()
    ckpt  = torch.load(weights_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt.get("model_state", ckpt))
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    print(f"  ✓  APOGEE model loaded from {weights_path}")
    return model


# ── Normalisation stats ───────────────────────────────────────────────────────

def _load_norm_stats(proc_dir: str):
    label_path   = os.path.join(proc_dir, "label_stats.npy")
    feature_path = os.path.join(proc_dir, "feature_stats.npy")
    for p in [label_path, feature_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Normalisation stats not found: {p}\n"
                "Run the APOGEE preprocessing pipeline first."
            )
    ls = np.load(label_path)      # (2, 3)
    fs = np.load(feature_path)    # (2, 30)
    return ls[0].astype(np.float32), ls[1].astype(np.float32), \
           fs[0].astype(np.float32), fs[1].astype(np.float32)


# ── GBS catalogue loader (same as GALAH script) ───────────────────────────────

def load_gbs_catalogue(fits_path: str) -> list[dict]:
    from astropy.io import fits as afits

    stars = []
    with afits.open(fits_path) as hdul:
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
        tmass  = _col(["_2MASS", "TMASS", "2MASS_ID", "TWOMASS"])
        teff   = _col(["TEFF", "T_EFF", "TEFF_GBS"])
        logg   = _col(["LOGG", "LOG_G", "LOGG_GBS"])
        feh    = _col(["__FE_H_", "FEH", "FE_H", "[FE/H]", "MET"])
        e_teff = _col(["E_TEFF", "ETEFF", "ERR_TEFF"])
        e_logg = _col(["E_LOGG", "ELOGG", "ERR_LOGG"])

        for i in range(len(tbl)):
            t = float(teff[i])  if teff  is not None else np.nan
            g = float(logg[i])  if logg  is not None else np.nan
            f = float(feh[i])   if feh   is not None else np.nan
            try:
                h = int("".join(c for c in str(hip[i]) if c.isdigit())) if hip is not None else -1
            except (ValueError, TypeError):
                h = -1
            tm = str(tmass[i]).strip() if tmass is not None else ""

            if not (np.isfinite(t) and np.isfinite(g)):
                continue
            stars.append({
                "hip_id":    h,
                "tmass_id":  tm,
                "teff_gbs":  t,
                "logg_gbs":  g,
                "feh_gbs":   f,
                "teff_err":  float(e_teff[i]) if e_teff is not None else np.nan,
                "logg_err":  float(e_logg[i]) if e_logg is not None else np.nan,
            })

    print(f"  ✓  GBS catalogue loaded — {len(stars)} stars with Teff & logg")
    return stars


# ── GBS × APOGEE cross-match ──────────────────────────────────────────────────

def crossmatch_gbs_apogee(gbs_stars: list[dict], apogee_cat: str) -> list[dict]:
    """
    Match GBS stars to APOGEE DR17 via 2MASS ID.
    APOGEE_ID format: '2M<HHMMSSSS±DDMMSS>'
    GBS 2MASS column: '<HHMMSSSS±DDMMSS>' (without '2M' prefix)
    """
    if not os.path.exists(apogee_cat):
        print(f"  [WARN] APOGEE catalogue not found: {apogee_cat}")
        return []

    ext = os.path.splitext(apogee_cat)[1].lower()
    tmass_to_apogee: dict[str, dict] = {}   # normalised_id → {apogee_id, file, field}

    def _normalise(apogee_id: str) -> str:
        """Strip '2M' prefix for matching."""
        s = str(apogee_id).strip()
        return s[2:] if s.upper().startswith("2M") else s

    if ext in (".fits", ".fit"):
        from astropy.io import fits as afits
        print("  Loading APOGEE catalogue (FITS) …")
        with afits.open(apogee_cat, memmap=True) as hdul:
            for ext_obj in hdul[1:]:
                if ext_obj.data is None:
                    continue
                names = [n.upper() for n in ext_obj.data.dtype.names]
                aid_col = next((ext_obj.data.dtype.names[i]
                                for i, n in enumerate(names) if "APOGEE_ID" in n), None)
                loc_col = next((ext_obj.data.dtype.names[i]
                                for i, n in enumerate(names) if "LOCATION" in n), None)
                fld_col = next((ext_obj.data.dtype.names[i]
                                for i, n in enumerate(names) if "FIELD" in n), None)
                if aid_col is None:
                    continue
                for row in ext_obj.data:
                    aid = str(row[aid_col]).strip()
                    key = _normalise(aid)
                    tmass_to_apogee[key] = {
                        "apogee_id":   aid,
                        "location_id": int(row[loc_col]) if loc_col else 0,
                        "field":       str(row[fld_col]).strip() if fld_col else "",
                    }
                break

    elif ext == ".csv":
        print("  Loading APOGEE catalogue (CSV) …")
        with open(apogee_cat, newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                aid_key = next((k for k in row if "APOGEE_ID" in k.upper()), None)
                if aid_key:
                    aid = row[aid_key].strip()
                    tmass_to_apogee[_normalise(aid)] = {
                        "apogee_id":   aid,
                        "location_id": 0,
                        "field":       "",
                    }

    # Build GBS 2MASS id set
    matched = []
    for star in gbs_stars:
        # GBS may have 2MASS directly or we derive from HIP name
        tm = star.get("tmass_id", "").strip()
        if not tm and star["hip_id"] > 0:
            tm = ""   # cannot derive without external lookup

        apogee_info = tmass_to_apogee.get(tm)
        if apogee_info:
            matched.append({**star, **apogee_info})

    print(f"  ✓  Cross-match: {len(matched)} / {len(gbs_stars)} GBS stars found in APOGEE")
    return matched


# ── APOGEE spectrum loader ────────────────────────────────────────────────────

def _find_apogee_fits(spec_dir: str, apogee_id: str, location_id: int) -> str | None:
    """
    Try several common APOGEE DR17 directory layouts for apStar/aspcapStar files.
    """
    # Strip '2M' prefix for filename
    star_id = apogee_id.strip()

    candidates = [
        # DR17 standard: apStar-dr17-<id>.fits
        os.path.join(spec_dir, str(location_id), f"apStar-dr17-{star_id}.fits"),
        os.path.join(spec_dir, f"apStar-dr17-{star_id}.fits"),
        # aspcapStar
        os.path.join(spec_dir, str(location_id), f"aspcapStar-dr17-{star_id}.fits"),
        os.path.join(spec_dir, f"aspcapStar-dr17-{star_id}.fits"),
        # Flat with apogee_id as-is
        os.path.join(spec_dir, f"{star_id}.fits"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def _continuum_normalize(flux: np.ndarray, wave: np.ndarray, deg: int = 4) -> np.ndarray:
    finite = np.isfinite(flux) & (flux > 0)
    if finite.sum() < deg + 2:
        return np.ones_like(flux)
    try:
        c = np.polyfit(wave[finite], flux[finite], deg)
        cont = np.polyval(c, wave)
        cont = np.where(cont > 0, cont, 1.0)
        return flux / cont
    except Exception:
        return np.ones_like(flux)


def load_apogee_spectrum(fits_path: str) -> np.ndarray | None:
    """
    Load an APOGEE combined spectrum (apStar or aspcapStar).
    APOGEE spectra are stored in vacuum wavelengths on a log-lambda grid.
    Returns array shape (3, APOGEE_NPIX_PER_ARM) or None.
    """
    from astropy.io import fits as afits
    from scipy.interpolate import interp1d

    try:
        with afits.open(fits_path) as hdul:
            # Extension 1: combined normalised spectrum (aspcapStar)
            # Extension 1 in apStar: combined flux; Ext 2: error
            flux = None
            wave_grid_full = None

            for ext_idx in [1, 0]:
                if hdul[ext_idx].data is not None:
                    data = hdul[ext_idx].data
                    if data.ndim == 2:
                        flux = data[0].astype(np.float32)   # first visit = combined
                    elif data.ndim == 1:
                        flux = data.astype(np.float32)
                    if flux is not None:
                        hdr = hdul[ext_idx].header
                        crval = hdr.get("CRVAL1", np.log10(15140.0))
                        cdelt = hdr.get("CDELT1", hdr.get("CD1_1", 6e-6))
                        naxis = len(flux)
                        log_wave = crval + cdelt * np.arange(naxis)
                        wave_grid_full = 10.0 ** log_wave   # vacuum Å
                        break

            if flux is None or wave_grid_full is None:
                return None

            # Zero-flux bad pixels → interpolate
            bad = (flux == 0) | ~np.isfinite(flux)
            if bad.sum() > 0 and bad.sum() < len(flux) * 0.5:
                good = ~bad
                f_interp = interp1d(wave_grid_full[good], flux[good],
                                    kind="linear", bounds_error=False,
                                    fill_value=np.nanmedian(flux[good]))
                flux = f_interp(wave_grid_full).astype(np.float32)

    except Exception:
        return None

    result = np.zeros((APOGEE_N_ARMS, APOGEE_NPIX_PER_ARM), dtype=np.float32)
    for arm_idx, (wmin, wmax) in enumerate(APOGEE_ARM_RANGES):
        mask = (wave_grid_full >= wmin) & (wave_grid_full <= wmax)
        if mask.sum() < 10:
            continue
        arm_wave = wave_grid_full[mask]
        arm_flux = flux[mask]
        arm_flux = _continuum_normalize(arm_flux, arm_wave)

        target_wave = np.linspace(wmin, wmax, APOGEE_NPIX_PER_ARM)
        f_interp    = interp1d(arm_wave, arm_flux, kind="linear",
                               bounds_error=False, fill_value=1.0)
        result[arm_idx] = f_interp(target_wave).astype(np.float32)

    return result


# ── Feature extraction (NIR H-band) ──────────────────────────────────────────

def _gaussian(x, amp, mu, sigma):
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def _fit_line(wave_arm, flux_arm, center, half_width=6.0):
    from scipy.optimize import curve_fit
    mask = np.abs(wave_arm - center) < half_width
    if mask.sum() < 5:
        return 0.0, 0.0, 0.0
    w, f = wave_arm[mask], flux_arm[mask]
    depth = max(0.0, 1.0 - float(np.nanmin(f)))
    try:
        popt, _ = curve_fit(
            lambda x, amp, mu, sig: 1.0 - _gaussian(x, amp, mu, sig),
            w, f,
            p0=[depth, center, half_width / 3.0],
            bounds=([0, center - half_width, 0.5],
                    [2.0, center + half_width, half_width]),
            maxfev=400,
        )
        amp, mu, sig = popt
        return float(amp * abs(sig) * np.sqrt(2 * np.pi)), float(2.355 * abs(sig)), float(amp)
    except Exception:
        return 0.0, 0.0, float(depth)


def extract_apogee_features(flux_3arm: np.ndarray) -> np.ndarray:
    """Extract 30D feature vector from (3, APOGEE_NPIX_PER_ARM) flux."""
    features = np.zeros(APOGEE_N_FEATURES, dtype=np.float32)
    for line_idx, (center, arm_idx, _) in enumerate(APOGEE_LINE_TABLE):
        wmin, wmax = APOGEE_ARM_RANGES[arm_idx]
        wave_arm   = np.linspace(wmin, wmax, APOGEE_NPIX_PER_ARM)
        ew, fwhm, depth = _fit_line(wave_arm, flux_3arm[arm_idx], center)
        features[line_idx * 3]     = ew
        features[line_idx * 3 + 1] = fwhm
        features[line_idx * 3 + 2] = depth
    return features


# ── Metrics ───────────────────────────────────────────────────────────────────

def _metrics(y_true: np.ndarray, y_pred: np.ndarray):
    diff = y_true - y_pred
    mae  = np.mean(np.abs(diff), axis=0)
    rmse = np.sqrt(np.mean(diff ** 2, axis=0))
    ss_res = np.sum(diff ** 2, axis=0)
    ss_tot = np.sum((y_true - y_true.mean(axis=0)) ** 2, axis=0)
    r2   = 1.0 - ss_res / (ss_tot + 1e-8)
    rel  = np.zeros(3)
    for i, thresh in enumerate([0.0, 0.1, 0.01]):
        mask = np.abs(y_true[:, i]) > thresh
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
    print("  GBS v3 Evaluation — APOGEE HybridNet")
    print(f"  Generated : {datetime.now():%Y-%m-%d  %H:%M:%S}")
    print("=" * 70)
    print(f"  Device    : {device}")

    label_mean, label_std, feature_mean, feature_std = _load_norm_stats(args.apogee_processed)
    model     = _load_model(args.weights, device)
    gbs_stars = load_gbs_catalogue(args.gbs_cat)
    matched   = crossmatch_gbs_apogee(gbs_stars, args.apogee_catalogue)

    if not matched:
        print("\n  [ERROR] No GBS stars matched in APOGEE. Check catalogue paths and 2MASS IDs.")
        sys.exit(1)

    print(f"\n  Running inference on {len(matched)} stars …")

    records, preds_list, labels_list, skipped = [], [], [], 0

    for star in matched:
        fits_path = _find_apogee_fits(args.apogee_dir,
                                      star["apogee_id"],
                                      star.get("location_id", 0))
        if fits_path is None:
            skipped += 1
            continue

        flux_3arm = load_apogee_spectrum(fits_path)
        if flux_3arm is None:
            skipped += 1
            continue

        try:
            features = extract_apogee_features(flux_3arm)
        except Exception:
            features = np.zeros(APOGEE_N_FEATURES, dtype=np.float32)

        import torch
        flux_t    = torch.tensor(flux_3arm, dtype=torch.float32).unsqueeze(0).to(device)
        feat_norm = (features - feature_mean) / (feature_std + 1e-8)
        feat_t    = torch.tensor(feat_norm, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            pred_norm = model(flux_t, feat_t).cpu().numpy()[0]

        pred = pred_norm * label_std + label_mean

        gbs_label = np.array([star["teff_gbs"], star["logg_gbs"],
                               star.get("feh_gbs", np.nan)], dtype=np.float32)

        preds_list.append(pred)
        labels_list.append(gbs_label)
        records.append({
            "hip_id":       star["hip_id"],
            "apogee_id":    star["apogee_id"],
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
    feh_valid = np.isfinite(labels[:, 2])
    mae, rmse, r2, rel = _metrics(labels, preds)

    # ── Report ─────────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("  GBS v3 Evaluation Report — APOGEE HybridNet")
    print(f"  Generated      : {datetime.now():%Y-%m-%d  %H:%M:%S}")
    print(f"  Weights        : {args.weights}")
    print(f"  Stars evaluated: {len(records)}   (skipped: {skipped})")
    print(f"{'=' * 70}")

    for i, (name, unit) in enumerate(zip(PARAM_NAMES, PARAM_UNITS)):
        if i == 2:
            n = int(feh_valid.sum())
            mae_i  = float(np.mean(np.abs(labels[feh_valid,2]-preds[feh_valid,2])))
            rmse_i = float(np.sqrt(np.mean((labels[feh_valid,2]-preds[feh_valid,2])**2)))
            r2_i   = float(1.0 - np.sum((labels[feh_valid,2]-preds[feh_valid,2])**2) /
                           (np.sum((labels[feh_valid,2]-labels[feh_valid,2].mean())**2)+1e-8))
        else:
            n, mae_i, rmse_i, r2_i = len(records), float(mae[i]), float(rmse[i]), float(r2[i])

        print(f"\n   {name}  (n={n}):")
        print(f"     MAE            : {mae_i:.4f} {unit}")
        print(f"     RMSE           : {rmse_i:.4f} {unit}")
        print(f"     Relative Error : {float(rel[i]):.2f}%")
        print(f"     R2 Score       : {r2_i:.4f}")

    print(f"\n   Note: Teff and logg are GBS FUNDAMENTAL values (spectroscopy-independent).")
    print(f"         [Fe/H] compared against APOGEE ASPCAP labels.")
    print(f"         APOGEE NIR H-band (1.51-1.70 µm) vs GBS optical — "
          f"this is a true cross-wavelength-regime validation.")
    print(f"\n{'=' * 70}\n")

    os.makedirs(args.outdir, exist_ok=True)
    csv_path = os.path.join(args.outdir, "gbs_apogee_eval.csv")
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    print(f"  Results saved → {csv_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate APOGEE HybridNet on Gaia FGK Benchmark Stars v3"
    )
    parser.add_argument("--weights",
        default="weights/apogee/stellar_hybrid_model.pth",
        help="Path to APOGEE model weights (.pth)")
    parser.add_argument("--gbs-cat",
        default="data/gbs/gbs_v3_params.fits",
        help="GBS v3 parameter catalogue FITS (from download_gbs.py)")
    parser.add_argument("--apogee-dir",
        default="data/apogee/raw/spectra",
        help="Root directory containing APOGEE apStar/aspcapStar FITS files")
    parser.add_argument("--apogee-catalogue",
        default="data/apogee/raw/allStar-dr17.fits",
        help="APOGEE DR17 allStar catalogue (FITS or CSV) for 2MASS cross-match")
    parser.add_argument("--apogee-processed",
        default="data/apogee/processed",
        help="Directory containing label_stats.npy and feature_stats.npy")
    parser.add_argument("--outdir",
        default="results/gbs",
        help="Output directory for evaluation CSV")
    args = parser.parse_args()

    for p, label in [
        (args.weights,           "APOGEE weights"),
        (args.gbs_cat,           "GBS catalogue"),
        (args.apogee_processed,  "APOGEE processed dir"),
    ]:
        if not os.path.exists(p):
            print(f"[ERROR] {label} not found: {p}")
            sys.exit(1)

    run_evaluation(args)


if __name__ == "__main__":
    main()