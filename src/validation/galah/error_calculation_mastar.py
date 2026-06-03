"""
Cross-domain evaluation: GALAH-trained model evaluated on MaStar validation spectra.

This module assesses the generalisation capability of the GALAH HybridNet by
applying it to out-of-distribution MaStar optical spectra. Because MaStar covers
3600–10000 Å (optical) and GALAH covers 4713–7887 Å across 4 CCDs, the MaStar
flux is mapped onto the 4-arm GALAH grid by wavelength-range intersection; arms
that fall outside the MaStar coverage are zero-filled.
"""

import os
import numpy as np
import torch
from tqdm import tqdm

from .eval_core_mastar import load_mastar_spectra
from src.models.galah.hybrid_net import StellarParameterHybridNet
from src.data.galah.extract_features import extract_45d_features_single_star
from src.validation.galah.eval_core import GALAH_ARM_WAVES

# ── Paths ─────────────────────────────────────────────────────────────────────
_base_dir       = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_mastar_proc    = os.path.join(_base_dir, "data", "mastar", "processed")
_galah_proc     = os.path.join(_base_dir, "data", "galah",  "processed")

# MaStar wave grid for feature extraction and arm-mapping
_wave_path = os.path.join(_mastar_proc, "standard_wave.npy")
if not os.path.exists(_wave_path):
    raise FileNotFoundError(
        f"MaStar standard_wave.npy not found at: {_wave_path}\n"
        "Execute src/data/mastar/preprocess_flux.py first."
    )
MASTAR_WAVE_GRID = np.load(_wave_path)  # (4563,)

# GALAH normalisation statistics (45-D features, trained on GALAH split)
_galah_label_stats   = os.path.join(_galah_proc, "label_stats.npy")
_galah_feature_stats = os.path.join(_galah_proc, "feature_stats.npy")

if not os.path.exists(_galah_label_stats):
    raise FileNotFoundError(
        f"GALAH label_stats.npy not found at: {_galah_label_stats}\n"
        "Execute the GALAH training pipeline first."
    )
if not os.path.exists(_galah_feature_stats):
    raise FileNotFoundError(
        f"GALAH feature_stats.npy not found at: {_galah_feature_stats}\n"
        "Execute the GALAH training pipeline first."
    )

_ls = np.load(_galah_label_stats)
LABEL_MEAN   = _ls[0].astype(np.float32)
LABEL_STD    = _ls[1].astype(np.float32)

_fs = np.load(_galah_feature_stats)
FEATURE_MEAN = _fs[0].astype(np.float32)   # shape (45,)
FEATURE_STD  = _fs[1].astype(np.float32)   # shape (45,)

print(f"[GALAH/MaStar Eval] GALAH label stats loaded  — "
      f"T_eff mean={LABEL_MEAN[0]:.1f} K, std={LABEL_STD[0]:.1f} K")
print(f"[GALAH/MaStar Eval] GALAH feature stats loaded — "
      f"feature dim={FEATURE_MEAN.shape[0]}")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _mastar_flux_to_galah_arms(mastar_flux_1d):
    """
    Map a continuum-normalised MaStar spectrum (4563 pixels, 3600–10000 Å) onto
    the 4 GALAH CCD arm grids.  Arms whose wavelength range lies outside the
    MaStar coverage are zero-filled; overlapping regions are linearly interpolated.

    Parameters
    ----------
    mastar_flux_1d : np.ndarray shape (4563,)

    Returns
    -------
    np.ndarray shape (4, 4000)
    """
    from scipy.interpolate import interp1d as _interp1d

    valid = np.isfinite(mastar_flux_1d)
    f_interp = _interp1d(
        MASTAR_WAVE_GRID[valid], mastar_flux_1d[valid],
        kind='linear', bounds_error=False, fill_value=0.0
    )
    arms = []
    for arm_wave in GALAH_ARM_WAVES:
        arm_flux = f_interp(arm_wave).astype(np.float32)
        arms.append(arm_flux)
    return np.stack(arms, axis=0)   # (4, 4000)


def _extract_galah_features_from_mastar(mastar_flux_1d):
    """
    Extract 45-D physical features from a MaStar spectrum by first mapping it
    onto the GALAH 4-arm grid, then running the GALAH Gaussian feature extractor.
    Lines whose arm falls outside the MaStar wavelength range will produce
    zero-filled (fallback) feature values.
    """
    flux_4arm  = _mastar_flux_to_galah_arms(mastar_flux_1d)
    wave_4arm  = np.stack(GALAH_ARM_WAVES, axis=0)   # (4, 4000)
    return extract_45d_features_single_star(wave_4arm, flux_4arm)


def calculate_statistical_metrics(y_true, y_pred):
    mae  = np.mean(np.abs(y_true - y_pred), axis=0)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))

    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    ss_tot = np.sum((y_true - np.mean(y_true, axis=0)) ** 2, axis=0)
    r2 = 1.0 - (ss_res / (ss_tot + 1e-8))

    rel_teff = np.mean(
        np.abs(y_true[:, 0] - y_pred[:, 0]) / (np.abs(y_true[:, 0]) + 1e-8)
    ) * 100

    logg_true, logg_pred = y_true[:, 1], y_pred[:, 1]
    logg_safe = np.abs(logg_true) > 0.1
    rel_logg = (
        np.mean(np.abs(logg_true[logg_safe] - logg_pred[logg_safe])
                / np.abs(logg_true[logg_safe])) * 100
        if np.any(logg_safe) else 0.0
    )

    feh_true, feh_pred = y_true[:, 2], y_pred[:, 2]
    feh_safe = np.abs(feh_true) > 0.01
    rel_feh = (
        np.mean(np.abs(feh_true[feh_safe] - feh_pred[feh_safe])
                / np.abs(feh_true[feh_safe])) * 100
        if np.any(feh_safe) else 0.0
    )
    return mae, rmse, r2, np.array([rel_teff, rel_logg, rel_feh])


# ── Main evaluation ───────────────────────────────────────────────────────────

def run_mastar_bulk_evaluation():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n{'='*70}")
    print("  Cross-Domain Evaluation: GALAH Model on MaStar Spectra")
    print(f"{'='*70}")
    print(f"  Compute device : {device}")

    weights_path = os.path.join(_base_dir, "weights", "galah", "stellar_hybrid_model.pth")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"GALAH model weights not found at: {weights_path}\n"
            "Execute the GALAH training pipeline before cross-domain evaluation."
        )

    model = StellarParameterHybridNet(use_features=True).to(device)
    checkpoint = torch.load(weights_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
    else:
        model.load_state_dict(checkpoint)
    print(f"  Model checkpoint : {weights_path}")
    model.eval()

    X_FLUX_ALL, Y_TRUE_ALL, _ = load_mastar_spectra()
    total = len(X_FLUX_ALL)
    print(f"  MaStar validation samples : {total}\n")

    pred_list = []
    for idx in tqdm(range(total), desc="Cross-domain inference (GALAH ← MaStar)"):
        raw_flux_1d = X_FLUX_ALL[idx]   # (4563,)

        # Map MaStar spectrum onto GALAH 4-arm grid
        flux_4arm = _mastar_flux_to_galah_arms(raw_flux_1d)

        # Per-arm z-score (identical to GALAH dataset.py)
        f_mean = np.mean(flux_4arm, axis=1, keepdims=True)
        f_std  = np.std(flux_4arm,  axis=1, keepdims=True) + 1e-8
        norm_4arm = np.clip((flux_4arm - f_mean) / f_std, -3.0, 3.0)

        # 45-D features via GALAH extractor applied to arm-mapped flux
        raw_feat  = _extract_galah_features_from_mastar(raw_flux_1d)
        norm_feat = (raw_feat - FEATURE_MEAN) / (FEATURE_STD + 1e-8)

        tensor_flux = torch.from_numpy(norm_4arm).float().unsqueeze(0).to(device)   # (1,4,4000)
        tensor_feat = torch.from_numpy(norm_feat).float().unsqueeze(0).to(device)   # (1,45)

        with torch.no_grad():
            norm_pred = model(tensor_flux, tensor_feat).cpu().numpy()[0]

        pred_list.append(norm_pred * LABEL_STD + LABEL_MEAN)

    Y_PRED_ALL = np.array(pred_list)
    mae, rmse, r2, rel_err = calculate_statistical_metrics(Y_TRUE_ALL, Y_PRED_ALL)

    parameters = ["T_eff  (K)", "log g  (dex)", "[Fe/H] (dex)"]
    units      = ["K", "dex", "dex"]

    print(f"\n{'='*80}")
    print("  Cross-Domain Performance: GALAH Model — MaStar Validation Set")
    print(f"{'='*80}")
    print(f"  {'Parameter':<20} | {'MAE':>10} | {'RMSE':>10} | "
          f"{'Rel. Error':>10} | {'R²':>10}")
    print(f"  {'-'*76}")
    for i in range(3):
        print(f"  {parameters[i]:<20} | {mae[i]:>10.3f} | {rmse[i]:>10.3f} | "
              f"{rel_err[i]:>9.2f}% | {r2[i]:>10.4f}")
    print(f"{'='*80}\n")

    report_dir = os.path.join(_base_dir, "report", "galah")
    os.makedirs(report_dir, exist_ok=True)
    out_path = os.path.join(report_dir, "dataset_error_report_mastar.txt")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("  Cross-Domain Evaluation: GALAH Model on MaStar Validation Spectra\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"  Evaluation samples     : {total}\n")
        f.write(f"  Model source           : GALAH DR4 HybridNet\n")
        f.write(f"  Input domain           : MaStar DR17 (optical, 3600–10000 Å)\n")
        f.write(f"  Feature extraction     : 45-D Gaussian profiles (GALAH arm grid)\n")
        f.write(f"  Label normalisation    : GALAH training split statistics\n\n")
        for i in range(3):
            f.write(f"  {parameters[i]}:\n")
            f.write(f"    MAE            : {mae[i]:.4f} {units[i]}\n")
            f.write(f"    RMSE           : {rmse[i]:.4f} {units[i]}\n")
            f.write(f"    Relative Error : {rel_err[i]:.2f}%\n")
            f.write(f"    R²             : {r2[i]:.4f}\n\n")

    print(f"  Report saved to: {out_path}")


if __name__ == "__main__":
    run_mastar_bulk_evaluation()
