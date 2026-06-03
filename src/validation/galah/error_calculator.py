import os
import numpy as np
import torch
from tqdm import tqdm

from src.models.galah.hybrid_net import StellarParameterHybridNet
from src.utils.galah.config import DEVICE

def calculate_statistical_metrics(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred), axis=0)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))

    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    ss_tot = np.sum((y_true - np.mean(y_true, axis=0)) ** 2, axis=0)
    r2 = 1.0 - (ss_res / (ss_tot + 1e-8))

    rel_teff = np.mean(np.abs(y_true[:, 0] - y_pred[:, 0]) / (np.abs(y_true[:, 0]) + 1e-8)) * 100

    logg_true, logg_pred = y_true[:, 1], y_pred[:, 1]
    logg_safe = np.abs(logg_true) > 0.1
    rel_logg = np.mean(np.abs(logg_true[logg_safe] - logg_pred[logg_safe]) / np.abs(logg_true[logg_safe])) * 100 \
               if np.any(logg_safe) else 0.0

    feh_true, feh_pred = y_true[:, 2], y_pred[:, 2]
    safe_mask = np.abs(feh_true) > 0.01
    rel_feh = np.mean(np.abs(feh_true[safe_mask] - feh_pred[safe_mask]) / np.abs(feh_true[safe_mask])) * 100 \
              if np.any(safe_mask) else 0.0

    return mae, rmse, r2, np.array([rel_teff, rel_logg, rel_feh])


def run_real_bulk_evaluation():
    print(f"\n{'='*70}")
    print("  GALAH Validation Split Evaluation")
    print(f"{'='*70}")
    print(f"  Compute device : {DEVICE}")

    base_dir     = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    proc_dir     = os.path.join(base_dir, "data", "galah", "processed")
    flux_path    = os.path.join(proc_dir, "X_flux_clean.npy")
    feature_path = os.path.join(proc_dir, "X_features_physical.npy")
    label_path   = os.path.join(proc_dir, "Y_labels.npy")
    weights_path = os.path.join(base_dir, "weights", "galah", "stellar_hybrid_model.pth")

    for p in (flux_path, feature_path, label_path):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Processed GALAH file not found: {p}\n"
                "Execute the GALAH data pipeline first."
            )

    _label_stats_path   = os.path.join(proc_dir, "label_stats.npy")
    _feature_stats_path = os.path.join(proc_dir, "feature_stats.npy")
    for p in (_label_stats_path, _feature_stats_path):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Normalisation statistics not found: {p}\n"
                "Execute the GALAH training pipeline first."
            )

    _ls = np.load(_label_stats_path)
    LABEL_MEAN = _ls[0].astype(np.float32)
    LABEL_STD  = _ls[1].astype(np.float32)

    _fs = np.load(_feature_stats_path)
    FEATURE_MEAN = _fs[0].astype(np.float32)
    FEATURE_STD  = _fs[1].astype(np.float32)

    # ── 1. Rebuild validation split indices ────────────────────────────────────
    _flux_n = np.load(flux_path, mmap_mode='r').shape[0]
    _feat_n = np.load(feature_path, mmap_mode='r').shape[0]
    raw_labels = np.load(label_path)
    n = min(_flux_n, _feat_n, raw_labels.shape[0])
    raw_labels = raw_labels[:n]

    valid_mask    = (raw_labels[:, 0] > -900) & \
                    (raw_labels[:, 1] > -900) & \
                    (raw_labels[:, 2] > -900)
    valid_indices = np.where(valid_mask)[0]

    rng = np.random.default_rng(42)
    rng.shuffle(valid_indices)
    train_size = int(0.8 * len(valid_indices))
    val_idx = valid_indices[train_size:]

    print(f"  Validation samples : {len(val_idx)}")

    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Model weights not found at: {weights_path}\n"
            "Execute the GALAH training pipeline first."
        )
    model = StellarParameterHybridNet(use_features=True).to(DEVICE)
    checkpoint = torch.load(weights_path, map_location=DEVICE)
    if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
    else:
        model.load_state_dict(checkpoint)
    print(f"  Model checkpoint : {weights_path}")
    model.eval()

    # Load validation data using mmap to avoid loading the full array into RAM
    fluxes   = np.load(flux_path,    mmap_mode='r')[val_idx]
    features = np.load(feature_path, mmap_mode='r')[val_idx]
    truths   = raw_labels[val_idx]

    pred_list = []
    
    # Run evaluation
    for idx in tqdm(range(len(val_idx)), desc="GALAH Evaluation"):
        # 1. Z-score flux per arm
        raw_flux = fluxes[idx] # (4, 4000)
        f_mean = np.mean(raw_flux, axis=1, keepdims=True)
        f_std  = np.std(raw_flux, axis=1, keepdims=True) + 1e-8
        norm_flux = np.clip((raw_flux - f_mean) / f_std, -3.0, 3.0)

        # 2. Normalize features
        raw_feat = features[idx]
        norm_feat = (raw_feat - FEATURE_MEAN) / (FEATURE_STD + 1e-8)

        # To tensor
        tensor_flux = torch.from_numpy(norm_flux).float().unsqueeze(0).to(DEVICE) # (1, 4, 4000)
        tensor_feat = torch.from_numpy(norm_feat).float().unsqueeze(0).to(DEVICE) # (1, 45)

        with torch.no_grad():
            norm_pred = model(tensor_flux, tensor_feat).cpu().numpy()[0]

        real_pred = norm_pred * LABEL_STD + LABEL_MEAN
        pred_list.append(real_pred)

    preds = np.array(pred_list)

    mae, rmse, r2, rel_err = calculate_statistical_metrics(truths, preds)

    parameters = ["T_eff  (K)", "log g  (dex)", "[Fe/H] (dex)"]
    units      = ["K",          "dex",          "dex"]

    print(f"\n{'='*80}")
    print("  GALAH Validation Split — Performance Summary")
    print(f"{'='*80}")
    print(f"  {'Parameter':<20} | {'MAE':>10} | {'RMSE':>10} | {'Rel. Error':>10} | {'R²':>10}")
    print(f"  {'-'*76}")
    for i in range(3):
        print(f"  {parameters[i]:<20} | {mae[i]:>10.3f} | {rmse[i]:>10.3f} | "
              f"{rel_err[i]:>9.2f}% | {r2[i]:>10.4f}")
    print(f"{'='*80}")

    report_dir = os.path.join(base_dir, "report", "galah")
    os.makedirs(report_dir, exist_ok=True)
    out_path = os.path.join(report_dir, "dataset_error_report.txt")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("  GALAH Validation Split Evaluation Report\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Total validation samples : {len(val_idx)}\n")
        f.write(f"Label normalization      : label_stats.npy\n\n")
        
        f.write("▶ [SECTION 1] Performance Metrics:\n")
        for i in range(3):
            f.write(f"   * {parameters[i]}:\n")
            f.write(f"     MAE            : {mae[i]:.4f} {units[i]}\n")
            f.write(f"     RMSE           : {rmse[i]:.4f} {units[i]}\n")
            f.write(f"     Relative Error : {rel_err[i]:.2f}%\n")
            f.write(f"     R2 Score       : {r2[i]:.4f}\n\n")

    print(f"  Report saved to: {out_path}")


if __name__ == "__main__":
    run_real_bulk_evaluation()
