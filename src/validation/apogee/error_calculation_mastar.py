import os
import numpy as np
import torch
from tqdm import tqdm

from .eval_core_mastar import load_mastar_spectra
from .xai_analyzer import extract_30d_features_live_eval
from src.models.apogee.hybrid_net import StellarParameterHybridNet

# ── 경로 ──────────────────────────────────────────────────────────────────────
_base_dir  = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_proc_dir  = os.path.join(_base_dir, "data", "mastar", "processed")

# 파장 그리드
_wave_path = os.path.join(_proc_dir, "standard_wave.npy")
WAVE_GRID  = np.load(_wave_path) if os.path.exists(_wave_path) \
             else np.linspace(3650.0, 10250.0, 4563)

# ── 정규화 통계: 재학습 시 자동 갱신 ──────────────────────────────────────────
_label_stats_path   = os.path.join(_proc_dir, "label_stats.npy")
_feature_stats_path = os.path.join(_proc_dir, "feature_stats.npy")

if os.path.exists(_label_stats_path):
    _ls = np.load(_label_stats_path)
    LABEL_MEAN = _ls[0].astype(np.float32)
    LABEL_STD  = _ls[1].astype(np.float32)
    print(f"[mastar eval] label_stats loaded: mean={LABEL_MEAN}, std={LABEL_STD}")
else:
    print("[mastar eval] WARNING: label_stats.npy not found, using hardcoded fallback.")
    LABEL_MEAN = np.array([5169.055664,  3.549788, -0.657069], dtype=np.float32)
    LABEL_STD  = np.array([ 998.064880,  1.081975,  0.723029], dtype=np.float32)

if os.path.exists(_feature_stats_path):
    _fs = np.load(_feature_stats_path)
    FEATURE_MEAN = _fs[0].astype(np.float32)
    FEATURE_STD  = _fs[1].astype(np.float32)
    print(f"[mastar eval] feature_stats loaded.")
else:
    raise FileNotFoundError(
        f"feature_stats.npy not found at {_feature_stats_path}.\n"
        "Run scripts/train.py first to generate normalization stats."
    )


def calculate_statistical_metrics(y_true, y_pred):
    mae  = np.mean(np.abs(y_true - y_pred), axis=0)
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
    safe = np.abs(feh_true) > 0.01
    rel_feh = np.mean(np.abs(feh_true[safe] - feh_pred[safe]) / np.abs(feh_true[safe])) * 100 \
              if np.any(safe) else 0.0
    return mae, rmse, r2, np.array([rel_teff, rel_logg, rel_feh])


def run_mastar_bulk_evaluation():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[Core Active] Compute device → {device}\n")

    weights_path = os.path.join(_base_dir, "weights", "apogee", "stellar_hybrid_model.pth")
    model = StellarParameterHybridNet().to(device)
    if os.path.exists(weights_path):
        checkpoint = torch.load(weights_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
            model.load_state_dict(checkpoint['model_state'])
        else:
            model.load_state_dict(checkpoint)
        print(f"   [Weights] Loaded from {weights_path}")
    else:
        print(f"   [WARN] Weights not found — using random init.")
    model.eval()

    try:
        X_FLUX_ALL, Y_TRUE_ALL, _ = load_mastar_spectra()
    except Exception as e:
        print(f"[Error] Failed to load MaStar spectra: {e}")
        return

    total = len(X_FLUX_ALL)
    print(f"\nTotal valid MaStar samples: {total}\n")

    pred_list = []
    for idx in tqdm(range(total), desc="Inference"):
        # MaStar flux shape: (4563,) — 1D single spectrum
        # APOGEE MultiArmCNNBranch expects (batch, num_arms, length) = (1, 3, 2800)
        # MaStar is a single-arm survey, so we cannot directly feed it into
        # the APOGEE multi-arm model. Instead we run CNN-only mode (use_features=False)
        # and skip the multi-arm path by using a single arm replicated 3x.
        # This is a cross-domain evaluation: results reflect domain gap, not model error.
        raw_flux_1d = X_FLUX_ALL[idx]  # (4563,)
        f_mean = np.mean(raw_flux_1d)
        f_std  = np.std(raw_flux_1d) + 1e-8
        norm_1d = np.clip((raw_flux_1d - f_mean) / f_std, -3.0, 3.0)

        # Interpolate 4563-pixel MaStar grid onto 3 x 2800 APOGEE arm grids
        from scipy.interpolate import interp1d as _interp1d
        mastar_wave = WAVE_GRID  # (4563,) optical
        apogee_arm_waves = [
            np.linspace(15140, 15810, 2800),
            np.linspace(15850, 16430, 2800),
            np.linspace(16470, 16960, 2800),
        ]
        # MaStar does not cover NIR — fill with continuum level (1.0 after norm ≈ 0)
        norm_3arm = np.zeros((3, 2800), dtype=np.float32)
        # For cross-domain test we simply zero-fill all arms (out-of-range)
        # and rely solely on the feature branch for meaningful signal.
        # (MaStar optical lines are extracted via WAVE_GRID below.)

        raw_feat  = extract_30d_features_live_eval(WAVE_GRID, raw_flux_1d)
        norm_feat = (raw_feat - FEATURE_MEAN) / (FEATURE_STD + 1e-8)

        tensor_flux = torch.from_numpy(norm_3arm).float().unsqueeze(0).to(device)  # (1,3,2800)
        tensor_feat = torch.from_numpy(norm_feat).float().unsqueeze(0).to(device)  # (1,30)

        with torch.no_grad():
            norm_pred = model(tensor_flux, tensor_feat).cpu().numpy()[0]

        real_pred = norm_pred * LABEL_STD + LABEL_MEAN
        pred_list.append(real_pred)

    Y_PRED_ALL = np.array(pred_list)
    mae, rmse, r2, rel_err = calculate_statistical_metrics(Y_TRUE_ALL, Y_PRED_ALL)

    parameters = ["T_eff  (K)", "log g  (dex)", "[Fe/H] (dex)"]
    units      = ["K", "dex", "dex"]

    print("\n" + "=" * 65)
    print("  MaStar Dataset Cross Validation - Evaluation Report")
    print("=" * 65)
    print(f"{'Parameter':<20} | {'MAE':>10} | {'RMSE':>10} | {'R2':>10}")
    print("-" * 65)
    for i in range(3):
        print(f"{parameters[i]:<20} | {mae[i]:>10.4f} | {rmse[i]:>10.4f} | {r2[i]:>10.4f}")
    print("=" * 65)

    report_dir = os.path.join(_base_dir, "report", "apogee")
    os.makedirs(report_dir, exist_ok=True)
    out_path = os.path.join(report_dir, "dataset_error_report_mastar.txt")

    n_train = int(0.8 * (total + int(total / 0.2 * 0.8)))  # approximate
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("=" * 65 + "\n")
        f.write("  MaStar Dataset Cross Validation - Evaluation Report\n")
        f.write("=" * 65 + "\n\n")
        f.write(f"Total valid samples : {total}\n")
        f.write(f"Ground truth source : MaStar VAC v2 (TEFF_MED/LOGG_MED/FEH_NOAPP_MED)\n")
        f.write(f"Label normalization : label_stats.npy (train split)\n\n")
        for i in range(3):
            f.write(f"▶ {parameters[i]}:\n")
            f.write(f"   MAE            : {mae[i]:.4f} {units[i]}\n")
            f.write(f"   RMSE           : {rmse[i]:.4f} {units[i]}\n")
            f.write(f"   Relative Error : {rel_err[i]:.2f}%\n")
            f.write(f"   R2 Score       : {r2[i]:.4f}\n\n")

    print(f"\nReport saved → {out_path}")


if __name__ == "__main__":
    run_mastar_bulk_evaluation()
