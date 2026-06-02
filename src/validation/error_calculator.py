import os
import numpy as np
import torch
import csv
from tqdm import tqdm

from .eval_core import align_wavelength_resolution, read_sdss_spec
from .xai_analyzer import extract_18d_features_live_eval
from ..models.hybrid_net import StellarParameterHybridNet

# ── 경로 설정 ─────────────────────────────────────────────────────────────────
_base_dir    = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_proc_dir    = os.path.join(_base_dir, "data", "processed")

# 파장 그리드: 훈련 시 저장된 표준 그리드 사용
_wave_path = os.path.join(_proc_dir, "standard_wave.npy")
WAVE_GRID  = np.load(_wave_path) if os.path.exists(_wave_path) \
             else np.linspace(3650.0, 10250.0, 4563)

# ── 정규화 통계: 재학습 시 자동 갱신되는 파일에서 로드 ─────────────────────────
# engine.py가 훈련 split에서 fit한 통계를 저장 → 하드코딩 값과 불일치 방지
_label_stats_path   = os.path.join(_proc_dir, "label_stats.npy")
_feature_stats_path = os.path.join(_proc_dir, "feature_stats.npy")

if os.path.exists(_label_stats_path):
    _ls = np.load(_label_stats_path)          # shape (2, 3): [mean, std]
    LABEL_MEAN = _ls[0].astype(np.float32)
    LABEL_STD  = _ls[1].astype(np.float32)
    print(f"[eval] label_stats loaded: mean={LABEL_MEAN}, std={LABEL_STD}")
else:
    # 재학습 전 fallback — 구버전 하드코딩 값
    print("[eval] WARNING: label_stats.npy not found, using hardcoded fallback.")
    LABEL_MEAN = np.array([5169.055664,  3.549788, -0.657069], dtype=np.float32)
    LABEL_STD  = np.array([ 998.064880,  1.081975,  0.723029], dtype=np.float32)

if os.path.exists(_feature_stats_path):
    _fs = np.load(_feature_stats_path)        # shape (2, 18): [mean, std]
    FEATURE_MEAN = _fs[0].astype(np.float32)
    FEATURE_STD  = _fs[1].astype(np.float32)
    print(f"[eval] feature_stats loaded.")
else:
    print("[eval] WARNING: feature_stats.npy not found, using hardcoded fallback.")
    FEATURE_MEAN = np.array([
        1.464409,  4.742932,  0.266495,
        1.455140,  4.436723,  0.270130,
        1.388086,  3.913455,  0.304693,
        6.086025,  8.080866,  0.618852,
        3.542716, 13.821565,  0.300151,
        2.593579,  8.094519,  0.259293,
    ], dtype=np.float32)
    FEATURE_STD = np.array([
        0.862422, 2.056033, 0.084388,
        1.264955, 2.470768, 0.091035,
        1.372020, 1.999545, 0.095620,
        2.815774, 3.239444, 0.199310,
        1.590929, 4.727457, 0.147723,
        2.830290, 4.755941, 0.179381,
    ], dtype=np.float32)


def read_csv(csv_path):
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        lines = [l for l in f if not l.startswith("#")]
    for row in csv.DictReader(lines):
        rows.append({
            "plate":   int(row["plate"]),
            "mjd":     int(row["mjd"]),
            "fiberid": int(row["fiberid"]),
            "teff":    float(row["teffadop"]),
            "logg":    float(row["loggadop"]),
            "feh":     float(row["fehadop"]),
        })
    return rows


def collect_spec_fits_files(directory):
    if not os.path.isdir(directory):
        return []
    return sorted([
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.startswith("spec-") and f.endswith(".fits")
    ])


def calculate_statistical_metrics(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred), axis=0)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))

    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    ss_tot = np.sum((y_true - np.mean(y_true, axis=0)) ** 2, axis=0)
    r2 = 1.0 - (ss_res / (ss_tot + 1e-8))

    rel_teff = np.mean(np.abs(y_true[:, 0] - y_pred[:, 0]) / (np.abs(y_true[:, 0]) + 1e-8)) * 100
    rel_logg = np.mean(np.abs(y_true[:, 1] - y_pred[:, 1]) / (np.abs(y_true[:, 1]) + 1e-8)) * 100

    feh_true = y_true[:, 2]
    feh_pred = y_pred[:, 2]
    safe_mask = np.abs(feh_true) > 0.01

    if np.any(safe_mask):
        rel_feh = np.mean(np.abs(feh_true[safe_mask] - feh_pred[safe_mask]) / np.abs(feh_true[safe_mask])) * 100
    else:
        rel_feh = 0.00

    return mae, rmse, r2, np.array([rel_teff, rel_logg, rel_feh])


def load_spectra_from_fits_list(file_paths, csv_path, dataset_dir):
    try:
        csv_rows = read_csv(csv_path)
        truth_dict = {f"{r['plate']}-{r['mjd']}-{r['fiberid']:04d}": r for r in csv_rows}
    except Exception as e:
        print(f"   [ERR] Could not load CSV catalog: {e}")
        return np.array([]), np.array([]), []

    all_flux, all_truth = [], []
    valid_paths         = []

    print(f"Scanning {len(file_paths)} spec FITS file(s) in {dataset_dir}/ ...")

    for path in file_paths:
        base = os.path.basename(path)
        
        parts = base.replace(".fits", "").split("-")
        if len(parts) != 4 or parts[0] != "spec":
            continue
        key = f"{int(parts[1])}-{int(parts[2])}-{int(parts[3]):04d}"
        
        if key not in truth_dict:
            print(f"   [SKIP]  {base}: Not found in CSV catalog.")
            continue
            
        csv_truth = truth_dict[key]
        truth = {'TEFF': csv_truth['teff'], 'LOGG': csv_truth['logg'], 'FEH': csv_truth['feh']}

        try:
            flux, loglam, is_star = read_sdss_spec(path)
        except Exception as e:
            print(f"   [ERR]   {base}: {e}")
            continue

        if not is_star:
            continue

        aligned = align_wavelength_resolution(loglam, flux, target_pixel_size=4563)
        if aligned is None:
            print(f"   [SKIP]  {base}: flux alignment failed.")
            continue

        print(f"   [OK]    {base}  "
              f"T={truth['TEFF']:.0f}K  "
              f"logg={truth['LOGG']:.2f}  "
              f"[Fe/H]={truth['FEH']:.2f}")

        all_flux.append(aligned[0])
        all_truth.append([truth['TEFF'], truth['LOGG'], truth['FEH']])
        valid_paths.append(path)

    if not all_flux:
        raise RuntimeError(
            f"No valid STAR samples found in {dataset_dir}/.\n"
            "Run scripts/download_spec.py first to populate the dataset directory."
        )

    return (np.array(all_flux,  dtype=np.float32),
            np.array(all_truth, dtype=np.float32),
            valid_paths)


def run_real_bulk_evaluation():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[Core Active] Compute device → {device}\n")

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    weights_path = os.path.join(base_dir, "weights", "stellar_hybrid_model.pth")
    dataset_dir = os.path.join(base_dir, "data", "validation_dataset")
    csv_path = os.path.join(dataset_dir, "Skyserver_SQL6_1_2026 10_51_26 PM.csv")

    model = StellarParameterHybridNet().to(device)
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"   [Weights] Loaded from {weights_path}")
    else:
        print(f"   [WARN]  Weights not found — using random init.")
    model.eval()

    all_spec_files = collect_spec_fits_files(dataset_dir)
    if not all_spec_files:
        print(f"[Error] No spec-*.fits files found in {dataset_dir}/")
        print("        Run:  python scripts/download_spec.py")
        return

    print(f"Found {len(all_spec_files)} spec FITS file(s) in {dataset_dir}/\n")

    X_FLUX_ALL, Y_TRUE_ALL, valid_paths = load_spectra_from_fits_list(all_spec_files, csv_path, dataset_dir)
    total = len(X_FLUX_ALL)
    print(f"\nTotal valid STAR samples: {total}\n")

    if total < 3:
        print(f"⚠  Only {total} sample(s) — R² will be unreliable. "
              "Run scripts/download_spec.py to get more files.\n")

    pred_list = []
    for idx in tqdm(range(total), desc="Inference"):
        raw_flux = X_FLUX_ALL[idx].reshape(1, -1)

        f_mean    = np.mean(raw_flux)
        f_std     = np.std(raw_flux) + 1e-8
        norm_flux = np.clip((raw_flux - f_mean) / f_std, -3.0, 3.0)

        raw_feat  = extract_18d_features_live_eval(WAVE_GRID, raw_flux[0])
        norm_feat = (raw_feat - FEATURE_MEAN) / (FEATURE_STD + 1e-8)

        tensor_flux = torch.from_numpy(norm_flux).float().unsqueeze(1).to(device)
        tensor_feat = torch.from_numpy(norm_feat).float().unsqueeze(0).to(device)

        with torch.no_grad():
            norm_pred = model(tensor_flux, tensor_feat).cpu().numpy()[0]

        real_pred = norm_pred * LABEL_STD + LABEL_MEAN
        pred_list.append(real_pred)

    Y_PRED_ALL = np.array(pred_list)

    mae, rmse, r2, rel_err = calculate_statistical_metrics(Y_TRUE_ALL, Y_PRED_ALL)
    
    parameters = ["T_eff  (K)", "log g  (dex)", "[Fe/H] (dex)"]
    units      = ["K",          "dex",          "dex"]

    print("\n" + "=" * 80)
    print(" RAW INFERENCE PERFORMANCE (Cross-Domain)")
    print("=" * 80)
    print(f"{'Parameter':<20} | {'MAE':>10} | {'RMSE':>10} | {'Rel Err':>10} | {'R2':>10}")
    print("-" * 80)
    for i in range(3):
        r2_str = f"{r2[i]:.4f}" if total > 1 else "N/A"
        print(f"{parameters[i]:<20} | {mae[i]:>10.3f} | {rmse[i]:>10.3f} | "
              f"{rel_err[i]:>9.2f}% | {r2_str:>10}")
    print("=" * 80)

    report_dir = os.path.join(base_dir, "report")
    os.makedirs(report_dir, exist_ok=True)
    out_path = os.path.join(report_dir, "dataset_error_report.txt")
    
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("  SDSS DR17 Real Spec FITS — Evaluation Report\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Total valid STAR samples : {total}\n")
        f.write(f"Ground truth source      : SSPP CSV (*adop)\n")
        f.write(f"Label normalization      : true training stats (6085 MaStar samples)\n\n")
        
        f.write("▶ [SECTION 1] Raw Performance (Cross-Domain):\n")
        for i in range(3):
            r2_str = f"{r2[i]:.4f}" if total > 1 else "N/A"
            f.write(f"   * {parameters[i]}:\n")
            f.write(f"     MAE            : {mae[i]:.4f} {units[i]}\n")
            f.write(f"     RMSE           : {rmse[i]:.4f} {units[i]}\n")
            f.write(f"     Relative Error : {rel_err[i]:.2f}%\n")
            f.write(f"     R2 Score       : {r2_str}\n\n")
            
        if total <= 25:
            f.write("\nPer-sample detail:\n")
            names = ["T_eff", "log g", "[Fe/H]"]
            for i in range(total):
                f.write(f"  {os.path.basename(valid_paths[i])}\n")
                for j in range(3):
                    f.write(f"    {names[j]:6s}  "
                            f"true={Y_TRUE_ALL[i,j]:8.3f}  "
                            f"pred={Y_PRED_ALL[i,j]:8.3f}\n")

    print(f"\nReport saved → {out_path}")


if __name__ == "__main__":
    run_real_bulk_evaluation()
