import os
import numpy as np
import torch
import csv
from tqdm import tqdm

from .eval_core import align_wavelength_resolution, read_sdss_spec
from .xai_analyzer import extract_30d_features_live_eval
from src.models.mastar.hybrid_net import StellarParameterHybridNet

# ── Paths ─────────────────────────────────────────────────────────────────────
_base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_proc_dir = os.path.join(_base_dir, "data", "mastar", "processed")

# ── Wave grid ─────────────────────────────────────────────────────────────────
_wave_path = os.path.join(_proc_dir, "standard_wave.npy")
if not os.path.exists(_wave_path):
    raise FileNotFoundError(
        f"MaStar standard_wave.npy not found at: {_wave_path}\n"
        "Execute src/data/mastar/preprocess_flux.py first."
    )
WAVE_GRID = np.load(_wave_path)

# ── Normalisation statistics ──────────────────────────────────────────────────
_label_stats_path   = os.path.join(_proc_dir, "label_stats.npy")
_feature_stats_path = os.path.join(_proc_dir, "feature_stats.npy")

if not os.path.exists(_label_stats_path):
    raise FileNotFoundError(
        f"label_stats.npy not found at: {_label_stats_path}\n"
        "Execute the MaStar training pipeline first."
    )
if not os.path.exists(_feature_stats_path):
    raise FileNotFoundError(
        f"feature_stats.npy not found at: {_feature_stats_path}\n"
        "Execute the MaStar training pipeline first."
    )

_ls = np.load(_label_stats_path)
LABEL_MEAN = _ls[0].astype(np.float32)
LABEL_STD  = _ls[1].astype(np.float32)
print(f"[MaStar Eval] Label statistics loaded — "
      f"T_eff mean={LABEL_MEAN[0]:.1f} K, std={LABEL_STD[0]:.1f} K")

_fs = np.load(_feature_stats_path)
FEATURE_MEAN = _fs[0].astype(np.float32)
FEATURE_STD  = _fs[1].astype(np.float32)
print(f"[MaStar Eval] Feature statistics loaded — feature dim={FEATURE_MEAN.shape[0]}")


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


def load_spectra_from_fits_list(file_paths, csv_path, dataset_dir):
    try:
        csv_rows = read_csv(csv_path)
        truth_dict = {
            f"{r['plate']}-{r['mjd']}-{r['fiberid']:04d}": r for r in csv_rows
        }
    except Exception as e:
        raise RuntimeError(f"Failed to load CSV catalog from {csv_path}: {e}")

    all_flux, all_truth, valid_paths = [], [], []

    print(f"[MaStar Eval] Scanning {len(file_paths)} FITS files in: {dataset_dir}")

    for path in file_paths:
        base  = os.path.basename(path)
        parts = base.replace(".fits", "").split("-")
        if len(parts) != 4 or parts[0] != "spec":
            continue
        key = f"{int(parts[1])}-{int(parts[2])}-{int(parts[3]):04d}"

        if key not in truth_dict:
            print(f"   [SKIP] {base}: identifier not found in CSV catalog.")
            continue

        csv_truth = truth_dict[key]
        truth = {
            "TEFF": csv_truth["teff"],
            "LOGG": csv_truth["logg"],
            "FEH":  csv_truth["feh"],
        }

        try:
            flux, loglam, is_star = read_sdss_spec(path)
        except Exception as e:
            print(f"   [SKIP] {base}: FITS read error — {e}")
            continue

        if not is_star:
            continue

        aligned = align_wavelength_resolution(loglam, flux,
                                              target_pixel_size=4563,
                                              target_wave_grid=WAVE_GRID)
        if aligned is None:
            print(f"   [SKIP] {base}: wavelength alignment failed.")
            continue

        print(f"   [OK]   {base}  "
              f"T_eff={truth['TEFF']:.0f} K  "
              f"log g={truth['LOGG']:.2f}  "
              f"[Fe/H]={truth['FEH']:.2f}")

        all_flux.append(aligned[0])
        all_truth.append([truth["TEFF"], truth["LOGG"], truth["FEH"]])
        valid_paths.append(path)

    if not all_flux:
        raise RuntimeError(
            f"No valid stellar spectra found in: {dataset_dir}\n"
            "Execute scripts/mastar/download_spec.py to populate the dataset."
        )

    return (
        np.array(all_flux,  dtype=np.float32),
        np.array(all_truth, dtype=np.float32),
        valid_paths,
    )


def run_real_bulk_evaluation():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n{'='*70}")
    print("  MaStar Cross-Domain Evaluation: SDSS DR17 Spectra")
    print(f"{'='*70}")
    print(f"  Compute device : {device}")

    base_dir     = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    weights_path = os.path.join(base_dir, "weights", "mastar", "stellar_hybrid_model.pth")
    dataset_dir  = os.path.join(base_dir, "data", "mastar", "validation_dataset")
    csv_path     = os.path.join(dataset_dir, "Skyserver_SQL6_1_2026 10_51_26 PM.csv")

    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Model weights not found at: {weights_path}\n"
            "Execute the MaStar training pipeline first."
        )

    model = StellarParameterHybridNet().to(device)
    checkpoint = torch.load(weights_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    else:
        model.load_state_dict(checkpoint)
    print(f"  Model checkpoint : {weights_path}")
    model.eval()

    all_spec_files = collect_spec_fits_files(dataset_dir)
    if not all_spec_files:
        raise FileNotFoundError(
            f"No spec-*.fits files found in: {dataset_dir}\n"
            "Execute scripts/mastar/download_spec.py first."
        )

    X_FLUX_ALL, Y_TRUE_ALL, valid_paths = load_spectra_from_fits_list(
        all_spec_files, csv_path, dataset_dir
    )
    total = len(X_FLUX_ALL)
    print(f"\n  Valid stellar samples : {total}\n")

    if total < 3:
        print(f"  [NOTE] Only {total} sample(s) available. "
              "R² scores are unreliable with fewer than 3 samples.")

    pred_list = []
    for idx in tqdm(range(total), desc="Inference"):
        raw_flux  = X_FLUX_ALL[idx].reshape(1, -1)
        f_mean    = np.mean(raw_flux)
        f_std     = np.std(raw_flux) + 1e-8
        norm_flux = np.clip((raw_flux - f_mean) / f_std, -3.0, 3.0)

        raw_feat  = extract_30d_features_live_eval(WAVE_GRID, raw_flux[0])
        norm_feat = (raw_feat - FEATURE_MEAN) / (FEATURE_STD + 1e-8)

        tensor_flux = torch.from_numpy(norm_flux).float().unsqueeze(1).to(device)
        tensor_feat = torch.from_numpy(norm_feat).float().unsqueeze(0).to(device)

        with torch.no_grad():
            norm_pred = model(tensor_flux, tensor_feat).cpu().numpy()[0]

        pred_list.append(norm_pred * LABEL_STD + LABEL_MEAN)

    Y_PRED_ALL = np.array(pred_list)
    mae, rmse, r2, rel_err = calculate_statistical_metrics(Y_TRUE_ALL, Y_PRED_ALL)

    parameters = ["T_eff  (K)", "log g  (dex)", "[Fe/H] (dex)"]
    units      = ["K", "dex", "dex"]

    print(f"\n{'='*80}")
    print("  Cross-Domain Evaluation Performance — MaStar Model on SDSS DR17")
    print(f"{'='*80}")
    print(f"  {'Parameter':<20} | {'MAE':>10} | {'RMSE':>10} | "
          f"{'Rel. Error':>10} | {'R²':>10}")
    print(f"  {'-'*76}")
    for i in range(3):
        r2_str = f"{r2[i]:.4f}" if total > 1 else "N/A"
        print(f"  {parameters[i]:<20} | {mae[i]:>10.3f} | {rmse[i]:>10.3f} | "
              f"{rel_err[i]:>9.2f}% | {r2_str:>10}")
    print(f"{'='*80}\n")

    report_dir = os.path.join(base_dir, "report", "mastar")
    os.makedirs(report_dir, exist_ok=True)
    out_path = os.path.join(report_dir, "dataset_error_report.txt")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("  Cross-Domain Evaluation: MaStar Model on SDSS DR17 Spectra\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"  Valid stellar samples  : {total}\n")
        f.write(f"  Ground truth source    : SDSS SSPP (*adop parameters)\n")
        f.write(f"  Label normalisation    : label_stats.npy (training split)\n\n")
        f.write("  Performance Metrics:\n")
        for i in range(3):
            r2_str = f"{r2[i]:.4f}" if total > 1 else "N/A"
            f.write(f"    {parameters[i]}:\n")
            f.write(f"      MAE            : {mae[i]:.4f} {units[i]}\n")
            f.write(f"      RMSE           : {rmse[i]:.4f} {units[i]}\n")
            f.write(f"      Relative Error : {rel_err[i]:.2f}%\n")
            f.write(f"      R²             : {r2_str}\n\n")

        if total <= 25:
            f.write("  Per-sample results:\n")
            names = ["T_eff", "log g", "[Fe/H]"]
            for i in range(total):
                f.write(f"    {os.path.basename(valid_paths[i])}\n")
                for j in range(3):
                    f.write(f"      {names[j]:6s}  "
                            f"true={Y_TRUE_ALL[i,j]:8.3f}  "
                            f"pred={Y_PRED_ALL[i,j]:8.3f}\n")

    print(f"  Report saved to: {out_path}")


if __name__ == "__main__":
    run_real_bulk_evaluation()
