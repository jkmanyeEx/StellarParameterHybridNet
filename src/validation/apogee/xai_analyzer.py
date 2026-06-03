import os
import numpy as np
import torch
from tqdm import tqdm

from src.models.apogee.hybrid_net import StellarParameterHybridNet
from src.data.apogee.extract_features import extract_30d_features_single_star

def extract_30d_features_live_eval(wave, norm_flux):
    return extract_30d_features_single_star(wave, norm_flux)


# ── 아키텍처 차원 상수 ────────────────────────────────────────────────────────
CNN_BRANCH_DIM   = 4800   # 3 arms x 1600
DENSE_BRANCH_DIM = 128
FUSION_DIM       = CNN_BRANCH_DIM + DENSE_BRANCH_DIM  # 4928

LINE_NAMES_30D = [
    "Fe_I_15200",   "Fe_I_15648",   "Mg_I_15749",
    "Mg_I_15886",   "Si_I_15960",   "Br_14",        "Fe_I_16040",
    "Si_I_16680",   "Al_I_16755",   "Br_11"
]


def calculate_per_line_weight_attribution(model):
    """
    Measure each absorption line's contribution through the Dense branch for APOGEE.
    Dense branch first layer: Linear(30, 128).
    """
    try:
        first_weight = None
        for name, param in model.named_parameters():
            if "feature_branch" in name and "weight" in name and param.ndim == 2:
                if param.shape[1] == 30:
                    first_weight = param.detach().cpu().numpy()
                    break

        if first_weight is None:
            print("XAI WARNING: Dense branch Linear(30,*) not found.")
            return []

        total_mag = np.sum(np.abs(first_weight))
        if total_mag == 0:
            return [(n, 0.0) for n in LINE_NAMES_30D]

        results = []
        for i, line_name in enumerate(LINE_NAMES_30D):
            col_s = i * 3
            col_e = col_s + 3
            pct   = np.sum(np.abs(first_weight[:, col_s:col_e])) / total_mag * 100
            results.append((line_name, float(pct)))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    except Exception as e:
        print(f"XAI per-line attribution error: {e}")
        return []


def calculate_eval_model_weight_ratio(model):
    """
    Estimate Dense branch fraction of post-fusion layer L1 weight.
    Concat order: [CNN:4800 | Dense:128] = 4928 total.
    """
    try:
        target = None
        for name, param in model.named_parameters():
            if "weight" in name.lower() and param.ndim == 2 \
                    and param.shape[1] == FUSION_DIM:
                target = param.detach().cpu().numpy()
                break

        if target is None:
            print(f"XAI ERROR: no Linear(*, {FUSION_DIM}) found.")
            return -1.0

        total     = np.sum(np.abs(target))
        if total == 0:
            return 0.0
        cnn_mag   = np.sum(np.abs(target[:, :CNN_BRANCH_DIM]))
        dense_mag = np.sum(np.abs(target[:, CNN_BRANCH_DIM:]))
        print(f"   [XAI] CNN branch   weight share : {cnn_mag/total*100:.2f}%")
        print(f"   [XAI] Dense branch weight share : {dense_mag/total*100:.2f}%")
        return float(dense_mag / total * 100)

    except Exception as e:
        print(f"XAI CRASH: {e}")
        return -1.0


def run_xai_line_profile_analysis(num_samples=100):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[APOGEE XAI Engine] Initializing on: {device}")

    base_dir  = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    proc_dir  = os.path.join(base_dir, "data", "apogee", "processed")

    # ── label stats ──
    _ls_path = os.path.join(proc_dir, "label_stats.npy")
    if os.path.exists(_ls_path):
        _ls       = np.load(_ls_path)
        LABEL_STD = _ls[1].astype(np.float32)
        print(f"   [XAI] label_stats loaded: std={LABEL_STD}")
    else:
        LABEL_STD = np.array([1000.0, 1.0, 0.5], dtype=np.float32)

    # ── 모델 로드 ──
    weights_path = os.path.join(base_dir, "weights", "apogee", "stellar_hybrid_model.pth")
    model = StellarParameterHybridNet(use_features=True).to(device)
    if os.path.exists(weights_path):
        ckpt = torch.load(weights_path, map_location=device)
        if isinstance(ckpt, dict) and 'model_state' in ckpt:
            model.load_state_dict(ckpt['model_state'])
        else:
            model.load_state_dict(ckpt)
        print(f"   [Weights] Loaded from {weights_path}")
    else:
        print("   [WARN] Weights not found — running with random init.")
    model.eval()

    # ── 스펙트럼 로드 ──
    flux_path = os.path.join(proc_dir, "X_flux_clean.npy")
    wave_path = os.path.join(proc_dir, "standard_wave.npy")
    if not os.path.exists(flux_path):
        raise FileNotFoundError(f"Preprocessed flux not found: {flux_path}")

    X_flux_all = np.load(flux_path)
    wave_grid  = np.load(wave_path)

    total_available = X_flux_all.shape[0]
    actual_samples  = min(num_samples, total_available)
    print(f"   > Dataset: {total_available} stars, using {actual_samples} for XAI.")

    np.random.seed(42)
    sample_indices = np.random.choice(total_available, size=actual_samples, replace=False)

    # APOGEE Absorption lines mapping to chips
    absorption_lines = {
        "Fe-I-15648 (Blue Chip)": (15610.0, 15690.0, 0),
        "Br-14 (Green Chip)":     (15850.0, 15920.0, 1),
        "Br-11 (Red Chip)":       (16780.0, 16840.0, 2),
    }

    num_arms = 3
    n_pixels = 2800
    
    jac_acc         = np.zeros((3, num_arms, n_pixels))
    jac_acc_ablated = np.zeros((3, num_arms, n_pixels))
    baseline_preds  = []
    ablated_preds   = []
    valid_count     = 0

    print(f"Calculating Jacobians over {actual_samples} spectra...\n")

    for idx in tqdm(sample_indices, desc="APOGEE Jacobian XAI"):
        raw_flux  = X_flux_all[idx] # (3, 2800)
        f_mean    = np.mean(raw_flux, axis=1, keepdims=True)
        f_std     = np.std(raw_flux, axis=1, keepdims=True) + 1e-8
        norm_flux = np.clip((raw_flux - f_mean) / f_std, -3.0, 3.0)

        features_30d = extract_30d_features_live_eval(wave_grid, raw_flux)
        feat_tensor  = torch.from_numpy(features_30d).float().unsqueeze(0).to(device)
        zero_feat    = torch.zeros_like(feat_tensor)
        norm_flux_t  = torch.from_numpy(norm_flux).float()

        # Baseline Jacobian
        input_base = norm_flux_t.unsqueeze(0).to(device).requires_grad_(True) # (1, 3, 2800)
        pred = model(input_base, feat_tensor)
        baseline_preds.append(pred.detach().cpu().numpy()[0])
        for p_idx in range(3):
            input_base.grad = None
            g = torch.zeros_like(pred); g[0, p_idx] = 1.0
            pred.backward(g, retain_graph=(p_idx < 2))
            jac_acc[p_idx] += np.abs(input_base.grad.cpu().numpy()[0])

        # Ablated Jacobian
        input_abl = norm_flux_t.unsqueeze(0).to(device).requires_grad_(True)
        pred_abl  = model(input_abl, zero_feat)
        ablated_preds.append(pred_abl.detach().cpu().numpy()[0])
        for p_idx in range(3):
            input_abl.grad = None
            g = torch.zeros_like(pred_abl); g[0, p_idx] = 1.0
            pred_abl.backward(g, retain_graph=(p_idx < 2))
            jac_acc_ablated[p_idx] += np.abs(input_abl.grad.cpu().numpy()[0])

        valid_count += 1

    mean_jac         = jac_acc         / max(valid_count, 1)
    mean_jac_ablated = jac_acc_ablated / max(valid_count, 1)

    baseline_preds = np.array(baseline_preds)
    ablated_preds  = np.array(ablated_preds)
    mad            = np.mean(np.abs(baseline_preds - ablated_preds), axis=0)
    physical_mad   = mad * LABEL_STD

    # ── 리포트 계산 ──
    line_scores = {}
    for name, (lo, hi, arm_idx) in absorption_lines.items():
        wave = wave_grid[arm_idx]
        mask = (wave >= lo) & (wave <= hi)
        line_scores[name] = (
            float(np.mean(mean_jac[0, arm_idx, mask])),
            float(np.mean(mean_jac[1, arm_idx, mask])),
        )

    # Proof ratio based on Green Chip Br-14 vs non Br-14 in Green Chip
    wave_green = wave_grid[1]
    cont_mask = ~((wave_green >= 15850) & (wave_green <= 15920))
    bg         = float(np.mean(mean_jac[0, 1, cont_mask]))
    proof_r    = line_scores["Br-14 (Green Chip)"][0] / (bg + 1e-8)

    bg_abl     = float(np.mean(mean_jac_ablated[0, 1, cont_mask]))
    ha_abl     = float(np.mean(mean_jac_ablated[0, 1, (wave_green >= 15850) & (wave_green <= 15920)]))
    proof_r_abl = ha_abl / (bg_abl + 1e-8)

    weight_ratio = calculate_eval_model_weight_ratio(model)
    per_line     = calculate_per_line_weight_attribution(model)

    # ── 터미널 출력 ──
    print("\n" + "=" * 65)
    for name, (t, g) in line_scores.items():
        print(f"{name:<35} T={t:.5f}  g={g:.5f}")
    print(f"\nProof Ratio (Normal):  {proof_r:.4f}")
    print(f"Proof Ratio (Ablated): {proof_r_abl:.4f}")
    print(f"30D branch weight:     {weight_ratio:.2f}%")
    print(f"T_eff ablation shift:  {physical_mad[0]:.4f} K")
    print(f"log g ablation shift:  {physical_mad[1]:.4f} dex")
    print(f"[Fe/H] ablation shift: {physical_mad[2]:.4f} dex")

    # ── 파일 저장 ──
    report_dir = os.path.join(base_dir, "report", "apogee")
    os.makedirs(report_dir, exist_ok=True)
    out_path   = os.path.join(report_dir, "xai_physics_report.txt")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("      APOGEE Stellar HybridNet XAI Physics Report          \n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Analyzed Samples: {valid_count} Synthetic/Real APOGEE Spectra\n\n")
        f.write("▶ Element Feature Importance Metrics:\n")
        for name, (t, g) in line_scores.items():
            f.write(f"   - {name}:\n")
            f.write(f"     * Temperature Sensitivity : {t:.6f}\n")
            f.write(f"     * Gravity Sensitivity     : {g:.6f}\n\n")
        f.write(f"Proof Ratio: {proof_r:.4f}\n")
        f.write(f"Proof Ratio (Ablated 30D Features): {proof_r_abl:.4f}\n\n")
        f.write("=" * 60 + "\n")
        f.write("▶ Global Weight Attribution Architecture\n")
        f.write("=" * 60 + "\n")
        f.write(f"   - 30D Physical Feature Layer Contribution: {weight_ratio:.4f}%\n\n")
        if per_line:
            f.write("=" * 60 + "\n")
            f.write("▶ Per-Absorption-Line Weight Attribution (Dense Branch)\n")
            f.write("=" * 60 + "\n")
            for ln, pct in per_line:
                f.write(f"   {ln:<12s}  {pct:6.2f}%  {'█' * int(pct/2)}\n")
            f.write("\n")
        f.write("=" * 60 + "\n")
        f.write("▶ Zero-Ablation Sensitivity Analysis\n")
        f.write("=" * 60 + "\n")
        f.write(f"   - T_eff  : {physical_mad[0]:.4f} K\n")
        f.write(f"   - log g  : {physical_mad[1]:.4f} dex\n")
        f.write(f"   - [Fe/H] : {physical_mad[2]:.4f} dex\n")

    print(f"\nReport saved: {out_path}")


if __name__ == "__main__":
    run_xai_line_profile_analysis()
