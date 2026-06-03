import os
import numpy as np
import torch
from tqdm import tqdm

from src.models.mastar.hybrid_net import StellarParameterHybridNet
from src.data.mastar.extract_features import extract_30d_features_single_star


def extract_30d_features_live_eval(wave, norm_flux):
    """
    Thin wrapper — delegates to the canonical extract_30d_features_single_star.
    Guarantees train/inference feature parity (same maxfev, fallback, windows).
    """
    return extract_30d_features_single_star(wave, norm_flux)


# ── 아키텍처 차원 상수 ────────────────────────────────────────────────────────
CNN_BRANCH_DIM   = 1216   # 64ch x 19 (AdaptiveAvgPool1d(19))
DENSE_BRANCH_DIM = 128
FUSION_DIM       = CNN_BRANCH_DIM + DENSE_BRANCH_DIM  # 1344

LINE_NAMES_30D = [
    "H_alpha",   "H_beta",    "H_gamma",   "H_delta",
    "Ca_II_K",   "Ca_II_H",   "Mg_I_b",
    "Fe_I_5270", "Fe_I_4383", "Na_I",
]


def calculate_per_line_weight_attribution(model):
    """
    Measure each absorption line's contribution through the Dense branch.
    Dense branch first layer: Linear(30, 128).
    Each line owns 3 consecutive input dims [EW, FWHM, depth].
    Returns list of (line_name, percent) sorted descending.
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
    Concat order: [CNN:1216 | Dense:128] = 1344 total.
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


def run_xai_line_profile_analysis(num_samples=1000):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[XAI Engine] Initializing on: {device}")

    base_dir  = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    proc_dir  = os.path.join(base_dir, "data", "mastar", "processed")

    # ── 정규화 통계: label_stats.npy에서 로드 (ablation 물리 단위 변환용) ────────
    _ls_path = os.path.join(proc_dir, "label_stats.npy")
    if os.path.exists(_ls_path):
        _ls       = np.load(_ls_path)
        LABEL_STD = _ls[1].astype(np.float32)
        print(f"   [XAI] label_stats loaded: std={LABEL_STD}")
    else:
        print("   [XAI] WARNING: label_stats.npy not found — "
              "ablation shifts will use fallback std.")
        LABEL_STD = np.array([998.064880, 1.081975, 0.723029], dtype=np.float32)

    # ── 모델 로드 ──────────────────────────────────────────────────────────────
    weights_path = os.path.join(base_dir, "weights", "mastar", "stellar_hybrid_model.pth")
    model = StellarParameterHybridNet().to(device)
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

    # ── MaStar 스펙트럼 로드 ───────────────────────────────────────────────────
    flux_path = os.path.join(proc_dir, "X_flux_clean.npy")
    wave_path = os.path.join(proc_dir, "standard_wave.npy")
    if not os.path.exists(flux_path):
        raise FileNotFoundError(
            f"Preprocessed flux not found: {flux_path}\n"
            "Run src/data/preprocess_flux.py first."
        )

    print("Loading real MaStar spectra for XAI analysis...")
    X_flux_all = np.load(flux_path)
    wave_grid  = np.load(wave_path) if os.path.exists(wave_path) \
                 else np.linspace(3650.0, 10250.0, X_flux_all.shape[1])

    total_available = X_flux_all.shape[0]
    actual_samples  = min(num_samples, total_available)
    print(f"   > Dataset: {total_available} stars, using {actual_samples} for XAI.")

    np.random.seed(42)
    sample_indices = np.random.choice(total_available, size=actual_samples, replace=False)

    absorption_lines = {
        "H-alpha (Hydrogen Balmer)": (6513.0, 6613.0),
        "Mg-b Triplet (Magnesium)":  (5140.0, 5200.0),
        "Na-D Doublet (Sodium)":     (5860.0, 5920.0),
        "H-beta (Hydrogen Balmer)":  (4830.0, 4890.0),
    }

    n_pixels             = X_flux_all.shape[1]
    jac_acc              = np.zeros((3, n_pixels))
    jac_acc_ablated      = np.zeros((3, n_pixels))
    baseline_preds       = []
    ablated_preds        = []
    valid_count          = 0

    print(f"Calculating Jacobians over {actual_samples} spectra...\n")

    for idx in tqdm(sample_indices, desc="Jacobian XAI"):
        raw_flux  = X_flux_all[idx]
        f_mean    = np.mean(raw_flux)
        f_std     = np.std(raw_flux) + 1e-8
        norm_flux = np.clip((raw_flux - f_mean) / f_std, -3.0, 3.0).reshape(1, -1)

        features_30d = extract_30d_features_live_eval(wave_grid, raw_flux)
        feat_tensor  = torch.from_numpy(features_30d).float().unsqueeze(0).to(device)
        zero_feat    = torch.zeros_like(feat_tensor)
        norm_flux_t  = torch.from_numpy(norm_flux).float()

        # ── Baseline Jacobian ────────────────────────────────────────────────
        input_base = norm_flux_t.unsqueeze(1).to(device).requires_grad_(True)
        pred = model(input_base, feat_tensor)
        baseline_preds.append(pred.detach().cpu().numpy()[0])
        for p_idx in range(3):
            input_base.grad = None
            g = torch.zeros_like(pred); g[0, p_idx] = 1.0
            pred.backward(g, retain_graph=(p_idx < 2))
            jac_acc[p_idx] += np.abs(input_base.grad.cpu().numpy()[0, 0])

        # ── Ablated Jacobian (30D zeroed) ────────────────────────────────────
        input_abl = norm_flux_t.unsqueeze(1).to(device).requires_grad_(True)
        pred_abl  = model(input_abl, zero_feat)
        ablated_preds.append(pred_abl.detach().cpu().numpy()[0])
        for p_idx in range(3):
            input_abl.grad = None
            g = torch.zeros_like(pred_abl); g[0, p_idx] = 1.0
            pred_abl.backward(g, retain_graph=(p_idx < 2))
            jac_acc_ablated[p_idx] += np.abs(input_abl.grad.cpu().numpy()[0, 0])

        valid_count += 1

    mean_jac         = jac_acc         / max(valid_count, 1)
    mean_jac_ablated = jac_acc_ablated / max(valid_count, 1)

    baseline_preds = np.array(baseline_preds)
    ablated_preds  = np.array(ablated_preds)
    mad            = np.mean(np.abs(baseline_preds - ablated_preds), axis=0)
    physical_mad   = mad * LABEL_STD   # now uses correct training stats

    # ── 리포트 계산 ───────────────────────────────────────────────────────────
    line_scores = {}
    for name, (lo, hi) in absorption_lines.items():
        mask = (wave_grid >= lo) & (wave_grid <= hi)
        line_scores[name] = (
            float(np.mean(mean_jac[0, mask])),
            float(np.mean(mean_jac[1, mask])),
        )

    cont_mask  = ~((wave_grid >= 6513) & (wave_grid <= 6613))
    bg         = float(np.mean(mean_jac[0, cont_mask]))
    proof_r    = line_scores["H-alpha (Hydrogen Balmer)"][0] / (bg + 1e-8)

    bg_abl     = float(np.mean(mean_jac_ablated[0, cont_mask]))
    ha_abl     = float(np.mean(mean_jac_ablated[0, (wave_grid >= 6513) & (wave_grid <= 6613)]))
    proof_r_abl = ha_abl / (bg_abl + 1e-8)

    weight_ratio = calculate_eval_model_weight_ratio(model)
    per_line     = calculate_per_line_weight_attribution(model)

    # ── 터미널 출력 ───────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    for name, (t, g) in line_scores.items():
        print(f"{name:<35} T={t:.5f}  g={g:.5f}")
    print(f"\nProof Ratio (Normal):  {proof_r:.4f}")
    print(f"Proof Ratio (Ablated): {proof_r_abl:.4f}")
    print(f"30D branch weight:     {weight_ratio:.2f}%")
    print(f"T_eff ablation shift:  {physical_mad[0]:.4f} K")
    print(f"log g ablation shift:  {physical_mad[1]:.4f} dex")
    print(f"[Fe/H] ablation shift: {physical_mad[2]:.4f} dex")

    # ── 파일 저장 ─────────────────────────────────────────────────────────────
    report_dir = os.path.join(base_dir, "report", "mastar")
    os.makedirs(report_dir, exist_ok=True)
    out_path   = os.path.join(report_dir, "xai_physics_report.txt")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("      SDSS DR17 Stellar HybridNet XAI Physics Report        \n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Analyzed Samples: {valid_count} Real MaStar Spectra "
                f"(Cumulative Jacobian Gradients)\n\n")
        f.write("▶ Element Feature Importance Metrics (Hypothesis 2):\n")
        for name, (t, g) in line_scores.items():
            f.write(f"   - {name}:\n")
            f.write(f"     * Temperature Sensitivity : {t:.6f}\n")
            f.write(f"     * Gravity Sensitivity     : {g:.6f}\n\n")
        f.write(f"Hypothesis 2 Proof Ratio: {proof_r:.4f}\n")
        f.write(f"Hypothesis 2 Proof Ratio (Ablated 30D Features): {proof_r_abl:.4f}\n\n")
        f.write("=" * 60 + "\n")
        f.write("▶ Global Weight Attribution Architecture (Hypothesis 1)\n")
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
        f.write("Mean output shift when physical features are artificially zeroed:\n")
        f.write(f"   - T_eff  : {physical_mad[0]:.4f} K\n")
        f.write(f"   - log g  : {physical_mad[1]:.4f} dex\n")
        f.write(f"   - [Fe/H] : {physical_mad[2]:.4f} dex\n")

    print(f"\nReport saved: {out_path}")


if __name__ == "__main__":
    run_xai_line_profile_analysis(num_samples=1000)
