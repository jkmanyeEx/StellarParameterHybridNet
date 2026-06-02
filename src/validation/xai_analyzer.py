import os
import sys
import numpy as np
import torch
from tqdm import tqdm
from scipy.optimize import curve_fit

from ..models.hybrid_net import StellarParameterHybridNet

def gaussian_profile(x, a, x0, sigma, c):
    return c - a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))


def extract_18d_features_live_eval(wave, norm_flux):
    """
    Extract 18D physical feature vector from a single stellar spectrum.
    6 absorption lines x 3 values each: equivalent width, FWHM, depth.
    Falls back to [0, 0, 0] per line if Gaussian fit does not converge.
    """
    target_lines = {
        # ── Balmer series (Teff) ────────────────────────────────────
        "H_alpha": (6563.0, 20),
        "H_beta":  (4861.0, 20),
        "H_gamma": (4340.0, 20),
        "H_delta": (4102.0, 20),
        # ── Calcium (log g + [Fe/H]) ──────────────────────────────
        "Ca_II_K": (3934.0, 15),
        "Ca_II_H": (3968.0, 15),
        # ── Magnesium (log g 전진 지표) ───────────────────────────
        "Mg_I_b":  (5175.0, 20),
        # ── Iron ([Fe/H]) ─────────────────────────────────────────
        "Fe_I_5270": (5270.0, 15),
        "Fe_I_4383": (4383.0, 15),
        # ── Sodium (log g + [Fe/H]) ───────────────────────────────
        "Na_I":    (5892.0, 15),
    }
    feature_vector = []

    for line_name, (center_wave, window_half) in target_lines.items():
        mask  = (wave >= center_wave - window_half) & (wave <= center_wave + window_half)
        w_sub = wave[mask]
        f_sub = norm_flux[mask]

        if len(w_sub) == 0:
            feature_vector.extend([0.0, 0.0, 0.0])
            continue

        p0     = [1.0 - np.min(f_sub), center_wave, 2.0, 1.0]
        bounds = ([0.0, center_wave - 5, 0.1, 0.8], [1.0, center_wave + 5, 10.0, 1.2])

        try:
            popt, _ = curve_fit(gaussian_profile, w_sub, f_sub, p0=p0, bounds=bounds, maxfev=1500)
            a, x0, sigma, c = popt
            fwhm             = 2.355 * np.abs(sigma)
            depth            = a
            dw               = np.gradient(w_sub)
            equivalent_width = max(0.0, np.sum((1.0 - f_sub / c) * dw))
            feature_vector.extend([equivalent_width, fwhm, depth])
        except (RuntimeError, ValueError):
            depth = float(max(0.0, 1.0 - np.min(f_sub)))
            dw = np.gradient(w_sub)
            equivalent_width = float(max(0.0, np.sum((1.0 - f_sub) * dw)))
            
            half_val = 1.0 - (depth / 2.0)
            below_half = np.where(f_sub < half_val)[0]
            if len(below_half) > 1:
                fwhm = float(w_sub[below_half[-1]] - w_sub[below_half[0]])
            else:
                fwhm = 3.0
                
            feature_vector.extend([equivalent_width, fwhm, depth])

    return np.array(feature_vector, dtype=np.float32)


# CNN branch output dim: 64ch * 19 = 1216, Dense branch output dim: 128
# Concat order in fusion.py: cat(cnn_output, feature_output) → [CNN:1216 | Dense:128]
CNN_BRANCH_DIM   = 1216
DENSE_BRANCH_DIM = 128
FUSION_DIM = CNN_BRANCH_DIM + DENSE_BRANCH_DIM  # 1344


def calculate_eval_model_weight_ratio(model):
    """
    Estimate what fraction of the first post-fusion layer's L1 weight magnitude
    comes from the Dense (18D physical feature) branch vs the CNN branch.

    Concat order: [CNN: 320 dims | Dense branch: 128 dims] = 448 total.
    We slice W[:, :320] for CNN and W[:, 320:] for the Dense branch.
    """
    try:
        # Target: the first Linear(448, *) layer — the post-fusion projection
        target_weight_tensor = None
        for name, param in model.named_parameters():
            if ("weight" in name.lower()
                    and param.ndim == 2
                    and param.shape[1] == FUSION_DIM):
                target_weight_tensor = param.detach().cpu().numpy()
                break

        if target_weight_tensor is None:
            print(f"XAI ERROR: no Linear(*, {FUSION_DIM}) weight tensor found — "
                  "check CNN_BRANCH_DIM / DENSE_BRANCH_DIM constants.")
            return -1.00

        total_mag = np.sum(np.abs(target_weight_tensor))
        if total_mag == 0:
            return 0.00

        # Correct slice: Dense branch occupies the LAST 128 columns
        cnn_mag   = np.sum(np.abs(target_weight_tensor[:, :CNN_BRANCH_DIM]))
        dense_mag = np.sum(np.abs(target_weight_tensor[:, CNN_BRANCH_DIM:]))

        ratio_dense = (dense_mag / total_mag) * 100
        ratio_cnn   = (cnn_mag   / total_mag) * 100

        print(f"   [XAI] CNN branch   weight share : {ratio_cnn:.2f}%")
        print(f"   [XAI] Dense branch weight share : {ratio_dense:.2f}%")
        return ratio_dense

    except Exception as e:
        print(f"XAI CRASH: {e}")
        return -1.00


def run_xai_line_profile_analysis(num_samples=1000):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[XAI Engine] Initializing on: {device}")

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    # --- Load model ---
    weights_path = os.path.join(base_dir, "weights", "stellar_hybrid_model.pth")
    model = StellarParameterHybridNet().to(device)
    if os.path.exists(weights_path):
        checkpoint = torch.load(weights_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
            model.load_state_dict(checkpoint['model_state'])
        else:
            model.load_state_dict(checkpoint)
        print(f"   [Weights] Loaded from {weights_path}")
    else:
        print(f"   [WARN] Weights not found — running with random init.")
    model.eval()

    # --- Load real MaStar spectra ---
    data_flux_path = os.path.join(base_dir, "data", "processed", "X_flux_telluric.npy")
    data_wave_path = os.path.join(base_dir, "data", "processed", "standard_wave.npy")
    if not os.path.exists(data_flux_path):
        raise FileNotFoundError(
            f"Preprocessed flux not found: {data_flux_path}\n"
            "Run src/data/preprocess_flux.py first."
        )

    print("Loading real MaStar spectra for XAI analysis...")
    X_flux_all = np.load(data_flux_path)
    wave_grid  = (np.load(data_wave_path)
                  if os.path.exists(data_wave_path)
                  else np.linspace(3650.0, 10250.0, X_flux_all.shape[1]))

    total_available = X_flux_all.shape[0]
    actual_samples  = min(num_samples, total_available)
    print(f"   > Dataset: {total_available} stars available, "
          f"using {actual_samples} random samples for XAI.")

    np.random.seed(42)
    sample_indices = np.random.choice(total_available, size=actual_samples, replace=False)

    # --- Absorption line regions ---
    absorption_lines = {
        "H-alpha (Hydrogen Balmer)": (6513.0, 6613.0),
        "Mg-b Triplet (Magnesium)":  (5140.0, 5200.0),
        "Na-D Doublet (Sodium)":     (5860.0, 5920.0),
        "H-beta (Hydrogen Balmer)":  (4830.0, 4890.0),
    }

    n_pixels             = X_flux_all.shape[1]
    jacobian_accumulator = np.zeros((3, n_pixels))
    jacobian_accumulator_ablated = np.zeros((3, n_pixels))
    valid_count          = 0

    print(f"Calculating Jacobians over {actual_samples} real stellar spectra...\n")
    
    baseline_preds = []
    ablated_preds = []

    for idx in tqdm(sample_indices, desc="Jacobian XAI"):
        raw_flux = X_flux_all[idx]

        f_mean = np.mean(raw_flux)
        f_std  = np.std(raw_flux) + 1e-8
        norm_flux = np.clip((raw_flux - f_mean) / f_std, -3.0, 3.0).reshape(1, -1)

        features_18d = extract_18d_features_live_eval(wave_grid, raw_flux)
        feat_tensor  = torch.from_numpy(features_18d).float().unsqueeze(0).to(device)
        zero_feat    = torch.zeros_like(feat_tensor)
        norm_flux_t  = torch.from_numpy(norm_flux).float()

        # ── Baseline Jacobian ──────────────────────────────────────────────────
        input_base = norm_flux_t.unsqueeze(1).to(device).requires_grad_(True)
        pred = model(input_base, feat_tensor)
        baseline_preds.append(pred.detach().cpu().numpy()[0])

        # 3개 파라미터 gradient를 한 번에 계산
        for param_idx in range(3):
            model.zero_grad()
            input_base.grad = None
            grad_out = torch.zeros_like(pred)
            grad_out[0, param_idx] = 1.0
            pred.backward(grad_out, retain_graph=(param_idx < 2))
            jacobian_accumulator[param_idx] += np.abs(
                input_base.grad.cpu().numpy()[0, 0]
            )

        # ── Ablated Jacobian ──────────────────────────────────────────────────
        input_abl = norm_flux_t.unsqueeze(1).to(device).requires_grad_(True)
        pred_ablated = model(input_abl, zero_feat)
        ablated_preds.append(pred_ablated.detach().cpu().numpy()[0])

        for param_idx in range(3):
            model.zero_grad()
            input_abl.grad = None
            grad_out = torch.zeros_like(pred_ablated)
            grad_out[0, param_idx] = 1.0
            pred_ablated.backward(grad_out, retain_graph=(param_idx < 2))
            jacobian_accumulator_ablated[param_idx] += np.abs(
                input_abl.grad.cpu().numpy()[0, 0]
            )

        valid_count += 1

    mean_jacobian = jacobian_accumulator / max(valid_count, 1)
    mean_jacobian_ablated = jacobian_accumulator_ablated / max(valid_count, 1)

    baseline_preds = np.array(baseline_preds)
    ablated_preds = np.array(ablated_preds)
    mad = np.mean(np.abs(baseline_preds - ablated_preds), axis=0)
    LABEL_STD = np.array([998.064880, 1.081975, 0.723029])
    physical_mad = mad * LABEL_STD

    print("\n" + "=" * 65)
    print(f"{'Target Absorption Line':<30} | {'T_eff Impact':<12} | {'Log g Impact':<12}")
    print("-" * 65)

    line_scores = {}
    for name, (low_w, high_w) in absorption_lines.items():
        mask        = (wave_grid >= low_w) & (wave_grid <= high_w)
        teff_impact = float(np.mean(mean_jacobian[0, mask]))
        logg_impact = float(np.mean(mean_jacobian[1, mask]))
        line_scores[name] = (teff_impact, logg_impact)
        print(f"{name:<30} | {teff_impact:<12.5f} | {logg_impact:<12.5f}")
    print("=" * 65)

    continuum_mask     = ~((wave_grid >= 6513) & (wave_grid <= 6613))
    bg_impact          = float(np.mean(mean_jacobian[0, continuum_mask]))
    h_alpha_impact     = line_scores["H-alpha (Hydrogen Balmer)"][0]
    verification_ratio = h_alpha_impact / (bg_impact + 1e-8)

    bg_impact_ablated = float(np.mean(mean_jacobian_ablated[0, continuum_mask]))
    h_alpha_mask = (wave_grid >= 6513) & (wave_grid <= 6613)
    h_alpha_impact_ablated = float(np.mean(mean_jacobian_ablated[0, h_alpha_mask]))
    verification_ratio_ablated = h_alpha_impact_ablated / (bg_impact_ablated + 1e-8)

    final_18d_weight_ratio = calculate_eval_model_weight_ratio(model)

    print(f"\n[Hypothesis 2] H-alpha region impact is {verification_ratio:.4f}x "
          f"higher than continuum background.")
    print(f"[Hypothesis 2 - Ablated] When 18D features are removed, H-alpha region impact grows to {verification_ratio_ablated:.4f}x "
          f"higher than continuum background.")
    print(f"[Hypothesis 1] 18D Physical Features claim "
          f"{final_18d_weight_ratio:.2f}% of dense layer attention.")

    print("\n" + "=" * 65)
    print("▶ Feature Branch Ablation Impact (Absolute Mean Shift)")
    print("=" * 65)
    print(f"   - T_eff  Prediction Shift: {physical_mad[0]:.4f} K")
    print(f"   - log g  Prediction Shift: {physical_mad[1]:.4f} dex")
    print(f"   - [Fe/H] Prediction Shift: {physical_mad[2]:.4f} dex")

    report_dir = os.path.join(base_dir, "report")
    os.makedirs(report_dir, exist_ok=True)
    out_path = os.path.join(report_dir, "xai_physics_report.txt")
    
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("============================================================\n")
        f.write("      SDSS DR17 Stellar HybridNet XAI Physics Report        \n")
        f.write("============================================================\n\n")
        f.write(f"Analyzed Samples: {valid_count} Real MaStar Spectra "
                f"(Cumulative Jacobian Gradients)\n\n")
        f.write("▶ Element Feature Importance Metrics (Hypothesis 2):\n")
        for name, (t_imp, g_imp) in line_scores.items():
            f.write(f"   - {name}:\n")
            f.write(f"     * Temperature Sensitivity : {t_imp:.6f}\n")
            f.write(f"     * Gravity Sensitivity     : {g_imp:.6f}\n\n")
        f.write(f"Hypothesis 2 Proof Ratio: {verification_ratio:.4f}\n")
        f.write(f"Hypothesis 2 Proof Ratio (Ablated 18D Features): {verification_ratio_ablated:.4f}\n\n")
        f.write("============================================================\n")
        f.write("▶ Global Weight Attribution Architecture (Hypothesis 1)\n")
        f.write("============================================================\n")
        f.write(f"   - 18D Physical Feature Layer Contribution: "
                f"{final_18d_weight_ratio:.4f}%\n\n")
        f.write("============================================================\n")
        f.write("▶ Zero-Ablation Sensitivity Analysis\n")
        f.write("============================================================\n")
        f.write(f"Mean output shift when physical features are artificially zeroed:\n")
        f.write(f"   - T_eff  : {physical_mad[0]:.4f} K\n")
        f.write(f"   - log g  : {physical_mad[1]:.4f} dex\n")
        f.write(f"   - [Fe/H] : {physical_mad[2]:.4f} dex\n")
    print(f"\nReport saved: {out_path}")


if __name__ == "__main__":
    run_xai_line_profile_analysis(num_samples=1000)
