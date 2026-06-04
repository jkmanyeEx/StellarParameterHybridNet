"""
APOGEE HybridNet XAI Analysis.

Two complementary attribution methods:

1. Jacobian Sensitivity (∂output/∂input)
2. Integrated Gradients (Sundararajan et al., 2017)
"""

import os
import re
import numpy as np
import torch
from tqdm import tqdm

from src.models.apogee.hybrid_net import StellarParameterHybridNet
from src.data.apogee.extract_features import extract_30d_features_single_star


def extract_30d_features_live_eval(wave, norm_flux):
    return extract_30d_features_single_star(wave, norm_flux)


CNN_BRANCH_DIM   = 4800
DENSE_BRANCH_DIM = 128
FUSION_DIM       = CNN_BRANCH_DIM + DENSE_BRANCH_DIM

LINE_NAMES_30D = [
    "Fe_I_15200", "Fe_I_15648", "Mg_I_15749",
    "Si_I_15960", "Br_14",      "Fe_I_16040", "Si_I_16094",
    "Si_I_16680", "Al_I_16755", "Br_11",
]


def _report_tag(weights_path):
    """stellar_hybrid_model_n37500.pth -> '_n37500',  *.pth -> ''"""
    stem = os.path.splitext(os.path.basename(weights_path))[0]
    m    = re.search(r'(_n\d+)$', stem)
    return m.group(1) if m else ''


def calculate_per_line_weight_attribution(model):
    try:
        first_weight = None
        for name, param in model.named_parameters():
            if "feature_branch" in name and "weight" in name and param.ndim == 2:
                if param.shape[1] == 30:
                    first_weight = param.detach().cpu().numpy()
                    break
        if first_weight is None:
            print("   [XAI] WARNING: Dense branch Linear(30,*) not found.")
            return []
        total_mag = np.sum(np.abs(first_weight))
        if total_mag == 0:
            return [(n, 0.0) for n in LINE_NAMES_30D]
        results = []
        for i, name in enumerate(LINE_NAMES_30D):
            pct = np.sum(np.abs(first_weight[:, i*3:(i+1)*3])) / total_mag * 100
            results.append((name, float(pct)))
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    except Exception as e:
        print(f"   [XAI] Per-line attribution error: {e}")
        return []


def calculate_eval_model_weight_ratio(model):
    try:
        target = None
        for name, param in model.named_parameters():
            if "weight" in name.lower() and param.ndim == 2 \
                    and param.shape[1] == FUSION_DIM:
                target = param.detach().cpu().numpy()
                break
        if target is None:
            return -1.0, -1.0
        total     = np.sum(np.abs(target))
        if total == 0:
            return 0.0, 0.0
        dense_mag = np.sum(np.abs(target[:, CNN_BRANCH_DIM:]))
        cnn_mag   = np.sum(np.abs(target[:, :CNN_BRANCH_DIM]))
        learned   = float(dense_mag / total * 100)
        nominal   = float(DENSE_BRANCH_DIM / FUSION_DIM * 100)
        print(f"   [XAI] Fusion dim          : {FUSION_DIM} D "
              f"(CNN {CNN_BRANCH_DIM} D + Dense {DENSE_BRANCH_DIM} D)")
        print(f"   [XAI] Nominal dim ratio   : {DENSE_BRANCH_DIM}/{FUSION_DIM} "
              f"= {nominal:.2f}%")
        print(f"   [XAI] CNN  branch L1 share: {cnn_mag/total*100:.2f}%")
        print(f"   [XAI] Dense branch L1 share (learned): {learned:.2f}%")
        print(f"   [XAI] Learned / Nominal   : x{learned/nominal:.2f}")
        return learned, nominal
    except Exception as e:
        print(f"   [XAI] ERROR in weight ratio: {e}")
        return -1.0, -1.0


def integrated_gradients_apogee(model, norm_flux_3arm, feat_tensor,
                                 param_idx, device, n_steps=50):
    model.eval()
    x      = torch.from_numpy(norm_flux_3arm).float().unsqueeze(0).to(device)
    x_base = torch.zeros_like(x)
    with torch.no_grad():
        f_input    = model(x,      feat_tensor)[0, param_idx].item()
        f_baseline = model(x_base, feat_tensor)[0, param_idx].item()
    delta_f  = f_input - f_baseline
    grad_acc = torch.zeros_like(x)
    for k in range(n_steps):
        alpha    = (k + 0.5) / n_steps
        x_interp = (x_base + alpha * (x - x_base)).detach().requires_grad_(True)
        pred     = model(x_interp, feat_tensor)
        g        = torch.zeros_like(pred)
        g[0, param_idx] = 1.0
        pred.backward(g)
        grad_acc += x_interp.grad.detach()
    ig    = ((x - x_base) * grad_acc / n_steps).squeeze(0).cpu().numpy()
    delta = abs(ig.sum() - delta_f)
    return ig, delta


def run_xai_line_profile_analysis(num_samples=100, ig_steps=50, weights_path=None):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n{'='*70}")
    print("  APOGEE XAI — Jacobian + Integrated Gradients Analysis")
    print(f"{'='*70}")
    print(f"  Compute device : {device}")
    print(f"  Samples        : {num_samples}")
    print(f"  IG steps       : {ig_steps}")

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    proc_dir = os.path.join(base_dir, "data", "apogee", "processed")

    for fname, label in [
        ("label_stats.npy",  "label_stats.npy"),
        ("X_flux_clean.npy", "preprocessed flux"),
        ("standard_wave.npy","standard_wave.npy"),
    ]:
        p = os.path.join(proc_dir, fname)
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"{label} not found at: {p}\n"
                "Execute the APOGEE preprocessing and training pipelines first."
            )

    LABEL_STD = np.load(os.path.join(proc_dir, "label_stats.npy"))[1].astype(np.float32)
    print(f"   [XAI] Label statistics loaded — std={LABEL_STD}")

    if weights_path is None:
        weights_path = os.path.join(base_dir, "weights", "apogee", "stellar_hybrid_model.pth")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Model weights not found at: {weights_path}\n"
            "Execute the APOGEE training pipeline first."
        )
    model = StellarParameterHybridNet(use_features=True).to(device)
    ckpt  = torch.load(weights_path, map_location=device)
    if isinstance(ckpt, dict) and 'model_state' in ckpt:
        model.load_state_dict(ckpt['model_state'])
    else:
        model.load_state_dict(ckpt)
    print(f"   [XAI] Weights loaded from: {weights_path}")
    model.eval()

    n_train = None
    meta_path = os.path.join(proc_dir, "train_meta.npy")
    if os.path.exists(meta_path):
        try:
            m = np.load(meta_path)
            n_train = int(m[0])
            print(f"   [XAI] Training set size : {n_train} stars")
        except Exception:
            pass
    if n_train is None:
        try:
            _c = torch.load(weights_path, map_location='cpu')
            if isinstance(_c, dict) and 'n_train' in _c:
                n_train = int(_c['n_train'])
                print(f"   [XAI] Training set size : {n_train} stars")
        except Exception:
            pass

    X_flux_all = np.load(os.path.join(proc_dir, "X_flux_clean.npy"))
    wave_grid  = np.load(os.path.join(proc_dir, "standard_wave.npy"))

    total_available = X_flux_all.shape[0]
    actual_samples  = min(num_samples, total_available)
    print(f"   [XAI] Spectra available: {total_available} | Using: {actual_samples}\n")

    np.random.seed(42)
    sample_indices = np.random.choice(total_available, size=actual_samples, replace=False)

    absorption_lines = {
        "Fe-I-15648 (Blue)": (15610.0, 15690.0, 0),
        "Br-14 (Green)":     (15850.0, 15920.0, 1),
        "Br-11 (Red)":       (16780.0, 16840.0, 2),
    }

    jac_acc         = np.zeros((3, 3, 2800))
    jac_acc_ablated = np.zeros((3, 3, 2800))
    ig_acc          = np.zeros((3, 3, 2800))
    baseline_preds, ablated_preds, ig_completeness_errors = [], [], []
    valid_count = 0

    print(f"   [XAI] Running Jacobian + IG over {actual_samples} spectra...\n")

    for idx in tqdm(sample_indices, desc="APOGEE XAI (Jacobian + IG)"):
        raw_flux  = X_flux_all[idx]
        f_mean    = np.mean(raw_flux, axis=1, keepdims=True)
        f_std     = np.std(raw_flux,  axis=1, keepdims=True) + 1e-8
        norm_flux = np.clip((raw_flux - f_mean) / f_std, -3.0, 3.0)
        features_30d = extract_30d_features_live_eval(wave_grid, raw_flux)
        feat_tensor  = torch.from_numpy(features_30d).float().unsqueeze(0).to(device)
        zero_feat    = torch.zeros_like(feat_tensor)
        norm_flux_t  = torch.from_numpy(norm_flux).float()

        input_base = norm_flux_t.unsqueeze(0).to(device).requires_grad_(True)
        pred = model(input_base, feat_tensor)
        baseline_preds.append(pred.detach().cpu().numpy()[0])
        for p_idx in range(3):
            input_base.grad = None
            g = torch.zeros_like(pred); g[0, p_idx] = 1.0
            pred.backward(g, retain_graph=True)
            jac_acc[p_idx] += np.abs(input_base.grad.cpu().numpy()[0])

        input_abl = norm_flux_t.unsqueeze(0).to(device).requires_grad_(True)
        pred_abl  = model(input_abl, zero_feat)
        ablated_preds.append(pred_abl.detach().cpu().numpy()[0])
        for p_idx in range(3):
            input_abl.grad = None
            g = torch.zeros_like(pred_abl); g[0, p_idx] = 1.0
            pred_abl.backward(g, retain_graph=True)
            jac_acc_ablated[p_idx] += np.abs(input_abl.grad.cpu().numpy()[0])

        star_ce = 0.0
        for p_idx in range(3):
            ig, ce = integrated_gradients_apogee(
                model, norm_flux, feat_tensor,
                param_idx=p_idx, device=device, n_steps=ig_steps)
            ig_acc[p_idx] += ig
            star_ce += ce
        ig_completeness_errors.append(star_ce / 3.0)
        valid_count += 1

    mean_jac         = jac_acc         / max(valid_count, 1)
    mean_jac_ablated = jac_acc_ablated / max(valid_count, 1)
    mean_ig          = ig_acc          / max(valid_count, 1)
    mean_ce = float(np.mean(ig_completeness_errors))
    print(f"\n   [IG] Mean completeness error : {mean_ce:.6f}  "
          f"({'good' if mean_ce < 0.01 else 'acceptable' if mean_ce < 0.05 else 'high'})")

    physical_mad = np.mean(np.abs(
        np.array(baseline_preds) - np.array(ablated_preds)), axis=0) * LABEL_STD

    line_scores_jac, line_scores_ig = {}, {}
    for name, (lo, hi, arm_idx) in absorption_lines.items():
        wave = wave_grid[arm_idx]
        mask = (wave >= lo) & (wave <= hi)
        line_scores_jac[name] = (float(np.mean(mean_jac[0, arm_idx, mask])),
                                  float(np.mean(mean_jac[1, arm_idx, mask])))
        line_scores_ig[name]  = (float(np.mean(np.abs(mean_ig[0, arm_idx, mask]))),
                                  float(np.mean(np.abs(mean_ig[1, arm_idx, mask]))))

    wave_green = wave_grid[1]
    br14_mask  = (wave_green >= 15850) & (wave_green <= 15920)
    cont_mask  = ~br14_mask
    proof_r_jac = line_scores_jac["Br-14 (Green)"][0] / (float(np.mean(mean_jac[0, 1, cont_mask])) + 1e-8)
    proof_r_abl = float(np.mean(mean_jac_ablated[0, 1, br14_mask])) / (float(np.mean(mean_jac_ablated[0, 1, cont_mask])) + 1e-8)
    proof_r_ig  = line_scores_ig["Br-14 (Green)"][0] / (float(np.mean(np.abs(mean_ig[0, 1, cont_mask]))) + 1e-8)

    weight_ratio, nominal_ratio = calculate_eval_model_weight_ratio(model)
    per_line = calculate_per_line_weight_attribution(model)

    print("\n" + "=" * 70)
    print("  Jacobian Line Sensitivity")
    print("=" * 70)
    for name, (t, g) in line_scores_jac.items():
        print(f"  {name:<35} T_eff={t:.5f}  log_g={g:.5f}")
    print(f"\n  Proof Ratio (Jacobian, normal)  : {proof_r_jac:.4f}")
    print(f"  Proof Ratio (Jacobian, ablated) : {proof_r_abl:.4f}")
    print("\n" + "=" * 70)
    print("  Integrated Gradients Line Attribution")
    print("=" * 70)
    for name, (t, g) in line_scores_ig.items():
        print(f"  {name:<35} T_eff={t:.5f}  log_g={g:.5f}")
    print(f"\n  Proof Ratio (IG)                : {proof_r_ig:.4f}")
    print(f"  Mean completeness error (IG)    : {mean_ce:.6f}")
    print(f"\n  30D branch weight (learned)     : {weight_ratio:.2f}%")
    print(f"  30D branch dim   (nominal)      : {DENSE_BRANCH_DIM}/{FUSION_DIM} = {nominal_ratio:.2f}%")
    print(f"  Learned / Nominal               : x{weight_ratio/nominal_ratio:.2f}")
    print(f"\n  Ablation shift  T_eff           : {physical_mad[0]:.4f} K")
    print(f"  Ablation shift  log g           : {physical_mad[1]:.4f} dex")
    print(f"  Ablation shift  [Fe/H]          : {physical_mad[2]:.4f} dex")

    # ── Save report — filename includes weight tag ────────────────────────────
    tag        = _report_tag(weights_path)
    report_dir = os.path.join(base_dir, "report", "apogee")
    os.makedirs(report_dir, exist_ok=True)
    out_path   = os.path.join(report_dir, f"xai_physics_report{tag}.txt")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("  APOGEE Stellar HybridNet — XAI Physics Report\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"  Weights file             : {os.path.basename(weights_path)}\n")
        f.write(f"  Analyzed spectra         : {valid_count}\n")
        if n_train is not None:
            f.write(f"  Training set size        : {n_train} stars\n")
        f.write(f"  Jacobian method          : ∂output/∂input (single backward pass)\n")
        f.write(f"  IG method                : Integrated Gradients "
                f"(Sundararajan et al., 2017)\n")
        f.write(f"  IG interpolation steps   : {ig_steps}\n")
        f.write(f"  IG mean completeness err : {mean_ce:.6f}\n\n")

        f.write("=" * 70 + "\n")
        f.write("▶ Section 1 — Jacobian Line Sensitivity\n")
        f.write("=" * 70 + "\n")
        for name, (t, g) in line_scores_jac.items():
            f.write(f"   {name}:\n")
            f.write(f"     T_eff sensitivity : {t:.6f}\n")
            f.write(f"     log g sensitivity : {g:.6f}\n\n")
        f.write(f"   Proof Ratio (normal)  : {proof_r_jac:.4f}\n")
        f.write(f"   Proof Ratio (ablated) : {proof_r_abl:.4f}\n\n")

        f.write("=" * 70 + "\n")
        f.write("▶ Section 2 — Integrated Gradients Line Attribution\n")
        f.write("=" * 70 + "\n")
        f.write("   Method: attribution = (input - baseline) × mean(∂f/∂x) "
                "along interpolation path.\n")
        f.write("   Baseline: zero-flux (all pixels = 0).\n")
        f.write("   Completeness axiom: Σ IG ≈ f(input) - f(baseline).\n\n")
        for name, (t, g) in line_scores_ig.items():
            f.write(f"   {name}:\n")
            f.write(f"     |IG| T_eff : {t:.6f}\n")
            f.write(f"     |IG| log g : {g:.6f}\n\n")
        f.write(f"   Proof Ratio (IG)         : {proof_r_ig:.4f}\n")
        f.write(f"   Mean completeness error  : {mean_ce:.6f}\n\n")
        f.write(f"   {'Line':<35} {'Jac T_eff':>12} {'IG T_eff':>12}\n")
        f.write(f"   {'-'*60}\n")
        for name in line_scores_jac:
            f.write(f"   {name:<35} {line_scores_jac[name][0]:>12.6f} "
                    f"{line_scores_ig[name][0]:>12.6f}\n")
        f.write("\n")

        f.write("=" * 70 + "\n")
        f.write("▶ Section 3 — Global Weight Attribution\n")
        f.write("=" * 70 + "\n")
        f.write(f"   Architecture (APOGEE):\n")
        f.write(f"     CNN  branch output : {CNN_BRANCH_DIM} D\n")
        f.write(f"     Dense branch output: {DENSE_BRANCH_DIM} D\n")
        f.write(f"     Fusion (concat)    : {FUSION_DIM} D\n\n")
        f.write(f"   Nominal dim ratio  : {DENSE_BRANCH_DIM}/{FUSION_DIM} "
                f"= {nominal_ratio:.4f}%\n")
        f.write(f"   Learned L1 ratio   : {weight_ratio:.4f}%\n")
        f.write(f"   Learned / Nominal  : x{weight_ratio/nominal_ratio:.2f}\n")
        if weight_ratio > nominal_ratio:
            f.write(f"\n   The Dense branch occupies {weight_ratio:.2f}% of post-fusion\n")
            f.write(f"   L1 weight despite contributing only {nominal_ratio:.2f}% of\n")
            f.write(f"   dimensions — {weight_ratio/nominal_ratio:.2f}x upweighted.\n\n")
        else:
            f.write(f"\n   {weight_ratio:.2f}% learned vs {nominal_ratio:.2f}% nominal.\n\n")

        if per_line:
            f.write("=" * 70 + "\n")
            f.write("▶ Section 4 — Per-Line Weight Attribution (Dense Branch)\n")
            f.write("=" * 70 + "\n")
            for ln, pct in per_line:
                f.write(f"   {ln:<14s}  {pct:6.2f}%  {'█' * int(pct/2)}\n")
            f.write("\n")

        f.write("=" * 70 + "\n")
        f.write("▶ Section 5 — Zero-Ablation Sensitivity\n")
        f.write("=" * 70 + "\n")
        f.write(f"   T_eff  shift : {physical_mad[0]:.4f} K\n")
        f.write(f"   log g  shift : {physical_mad[1]:.4f} dex\n")
        f.write(f"   [Fe/H] shift : {physical_mad[2]:.4f} dex\n")

    print(f"\n   [XAI] Report saved to: {out_path}")


if __name__ == "__main__":
    run_xai_line_profile_analysis(num_samples=100, ig_steps=50)
