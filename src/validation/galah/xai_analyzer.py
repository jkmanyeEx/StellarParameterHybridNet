"""
GALAH HybridNet XAI Analysis.

Two complementary attribution methods:

1. Jacobian Sensitivity (∂output/∂input)
   — Single backward pass per sample.
   — Measures local gradient magnitude at the observed spectrum.
   — Fast, but susceptible to gradient saturation in flat regions.

2. Integrated Gradients (Sundararajan et al., 2017)
   — Accumulates gradients along a straight path from a zero-flux
     baseline to the observed spectrum over N interpolation steps.
   — Satisfies Completeness Axiom: attributions sum to
     f(input) - f(baseline) for each output.
   — More robust than plain Jacobian for non-linear models.
   — ~N × cost of Jacobian (default N=50, ~3 s per star on MPS).
"""

import os
import numpy as np
import torch
from tqdm import tqdm

from src.models.galah.hybrid_net import StellarParameterHybridNet
from src.data.galah.extract_features import extract_45d_features_single_star

# ── re-export for GUI ─────────────────────────────────────────────────────────
def extract_30d_features_live_eval(wave, norm_flux):
    return extract_45d_features_single_star(wave, norm_flux)


# ── Architecture constants ────────────────────────────────────────────────────
CNN_BRANCH_DIM   = 6400
DENSE_BRANCH_DIM = 128
FUSION_DIM       = CNN_BRANCH_DIM + DENSE_BRANCH_DIM   # 6528

LINE_NAMES_45D = [
    "H_beta",    "Fe_I_4882", "Mg_I_4703", "Ba_II_4897",
    "Fe_I_5662", "Mg_I_5711", "Fe_I_5782", "Fe_I_5862",
    "H_alpha",   "Fe_I_6495", "Ca_I_6499", "Li_I_6708",
    "Fe_I_7748", "K_I_7699",  "O_I_7772",
]


# ── Weight attribution helpers ────────────────────────────────────────────────

def calculate_per_line_weight_attribution(model):
    """L1 weight share of each 45D absorption line in the Dense branch."""
    try:
        first_weight = None
        for name, param in model.named_parameters():
            if "feature_branch" in name and "weight" in name and param.ndim == 2:
                if param.shape[1] == 45:
                    first_weight = param.detach().cpu().numpy()
                    break
        if first_weight is None:
            print("   [XAI] WARNING: Dense branch Linear(45,*) not found.")
            return []
        total_mag = np.sum(np.abs(first_weight))
        if total_mag == 0:
            return [(n, 0.0) for n in LINE_NAMES_45D]
        results = []
        for i, name in enumerate(LINE_NAMES_45D):
            pct = np.sum(np.abs(first_weight[:, i*3:(i+1)*3])) / total_mag * 100
            results.append((name, float(pct)))
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    except Exception as e:
        print(f"   [XAI] Per-line attribution error: {e}")
        return []


def calculate_eval_model_weight_ratio(model):
    """
    Compute learned vs nominal dimension ratio at the first post-fusion layer.
    Returns (learned_ratio %, nominal_ratio %).
    """
    try:
        target = None
        for name, param in model.named_parameters():
            if "weight" in name.lower() and param.ndim == 2 \
                    and param.shape[1] == FUSION_DIM:
                target = param.detach().cpu().numpy()
                break
        if target is None:
            print(f"   [XAI] ERROR: no Linear(*, {FUSION_DIM}) found.")
            return -1.0, -1.0
        total     = np.sum(np.abs(target))
        if total == 0:
            return 0.0, 0.0
        cnn_mag   = np.sum(np.abs(target[:, :CNN_BRANCH_DIM]))
        dense_mag = np.sum(np.abs(target[:, CNN_BRANCH_DIM:]))
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


# ── Integrated Gradients ──────────────────────────────────────────────────────

def integrated_gradients_galah(model, norm_flux_4arm, feat_tensor,
                                param_idx, device, n_steps=50):
    """
    Compute Integrated Gradients for a single GALAH spectrum.

    Integrates ∂f_{param_idx}/∂x along a straight path from
    baseline (zero flux) to norm_flux_4arm over n_steps steps.

    Satisfies Completeness:
      sum(IG) ≈ f(input) - f(baseline)   for the chosen output.

    Parameters
    ----------
    norm_flux_4arm : np.ndarray  shape (4, 4000)
    feat_tensor    : torch.Tensor shape (1, 45)   — held fixed
    param_idx      : int  0=T_eff, 1=log g, 2=[Fe/H]
    n_steps        : int  interpolation steps (50 is sufficient)

    Returns
    -------
    ig : np.ndarray  shape (4, 4000)
    delta : float    completeness residual |sum(IG) - Δf|  (sanity check)
    """
    model.eval()
    x      = torch.from_numpy(norm_flux_4arm).float().unsqueeze(0).to(device)  # (1,4,4000)
    x_base = torch.zeros_like(x)

    # Prediction difference (completeness target)
    with torch.no_grad():
        f_input    = model(x,      feat_tensor)[0, param_idx].item()
        f_baseline = model(x_base, feat_tensor)[0, param_idx].item()
    delta_f = f_input - f_baseline

    # Accumulate gradients along the interpolation path
    grad_acc = torch.zeros_like(x)   # (1, 4, 4000)
    for k in range(n_steps):
        alpha      = (k + 0.5) / n_steps          # midpoint rule
        x_interp   = (x_base + alpha * (x - x_base)).detach().requires_grad_(True)
        pred       = model(x_interp, feat_tensor)
        g          = torch.zeros_like(pred)
        g[0, param_idx] = 1.0
        pred.backward(g)
        grad_acc  += x_interp.grad.detach()

    # IG = (input - baseline) × mean_gradient
    ig_tensor = (x - x_base) * grad_acc / n_steps
    ig        = ig_tensor.squeeze(0).cpu().numpy()   # (4, 4000)

    # Completeness check: sum(IG) should ≈ delta_f
    delta_check = abs(ig.sum() - delta_f)

    return ig, delta_check


# ── Main XAI pipeline ─────────────────────────────────────────────────────────

def run_xai_line_profile_analysis(num_samples=100, ig_steps=50):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n{'='*70}")
    print("  GALAH XAI — Jacobian + Integrated Gradients Analysis")
    print(f"{'='*70}")
    print(f"  Compute device : {device}")
    print(f"  Samples        : {num_samples}")
    print(f"  IG steps       : {ig_steps}")

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    proc_dir = os.path.join(base_dir, "data", "galah", "processed")

    for fname, label in [
        ("label_stats.npy",   "label_stats.npy"),
        ("X_flux_clean.npy",  "preprocessed flux"),
        ("standard_wave.npy", "standard_wave.npy"),
    ]:
        p = os.path.join(proc_dir, fname)
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"{label} not found at: {p}\n"
                "Execute the GALAH preprocessing and training pipelines first."
            )

    _ls       = np.load(os.path.join(proc_dir, "label_stats.npy"))
    LABEL_STD = _ls[1].astype(np.float32)
    print(f"   [XAI] Label statistics loaded — std={LABEL_STD}")

    weights_path = os.path.join(base_dir, "weights", "galah", "stellar_hybrid_model.pth")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Model weights not found at: {weights_path}\n"
            "Execute the GALAH training pipeline first."
        )
    model = StellarParameterHybridNet(use_features=True).to(device)
    ckpt  = torch.load(weights_path, map_location=device)
    if isinstance(ckpt, dict) and 'model_state' in ckpt:
        model.load_state_dict(ckpt['model_state'])
    else:
        model.load_state_dict(ckpt)
    print(f"   [XAI] Weights loaded from: {weights_path}")
    model.eval()

    X_flux_all = np.load(os.path.join(proc_dir, "X_flux_clean.npy"))
    wave_grid  = np.load(os.path.join(proc_dir, "standard_wave.npy"))

    total_available = X_flux_all.shape[0]
    actual_samples  = min(num_samples, total_available)
    print(f"   [XAI] Spectra available: {total_available} | Using: {actual_samples}\n")

    np.random.seed(42)
    sample_indices = np.random.choice(total_available, size=actual_samples, replace=False)

    absorption_lines = {
        "H-alpha (CCD3)": (6513.0, 6613.0, 2),
        "Mg-I-b (CCD2)":  (5680.0, 5740.0, 1),
        "H-beta  (CCD1)": (4830.0, 4890.0, 0),
    }

    # Accumulators
    jac_acc         = np.zeros((3, 4, 4000))
    jac_acc_ablated = np.zeros((3, 4, 4000))
    ig_acc          = np.zeros((3, 4, 4000))   # Integrated Gradients
    baseline_preds, ablated_preds = [], []
    ig_completeness_errors = []
    valid_count = 0

    print(f"   [XAI] Running Jacobian + Integrated Gradients "
          f"over {actual_samples} spectra...\n")

    for idx in tqdm(sample_indices, desc="GALAH XAI (Jacobian + IG)"):
        raw_flux = X_flux_all[idx]                          # (4, 4000)
        f_mean   = np.mean(raw_flux, axis=1, keepdims=True)
        f_std    = np.std(raw_flux,  axis=1, keepdims=True) + 1e-8
        norm_flux = np.clip((raw_flux - f_mean) / f_std, -3.0, 3.0)

        features_45d = extract_45d_features_single_star(wave_grid, raw_flux)
        feat_tensor  = torch.from_numpy(features_45d).float().unsqueeze(0).to(device)
        zero_feat    = torch.zeros_like(feat_tensor)
        norm_flux_t  = torch.from_numpy(norm_flux).float()

        # ── Jacobian (normal) ────────────────────────────────────────────────
        input_base = norm_flux_t.unsqueeze(0).to(device).requires_grad_(True)
        pred       = model(input_base, feat_tensor)
        baseline_preds.append(pred.detach().cpu().numpy()[0])
        for p_idx in range(3):
            input_base.grad = None
            g = torch.zeros_like(pred); g[0, p_idx] = 1.0
            pred.backward(g, retain_graph=True)
            jac_acc[p_idx] += np.abs(input_base.grad.cpu().numpy()[0])

        # ── Jacobian (ablated) ───────────────────────────────────────────────
        input_abl = norm_flux_t.unsqueeze(0).to(device).requires_grad_(True)
        pred_abl  = model(input_abl, zero_feat)
        ablated_preds.append(pred_abl.detach().cpu().numpy()[0])
        for p_idx in range(3):
            input_abl.grad = None
            g = torch.zeros_like(pred_abl); g[0, p_idx] = 1.0
            pred_abl.backward(g, retain_graph=True)
            jac_acc_ablated[p_idx] += np.abs(input_abl.grad.cpu().numpy()[0])

        # ── Integrated Gradients ─────────────────────────────────────────────
        star_ce = 0.0
        for p_idx in range(3):
            ig, ce = integrated_gradients_galah(
                model, norm_flux, feat_tensor,
                param_idx=p_idx, device=device, n_steps=ig_steps
            )
            ig_acc[p_idx] += ig          # signed attribution
            star_ce        += ce
        ig_completeness_errors.append(star_ce / 3.0)

        valid_count += 1

    # ── Normalise accumulators ────────────────────────────────────────────────
    mean_jac         = jac_acc         / max(valid_count, 1)
    mean_jac_ablated = jac_acc_ablated / max(valid_count, 1)
    mean_ig          = ig_acc          / max(valid_count, 1)   # signed mean IG

    mean_ce = float(np.mean(ig_completeness_errors))
    print(f"\n   [IG] Mean completeness error : {mean_ce:.6f}  "
          f"({'good' if mean_ce < 0.01 else 'acceptable' if mean_ce < 0.05 else 'high — increase ig_steps'})")

    # ── Ablation shift ────────────────────────────────────────────────────────
    baseline_preds = np.array(baseline_preds)
    ablated_preds  = np.array(ablated_preds)
    physical_mad   = np.mean(np.abs(baseline_preds - ablated_preds), axis=0) * LABEL_STD

    # ── Jacobian line scores ──────────────────────────────────────────────────
    line_scores_jac = {}
    for name, (lo, hi, arm_idx) in absorption_lines.items():
        wave = wave_grid[arm_idx]
        mask = (wave >= lo) & (wave <= hi)
        line_scores_jac[name] = (
            float(np.mean(mean_jac[0, arm_idx, mask])),   # T_eff sensitivity
            float(np.mean(mean_jac[1, arm_idx, mask])),   # log g  sensitivity
        )

    # ── IG line scores (|mean IG| per pixel in line window) ──────────────────
    line_scores_ig = {}
    for name, (lo, hi, arm_idx) in absorption_lines.items():
        wave = wave_grid[arm_idx]
        mask = (wave >= lo) & (wave <= hi)
        line_scores_ig[name] = (
            float(np.mean(np.abs(mean_ig[0, arm_idx, mask]))),
            float(np.mean(np.abs(mean_ig[1, arm_idx, mask]))),
        )

    # ── Proof ratios ──────────────────────────────────────────────────────────
    wave_ccd3  = wave_grid[2]
    ha_mask    = (wave_ccd3 >= 6513) & (wave_ccd3 <= 6613)
    cont_mask  = ~ha_mask

    # Jacobian proof ratio
    bg_jac      = float(np.mean(mean_jac[0, 2, cont_mask]))
    proof_r_jac = line_scores_jac["H-alpha (CCD3)"][0] / (bg_jac + 1e-8)

    bg_jac_abl  = float(np.mean(mean_jac_ablated[0, 2, cont_mask]))
    ha_abl      = float(np.mean(mean_jac_ablated[0, 2, ha_mask]))
    proof_r_abl = ha_abl / (bg_jac_abl + 1e-8)

    # IG proof ratio (|IG| in H-alpha vs continuum)
    bg_ig      = float(np.mean(np.abs(mean_ig[0, 2, cont_mask])))
    proof_r_ig = line_scores_ig["H-alpha (CCD3)"][0] / (bg_ig + 1e-8)

    # ── Weight attribution ────────────────────────────────────────────────────
    weight_ratio, nominal_ratio = calculate_eval_model_weight_ratio(model)
    per_line = calculate_per_line_weight_attribution(model)

    # ── Terminal summary ──────────────────────────────────────────────────────
    param_names = ["T_eff", "log g"]
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

    print(f"\n  45D branch weight (learned)     : {weight_ratio:.2f}%")
    print(f"  45D branch dim   (nominal)      : {DENSE_BRANCH_DIM}/{FUSION_DIM} = {nominal_ratio:.2f}%")
    print(f"  Learned / Nominal               : x{weight_ratio/nominal_ratio:.2f}")
    print(f"\n  Ablation shift  T_eff           : {physical_mad[0]:.4f} K")
    print(f"  Ablation shift  log g           : {physical_mad[1]:.4f} dex")
    print(f"  Ablation shift  [Fe/H]          : {physical_mad[2]:.4f} dex")

    # ── Save report ───────────────────────────────────────────────────────────
    report_dir = os.path.join(base_dir, "report", "galah")
    os.makedirs(report_dir, exist_ok=True)
    out_path   = os.path.join(report_dir, "xai_physics_report.txt")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("  GALAH Stellar HybridNet — XAI Physics Report\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"  Analyzed spectra         : {valid_count}\n")
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

        f.write("   Comparison — Jacobian vs. Integrated Gradients:\n")
        f.write(f"   {'Line':<35} {'Jac T_eff':>12} {'IG T_eff':>12}\n")
        f.write(f"   {'-'*60}\n")
        for name in line_scores_jac:
            jt = line_scores_jac[name][0]
            it = line_scores_ig[name][0]
            f.write(f"   {name:<35} {jt:>12.6f} {it:>12.6f}\n")
        f.write("\n")

        f.write("=" * 70 + "\n")
        f.write("▶ Section 3 — Global Weight Attribution\n")
        f.write("=" * 70 + "\n")
        f.write(f"   Architecture (GALAH):\n")
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
            f.write(f"   dimensions — {weight_ratio/nominal_ratio:.2f}x upweighted by training.\n\n")
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
