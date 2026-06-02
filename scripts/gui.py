import os
import sys
import numpy as np
import torch
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['SF Pro Display', 'Helvetica Neue', 'Helvetica', 'Arial', 'sans-serif'],
    'font.size': 7.0, 'axes.labelsize': 7.0, 'axes.titlesize': 9.0,
    'xtick.labelsize': 6.0, 'ytick.labelsize': 6.0,
    'legend.fontsize': 6.0, 'figure.titlesize': 9.0
})

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.hybrid_net import StellarParameterHybridNet
from src.validation.eval_core import align_wavelength_resolution, continuum_normalize
from src.validation.xai_analyzer import extract_18d_features_live_eval
from src.validation.error_calculator import collect_spec_fits_files, load_spectra_from_fits_list

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TARGET_DATASET_DIR = os.path.join(BASE_DIR, "data", "validation_dataset")

# ── 파장 그리드 ───────────────────────────────────────────────────────────────
_wave_path = os.path.join(BASE_DIR, "data", "processed", "standard_wave.npy")
WAVE_GRID  = np.load(_wave_path) if os.path.exists(_wave_path) \
             else np.linspace(3650.0, 10250.0, 4563)

# ── 정규화 통계: label_stats / feature_stats 로드 ─────────────────────────────
_proc_dir = os.path.join(BASE_DIR, "data", "processed")

_ls_path = os.path.join(_proc_dir, "label_stats.npy")
if os.path.exists(_ls_path):
    _ls = np.load(_ls_path)
    LABEL_MEAN = _ls[0].astype(np.float32)
    LABEL_STD  = _ls[1].astype(np.float32)
else:
    LABEL_MEAN = np.array([5169.055664,  3.549788, -0.657069], dtype=np.float32)
    LABEL_STD  = np.array([ 998.064880,  1.081975,  0.723029], dtype=np.float32)

_fs_path = os.path.join(_proc_dir, "feature_stats.npy")
if os.path.exists(_fs_path):
    _fs = np.load(_fs_path)
    FEATURE_MEAN = _fs[0].astype(np.float32)
    FEATURE_STD  = _fs[1].astype(np.float32)
else:
    FEATURE_MEAN = np.zeros(30, dtype=np.float32)
    FEATURE_STD  = np.ones(30,  dtype=np.float32)

# ── 10개 흡수선 정의 (스펙트럼 하이라이트 + XAI 레이블용) ─────────────────────
ABSORPTION_LINES = [
    ("H-alpha",    6543, 6583, "#ff4444"),
    ("H-beta",     4841, 4881, "#ff6644"),
    ("H-gamma",    4320, 4360, "#ff8844"),
    ("H-delta",    4082, 4122, "#ffaa44"),
    ("Ca II K",    3919, 3949, "#44aaff"),
    ("Ca II H",    3953, 3983, "#4488ff"),
    ("Mg I b",     5155, 5195, "#44ff44"),
    ("Fe I 5270",  5255, 5285, "#ffff44"),
    ("Fe I 4383",  4368, 4398, "#ffdd44"),
    ("Na I D",     5877, 5907, "#ff44ff"),
]

# ── XAI 분석 창 (Jacobian proof ratio 계산용) ────────────────────────────────
XAI_ABSORPTION_WINDOWS = {
    "H-alpha":  (6513.0, 6613.0),
    "Mg-b":     (5140.0, 5200.0),
    "Na-D":     (5860.0, 5920.0),
    "H-beta":   (4830.0, 4890.0),
}

# ── FUSION 차원: CNN(1216) + Dense(128) = 1344 ───────────────────────────────
CNN_BRANCH_DIM = 1216
FUSION_DIM     = CNN_BRANCH_DIM + 128  # 1344


class StellarValidatorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("SDSS DR17 Stellar Parameter Evaluation")
        self.root.geometry("1440x920")
        self.root.configure(bg="#151522")

        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.init_model()

        self.max_virtual_rows      = 0
        self.current_idx           = 0
        self.X_flux                = None
        self.Y_labels              = None
        self.valid_paths           = []
        self.current_flux          = None
        self.norm_flux             = None
        self.physical_features_30d = None
        self.show_ablated          = False   # toggle state

        self.setup_ui()
        self.root.after(100, self.lazy_load_real_fits)
        self.root.bind("<Left>",  lambda e: self.move_prev())
        self.root.bind("<Right>", lambda e: self.move_next())
        self.root.focus_set()

    # ── model ─────────────────────────────────────────────────────────────────
    def init_model(self):
        weights_path = os.path.join(BASE_DIR, "weights", "stellar_hybrid_model.pth")
        self.model = StellarParameterHybridNet().to(self.device)
        if os.path.exists(weights_path):
            ckpt = torch.load(weights_path, map_location=self.device)
            if isinstance(ckpt, dict) and 'model_state' in ckpt:
                self.model.load_state_dict(ckpt['model_state'])
            else:
                self.model.load_state_dict(ckpt)
            print(f"[Model] Loaded from {weights_path}")
        else:
            print("[WARN] Weights not found.")
        self.model.eval()

    # ── data loading ──────────────────────────────────────────────────────────
    def lazy_load_real_fits(self):
        all_spec_files = collect_spec_fits_files(TARGET_DATASET_DIR)
        if not all_spec_files:
            self.path_lbl.configure(text=f"[Error] No spec FITS in {TARGET_DATASET_DIR}/")
            return
        try:
            csv_path = os.path.join(TARGET_DATASET_DIR, "Skyserver_SQL6_1_2026 10_51_26 PM.csv")
            flux_matrix, label_matrix, valid_paths = load_spectra_from_fits_list(
                all_spec_files, csv_path, TARGET_DATASET_DIR)
            self.X_flux           = flux_matrix
            self.Y_labels         = label_matrix
            self.valid_paths      = valid_paths
            self.max_virtual_rows = len(self.Y_labels)
            self.path_lbl.configure(text=f"Dataset: {self.max_virtual_rows} SDSS DR17 FITS loaded")
            self.update_profile()
        except Exception as e:
            self.path_lbl.configure(text=f"[Error] {e}")

    # ── XAI weight ratio ──────────────────────────────────────────────────────
    def calculate_weight_attribution_ratio(self):
        try:
            for name, param in self.model.named_parameters():
                if "weight" in name.lower() and param.ndim == 2 \
                        and param.shape[1] == FUSION_DIM:
                    W = param.detach().cpu().numpy()
                    total = np.sum(np.abs(W))
                    if total == 0:
                        return 0.0
                    dense = np.sum(np.abs(W[:, CNN_BRANCH_DIM:]))
                    return (dense / total) * 100
            return -1.0
        except Exception:
            return -1.0

    # ── physics description ───────────────────────────────────────────────────
    def generate_physics_description(self, true_val, pred_val,
                                      err_teff, err_logg, err_feh, weight_ratio):
        def spectral_type(t):
            classes = [
                ("O", 30000.0, 60000.0),
                ("B", 10000.0, 30000.0),
                ("A", 7500.0,  10000.0),
                ("F", 6000.0,  7500.0),
                ("G", 5200.0,  6000.0),
                ("K", 3700.0,  5200.0),
                ("M", 2400.0,  3700.0)
            ]
            if t >= 60000.0:
                return "O0"
            if t < 2400.0:
                return "M9"
            for cls, min_t, max_t in classes:
                if t >= min_t:
                    val = (max_t - t) / (max_t - min_t)
                    subtype = int(val * 10)
                    subtype = max(0, min(9, subtype))
                    return f"{cls}{subtype}"
            return "M9"
        lum = lambda g: "V (dwarf)" if g >= 3.8 else "III (giant)"
        pop = "solar-abundance" if true_val[2] >= -0.3 else "metal-poor (Pop II)"
        fname = os.path.basename(self.valid_paths[self.current_idx]) \
                if self.valid_paths else "Unknown"
        mode  = "ABLATED (30D zeroed)" if self.show_ablated else "NORMAL"
        return (
            f" File: {fname}\n"
            f" XAI Mode: {mode}\n\n"
            f" Spectral Class\n"
            f"   True : {spectral_type(true_val[0])}{lum(true_val[1])}\n"
            f"   Pred : {spectral_type(pred_val[0])}{lum(pred_val[1])}\n\n"
            f" T_eff\n"
            f"   Catalog {true_val[0]:.0f} K  |  Model {pred_val[0]:.0f} K  |  Err {err_teff:.1f}%\n\n"
            f" log g\n"
            f"   Catalog {true_val[1]:.3f}  |  Model {pred_val[1]:.3f}  |  Err {err_logg:.1f}%\n\n"
            f" [Fe/H]\n"
            f"   Catalog {true_val[2]:+.3f} ({pop})\n"
            f"   Model   {pred_val[2]:+.3f}  |  Delta {err_feh:+.4f} dex\n\n"
            f" 30D Branch Weight Share: {weight_ratio:.2f}%\n"
        )

    # ── UI layout ─────────────────────────────────────────────────────────────
    def setup_ui(self):
        left = tk.Frame(self.root, bg="#12121f", width=470, bd=1, relief=tk.SOLID)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=18, pady=18)
        left.pack_propagate(False)

        right = tk.Frame(self.root, bg="#151522")
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=18, pady=18)

        tk.Label(left, text="SDSS DR17 Stellar Evaluation",
                 font=("Helvetica", 13, "bold"),
                 bg="#12121f", fg="#ffffff").pack(pady=12)

        self.path_lbl = tk.Label(left, text="Loading...",
                                 font=("Helvetica", 9, "italic"),
                                 bg="#12121f", fg="#8888aa")
        self.path_lbl.pack(pady=4)

        # navigation
        nav = tk.LabelFrame(left, text="Navigation", font=("Helvetica", 10),
                            bg="#12121f", fg="#ff8888", padx=8, pady=8)
        nav.pack(fill=tk.X, padx=18, pady=4)
        tk.Button(nav, text="<< Prev", command=self.move_prev,
                  bg="#1a1a2e", fg="#ffffff",
                  font=("Helvetica", 9)).pack(side=tk.LEFT, padx=4)
        self.idx_lbl = tk.Label(nav, text="Loading...",
                                font=("Helvetica", 10, "bold"),
                                bg="#12121f", fg="#ffffff")
        self.idx_lbl.pack(side=tk.LEFT, expand=True)
        tk.Button(nav, text="Next >>", command=self.move_next,
                  bg="#1a1a2e", fg="#ffffff",
                  font=("Helvetica", 9)).pack(side=tk.RIGHT, padx=4)

        # weight share
        self.weight_share_lbl = tk.Label(
            left, text="30D Feature Weight Share: --.--%",
            font=("Consolas", 11, "bold"),
            bg="#221133", fg="#ff77ff", bd=1, relief=tk.RIDGE, pady=4)
        self.weight_share_lbl.pack(fill=tk.X, padx=18, pady=4)

        # ablated toggle
        self.toggle_btn = tk.Button(
            left, text="Toggle: NORMAL Jacobian",
            font=("Helvetica", 10, "bold"),
            bg="#112233", fg="#77ddff",
            activebackground="#223344", activeforeground="#ffffff",
            bd=1, relief=tk.RIDGE, pady=6,
            command=self.toggle_ablated)
        self.toggle_btn.pack(fill=tk.X, padx=18, pady=4)

        # parameter table
        tbl = tk.LabelFrame(left, text="Parameter Comparison",
                            font=("Helvetica", 10), bg="#12121f", fg="#88ff88",
                            padx=8, pady=8)
        tbl.pack(fill=tk.X, padx=18, pady=4)

        for col, (txt, fg) in enumerate([
            ("Model Pred","#88ccff"), ("Catalog GT","#ffaaee"), ("Error","#ffffaa")
        ], start=1):
            tk.Label(tbl, text=txt, font=("Helvetica", 9, "bold"),
                     bg="#12121f", fg=fg).grid(row=0, column=col, padx=6, pady=4)

        self.pred_labels  = []
        self.true_labels  = []
        self.error_labels = []
        for i, name in enumerate(["T_eff (K)", "Log g (dex)", "[Fe/H] (dex)"]):
            tk.Label(tbl, text=name, font=("Helvetica", 10, "bold"),
                     bg="#12121f", fg="#ffffff").grid(row=i+1, column=0, sticky="w", padx=2, pady=8)
            for lst, fg in [(self.pred_labels,"#88ffcc"),
                            (self.true_labels,"#ff88aa"),
                            (self.error_labels,"#ffff77")]:
                lbl = tk.Label(tbl, text="-",
                               font=("Consolas", 11, "bold"),
                               bg="#12121f", fg=fg)
                lbl.grid(row=i+1, column=len(lst)+1 if lst is self.pred_labels
                         else (2 if lst is self.true_labels else 3), padx=6)
                lst.append(lbl)

        # physics description
        desc = tk.LabelFrame(left, text="Astrophysical Telemetry",
                             font=("Helvetica", 10), bg="#12121f", fg="#ffff77",
                             padx=8, pady=4)
        desc.pack(fill=tk.BOTH, expand=True, padx=18, pady=4)
        self.desc_text = tk.Text(desc, bg="#181826", fg="#ddddff",
                                 font=("Helvetica", 9), wrap=tk.WORD,
                                 bd=0, highlightthickness=1,
                                 highlightbackground="#252538")
        self.desc_text.pack(fill=tk.BOTH, expand=True)

        # matplotlib
        self.fig, (self.ax_spec, self.ax_xai) = plt.subplots(
            2, 1, figsize=(8.5, 7.5), facecolor="#151522", sharex=True)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ── navigation ────────────────────────────────────────────────────────────
    def move_prev(self):
        if self.X_flux is not None and self.current_idx > 0:
            self.current_idx -= 1
            self.update_profile()
        self.root.focus_set()

    def move_next(self):
        if self.X_flux is not None and self.current_idx < self.max_virtual_rows - 1:
            self.current_idx += 1
            self.update_profile()
        self.root.focus_set()

    def toggle_ablated(self):
        self.show_ablated = not self.show_ablated
        mode = "ABLATED (30D zeroed)" if self.show_ablated else "NORMAL"
        self.toggle_btn.configure(
            text=f"Toggle: {mode} Jacobian",
            bg="#331122" if self.show_ablated else "#112233",
            fg="#ff7777" if self.show_ablated else "#77ddff")
        if self.norm_flux is not None:
            self.execute_automated_jacobian()

    # ── inference + plot ──────────────────────────────────────────────────────
    def update_profile(self):
        if self.X_flux is None or len(self.X_flux) == 0:
            return
        self.idx_lbl.configure(
            text=f"Star #{self.current_idx + 1} / {self.max_virtual_rows}")

        raw_flux  = self.X_flux[self.current_idx]
        real_true = self.Y_labels[self.current_idx]

        f_mean = np.mean(raw_flux); f_std = np.std(raw_flux) + 1e-8
        self.current_flux = raw_flux.reshape(1, -1)
        self.norm_flux    = np.clip((self.current_flux - f_mean) / f_std, -3.0, 3.0)

        raw_feat  = extract_18d_features_live_eval(WAVE_GRID, raw_flux)
        norm_feat = (raw_feat - FEATURE_MEAN) / (FEATURE_STD + 1e-8)
        self.physical_features_30d = norm_feat.astype(np.float32)

        t_flux = torch.from_numpy(self.norm_flux).float().unsqueeze(1).to(self.device)
        t_feat = torch.from_numpy(self.physical_features_30d).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            norm_pred = self.model(t_flux, t_feat).cpu().numpy()[0]
        real_pred = norm_pred * LABEL_STD + LABEL_MEAN

        err_teff = (abs(real_true[0] - real_pred[0]) / (real_true[0] + 1e-8)) * 100
        err_logg = (abs(real_true[1] - real_pred[1]) / (abs(real_true[1]) + 1e-8)) * 100
        err_feh  = real_pred[2] - real_true[2]

        for i, (p, t, e) in enumerate(zip(self.pred_labels,
                                           self.true_labels,
                                           self.error_labels)):
            fmts = [f"{real_pred[i]:.1f} K", f"{real_pred[1]:.3f}", f"{real_pred[2]:.3f}"]
            trues = [f"{real_true[i]:.1f} K", f"{real_true[1]:.3f}", f"{real_true[2]:.3f}"]
            errs  = [f"{err_teff:.2f}%", f"{err_logg:.2f}%", f"{abs(err_feh):.4f} dex"]
            p.configure(text=[f"{real_pred[0]:.1f} K", f"{real_pred[1]:.3f}", f"{real_pred[2]:.3f}"][i])
            t.configure(text=[f"{real_true[0]:.1f} K", f"{real_true[1]:.3f}", f"{real_true[2]:.3f}"][i])
            e.configure(text=[f"{err_teff:.2f}%", f"{err_logg:.2f}%", f"{abs(err_feh):.4f} dex"][i])

        wr = self.calculate_weight_attribution_ratio()
        self.weight_share_lbl.configure(text=f"30D Feature Weight Share: {wr:.2f}%")

        self.desc_text.config(state=tk.NORMAL)
        self.desc_text.delete("1.0", tk.END)
        self.desc_text.insert(tk.END,
            self.generate_physics_description(real_true, real_pred,
                                              err_teff, err_logg, err_feh, wr))
        self.desc_text.config(state=tk.DISABLED)

        # spectrum plot
        self.ax_spec.clear()
        self.ax_spec.set_facecolor("#12121f")
        self.ax_spec.plot(WAVE_GRID, self.current_flux[0],
                          color="#8da9f4", alpha=0.85, linewidth=0.6,
                          label=f"SDSS #{self.current_idx + 1}", zorder=3)
        for name, lo, hi, color in ABSORPTION_LINES:
            self.ax_spec.axvspan(lo, hi, color=color, alpha=0.15, zorder=2)
            mid = (lo + hi) / 2
            self.ax_spec.text(mid, 3.85, name, color=color, fontsize=5,
                              ha='center', va='bottom', rotation=90, alpha=0.9)
        self.ax_spec.set_title(f"Stellar Spectrum — SDSS DR17 #{self.current_idx + 1}",
                               color="#ffffff", fontsize=9, fontweight="bold")
        self.ax_spec.set_ylabel("Norm. Flux", color="#ffffff", fontsize=7)
        self.ax_spec.tick_params(colors="#ffffff", labelsize=6)
        self.ax_spec.set_xlim(3650.0, 10250.0)
        self.ax_spec.set_ylim(-0.1, 4.4)
        self.ax_spec.grid(True, color="#252538", linestyle="--", alpha=0.4)
        self.ax_spec.legend(facecolor="#151522", edgecolor="#333344",
                            labelcolor="#ffffff", loc="upper right", fontsize=6)

        self.execute_automated_jacobian()

    def execute_automated_jacobian(self):
        if self.norm_flux is None:
            return

        t_flux = torch.from_numpy(self.norm_flux).float().unsqueeze(1).to(self.device)
        t_flux.requires_grad_(True)

        if self.show_ablated:
            t_feat = torch.zeros(1, len(self.physical_features_30d),
                                 device=self.device)
            label  = r"Jacobian (30D ABLATED) $\partial \log g / \partial \lambda$"
            color  = "#ff9944"
        else:
            t_feat = torch.from_numpy(
                self.physical_features_30d).float().unsqueeze(0).to(self.device)
            label  = r"Jacobian (Normal) $\partial \log g / \partial \lambda$"
            color  = "#ff77ff"

        pred = self.model(t_flux, t_feat)
        self.model.zero_grad()
        grad_out = torch.zeros_like(pred)
        grad_out[0, 1] = 1.0
        pred.backward(grad_out)

        jac      = np.abs(t_flux.grad.cpu().numpy()[0, 0])
        jac_sm   = np.convolve(jac, np.ones(15) / 15, mode='same')

        self.ax_xai.clear()
        self.ax_xai.set_facecolor("#12121f")
        self.ax_xai.plot(WAVE_GRID, jac_sm, color=color,
                         linewidth=0.7, alpha=0.9, label=label)

        for name, lo, hi, col in ABSORPTION_LINES:
            self.ax_xai.axvspan(lo, hi, color=col, alpha=0.12)

        mode_str = "ABLATED" if self.show_ablated else "Normal"
        self.ax_xai.set_title(f"XAI Jacobian Sensitivity — {mode_str}",
                              color="#ffffff", fontsize=9, fontweight="bold")
        self.ax_xai.set_ylabel("XAI Sensitivity", color="#ffffff", fontsize=7)
        self.ax_xai.set_xlabel("Wavelength (Angstrom)", color="#ffffff", fontsize=7)
        self.ax_xai.set_xlim(3650.0, 10250.0)
        self.ax_xai.tick_params(colors="#ffffff", labelsize=6)
        self.ax_xai.grid(True, color="#252538", linestyle="--", alpha=0.4)
        self.ax_xai.legend(facecolor="#151522", edgecolor="#333344",
                           labelcolor="#ffffff", loc="upper right", fontsize=6)

        self.fig.tight_layout()
        self.fig.subplots_adjust(hspace=0.12)
        self.canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app  = StellarValidatorGUI(root)
    root.mainloop()
