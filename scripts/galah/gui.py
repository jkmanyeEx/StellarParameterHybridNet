"""
GALAH DR4 Stellar Parameter Evaluation GUI.

Loads real GALAH DR4 spectra from data/galah/processed/ and displays
per-star predictions, Jacobian XAI sensitivity, and weight attribution.
"""

import os
import sys
import numpy as np
import torch
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif'],
    'font.size': 7.0, 'axes.labelsize': 7.0, 'axes.titlesize': 9.0,
    'xtick.labelsize': 6.0, 'ytick.labelsize': 6.0,
    'legend.fontsize': 6.0, 'figure.titlesize': 9.0
})

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.models.galah.hybrid_net import StellarParameterHybridNet
from src.data.galah.extract_features import extract_45d_features_single_star
from src.validation.galah.xai_analyzer import calculate_per_line_weight_attribution

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

# ── Paths ─────────────────────────────────────────────────────────────────────
_proc_dir    = os.path.join(BASE_DIR, "data", "galah", "processed")
_flux_path   = os.path.join(_proc_dir, "X_flux_clean.npy")
_label_path  = os.path.join(_proc_dir, "Y_labels.npy")
_wave_path   = os.path.join(_proc_dir, "standard_wave.npy")
_ls_path     = os.path.join(_proc_dir, "label_stats.npy")
_fs_path     = os.path.join(_proc_dir, "feature_stats.npy")

for p in (_flux_path, _label_path, _wave_path, _ls_path, _fs_path):
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"Required file not found: {p}\n"
            "Execute the GALAH preprocessing and training pipeline first."
        )

WAVE_GRID    = np.load(_wave_path)                       # (4, 4000)
_ls          = np.load(_ls_path)
LABEL_MEAN   = _ls[0].astype(np.float32)
LABEL_STD    = _ls[1].astype(np.float32)
_fs          = np.load(_fs_path)
FEATURE_MEAN = _fs[0].astype(np.float32)
FEATURE_STD  = _fs[1].astype(np.float32)

# ── Architecture dims ─────────────────────────────────────────────────────────
CNN_BRANCH_DIM   = 6400   # 4 arms × 1600
DENSE_BRANCH_DIM = 128
FUSION_DIM       = CNN_BRANCH_DIM + DENSE_BRANCH_DIM  # 6528

# ── GALAH 4-arm wavelength ranges for axis labels ─────────────────────────────
ARM_RANGES = [
    (4713, 4903, "CCD1 Blue"),
    (5648, 5873, "CCD2 Green"),
    (6478, 6737, "CCD3 Red"),
    (7585, 7887, "CCD4 NIR"),
]

# Absorption lines per arm for spectrum subplots (name, arm_idx, center_Å, color)
ABSORPTION_LINES = [
    ("H-beta",   0, 4861, "#ff6644"),
    ("Fe 4882",  0, 4882, "#ffaa44"),
    ("Mg 4703",  0, 4703, "#44ffaa"),
    ("Ba 4897",  0, 4897, "#aaffaa"),
    ("Mg 5711",  1, 5711, "#44ff44"),
    ("Fe 5662",  1, 5662, "#aaff44"),
    ("Fe 5782",  1, 5782, "#ffff44"),
    ("Fe 5862",  1, 5862, "#ffdd44"),
    ("H-alpha",  2, 6563, "#ff4444"),
    ("Li 6708",  2, 6708, "#44aaff"),
    ("Fe 6495",  2, 6495, "#ff8888"),
    ("Ca 6499",  2, 6499, "#ffaaaa"),
    ("K  7699",  3, 7699, "#ff44ff"),
    ("Fe 7748",  3, 7748, "#ff88ff"),
    ("O  7772",  3, 7772, "#cc88ff"),
]

# Same lines mapped to (arm_idx, pixel_index) for XAI concat plot
# pixel_index = arm_offset(4000*arm_idx) + index of nearest wavelength in arm grid
def _build_xai_line_markers():
    """
    Convert each absorption line's wavelength to a pixel index in the
    concatenated (4 × 4000 = 16000) Jacobian x-axis.
    Returns list of (name, pixel_idx, color).
    """
    markers = []
    arm_wave_grids = None   # loaded lazily after WAVE_GRID is available
    return markers  # filled after module-level WAVE_GRID is loaded

XAI_LINE_MARKERS = []   # populated in StellarValidatorGUI.__init__


class StellarValidatorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("GALAH DR4 Stellar Parameter Evaluation")
        self.root.geometry("1500x940")
        self.root.configure(bg="#151522")

        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.init_model()

        # Build validation split (same seed/logic as engine.py)
        self._load_val_split()

        # Build XAI pixel-index markers from ABSORPTION_LINES + WAVE_GRID
        global XAI_LINE_MARKERS
        XAI_LINE_MARKERS = []
        for name, arm_idx, center_wave, color in ABSORPTION_LINES:
            arm_wave = WAVE_GRID[arm_idx]           # (4000,)
            px_local = int(np.argmin(np.abs(arm_wave - center_wave)))
            px_global = arm_idx * 4000 + px_local  # position in concat axis
            XAI_LINE_MARKERS.append((name, px_global, color))

        self.current_idx          = 0
        self.norm_flux_4arm       = None
        self.physical_features    = None
        self.show_ablated         = False

        default_dir = os.path.join(BASE_DIR, "weights", "galah")
        fallback_file = os.path.join(BASE_DIR, "weights", "stellar_hybrid_model.pth")
        self.selected_weights_path = self.select_weights_dialog(default_dir, fallback_file)
        self.init_model()

        self.setup_ui()
        self.root.after(100, self.update_profile)
        self.root.bind("<Left>",  lambda e: self.move_prev())
        self.root.bind("<Right>", lambda e: self.move_next())
        self.root.focus_set()

    # ── weight selection dialog ───────────────────────────────────────────────
    def select_weights_dialog(self, default_dir, fallback_file=None):
        import glob
        from tkinter import filedialog, messagebox

        # Scan directories
        pth_files = []
        if os.path.exists(default_dir):
            pth_files = glob.glob(os.path.join(default_dir, "*.pth"))
        
        root_weights_dir = os.path.abspath(os.path.join(default_dir, ".."))
        if not pth_files and os.path.exists(root_weights_dir):
            pth_files = glob.glob(os.path.join(root_weights_dir, "*.pth"))
            
        if fallback_file and os.path.exists(fallback_file) and fallback_file not in pth_files:
            pth_files.append(fallback_file)

        pth_files = sorted(list(set(pth_files)))

        # Create Toplevel Window
        dialog = tk.Toplevel(self.root)
        dialog.title("Select Model Weights")
        dialog.geometry("520x350")
        dialog.configure(bg="#151522")
        dialog.resizable(False, False)
        
        dialog.transient(self.root)
        dialog.grab_set()
        
        dialog.update_idletasks()
        width = dialog.winfo_width()
        height = dialog.winfo_height()
        x = self.root.winfo_x() + (self.root.winfo_width() // 2) - (width // 2)
        y = self.root.winfo_y() + (self.root.winfo_height() // 2) - (height // 2)
        dialog.geometry(f"+{x}+{y}")

        selected_path = tk.StringVar()
        if pth_files:
            if fallback_file and fallback_file in pth_files:
                selected_path.set(fallback_file)
            else:
                selected_path.set(pth_files[0])
                
        cancelled = [True]

        tk.Label(
            dialog, 
            text="Stellar Parameter HybridNet Checkpoints",
            font=("Helvetica", 12, "bold"),
            bg="#151522", fg="#ffffff",
            pady=10
        ).pack()
        
        tk.Label(
            dialog,
            text="Choose a model weight file (.pth) to load on initialization:",
            font=("Helvetica", 9),
            bg="#151522", fg="#aaaaff"
        ).pack(pady=(0, 10))

        frame = tk.Frame(dialog, bg="#12121f", bd=1, relief=tk.SOLID)
        frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=5)
        
        canvas = tk.Canvas(frame, bg="#12121f", highlightthickness=0)
        scrollbar = tk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        scroll_frame = tk.Frame(canvas, bg="#12121f")

        scroll_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        scrollbar.pack(side="right", fill="y")

        if pth_files:
            for filepath in pth_files:
                filename = os.path.basename(filepath)
                parent_dir = os.path.basename(os.path.dirname(filepath))
                display_name = f"{filename}  ({parent_dir}/)" if parent_dir else filename
                
                rb = tk.Radiobutton(
                    scroll_frame,
                    text=display_name,
                    variable=selected_path,
                    value=filepath,
                    font=("Consolas", 9),
                    bg="#12121f", fg="#ffffff",
                    selectcolor="#151522",
                    activebackground="#1a1a2e", activeforeground="#ffffff",
                    anchor="w", justify="left"
                )
                rb.pack(fill="x", anchor="w", padx=10, pady=4)
        else:
            tk.Label(
                scroll_frame,
                text="No weights files (*.pth) found.\nClick 'Browse Custom...' to locate one manually.",
                font=("Helvetica", 9, "italic"),
                bg="#12121f", fg="#ff6666",
                padx=10, pady=20
            ).pack(fill="both", expand=True)

        def browse_custom():
            file_selected = filedialog.askopenfilename(
                parent=dialog,
                title="Locate Model Weights Checkpoint",
                filetypes=[("PyTorch weights", "*.pth")]
            )
            if file_selected:
                selected_path.set(file_selected)
                confirm()

        def confirm():
            path = selected_path.get()
            if not path or not os.path.exists(path):
                messagebox.showerror("Error", "Selected file does not exist.", parent=dialog)
                return
            cancelled[0] = False
            dialog.destroy()

        def cancel():
            dialog.destroy()

        btn_bar = tk.Frame(dialog, bg="#151522")
        btn_bar.pack(fill=tk.X, pady=15, padx=20)

        tk.Button(
            btn_bar, text="Browse Custom...", command=browse_custom,
            bg="#1a1a2e", fg="#77ddff", activebackground="#223344", activeforeground="#ffffff",
            font=("Helvetica", 9), relief=tk.GROOVE, bd=1, padx=8, pady=4
        ).pack(side=tk.LEFT)

        tk.Button(
            btn_bar, text="Cancel & Exit", command=cancel,
            bg="#1a1a2e", fg="#ff7777", activebackground="#331122", activeforeground="#ffffff",
            font=("Helvetica", 9), relief=tk.GROOVE, bd=1, padx=8, pady=4
        ).pack(side=tk.RIGHT, padx=5)

        tk.Button(
            btn_bar, text="Load Weights", command=confirm,
            bg="#1a1a2e", fg="#88ffcc", activebackground="#224433", activeforeground="#ffffff",
            font=("Helvetica", 9, "bold"), relief=tk.GROOVE, bd=1, padx=12, pady=4
        ).pack(side=tk.RIGHT, padx=5)

        self.root.wait_window(dialog)
        
        if cancelled[0]:
            print("[GUI] Initialization cancelled by user.")
            sys.exit(0)
            
        return selected_path.get()

    # ── Model ─────────────────────────────────────────────────────────────────
    def init_model(self):
        self.model = StellarParameterHybridNet(use_features=True).to(self.device)
        ckpt = torch.load(self.selected_weights_path, map_location=self.device)
        if isinstance(ckpt, dict) and 'model_state' in ckpt:
            self.model.load_state_dict(ckpt['model_state'])
        else:
            self.model.load_state_dict(ckpt)
        print(f"[GUI] Model loaded from: {self.selected_weights_path}")
        self.model.eval()
        self.per_line_attr = calculate_per_line_weight_attribution(self.model)

    # ── Val split ─────────────────────────────────────────────────────────────
    def _load_val_split(self):
        X_flux_all   = np.load(_flux_path,  mmap_mode='r')
        Y_labels_all = np.load(_label_path, mmap_mode='r')
        n = min(len(X_flux_all), len(Y_labels_all))
        raw_labels = Y_labels_all[:n]

        # Use sealed test indices if available, otherwise fall back to val split
        test_indices_path = os.path.join(_proc_dir, "test_indices.npy")
        if os.path.exists(test_indices_path):
            val_indices = np.load(test_indices_path)
            print(f"[GUI] Sealed test set loaded: {len(val_indices)} stars")
        else:
            valid_mask    = (raw_labels[:, 0] > -900) & \
                            (raw_labels[:, 1] > -900) & \
                            (raw_labels[:, 2] > -900)
            valid_indices = np.where(valid_mask)[0]
            rng = np.random.default_rng(42)
            rng.shuffle(valid_indices)
            train_size  = int(0.8 * len(valid_indices))
            val_indices = valid_indices[train_size:]
            print(f"[GUI] Legacy val split loaded: {len(val_indices)} stars")

        self.X_flux   = np.array(X_flux_all[val_indices])    # (N, 4, 4000)
        self.Y_labels = np.array(Y_labels_all[val_indices])  # (N, 3)
        self.n_val    = len(val_indices)

    # ── Weight ratio ──────────────────────────────────────────────────────────
    def calculate_weight_attribution_ratio(self):
        try:
            for name, param in self.model.named_parameters():
                if "weight" in name.lower() and param.ndim == 2 \
                        and param.shape[1] == FUSION_DIM:
                    W     = param.detach().cpu().numpy()
                    total = np.sum(np.abs(W))
                    if total == 0:
                        return 0.0, 0.0
                    dense = np.sum(np.abs(W[:, CNN_BRANCH_DIM:]))
                    learned  = dense / total * 100
                    nominal  = DENSE_BRANCH_DIM / FUSION_DIM * 100
                    return learned, nominal
            return -1.0, -1.0
        except Exception:
            return -1.0, -1.0

    # ── Physics description ───────────────────────────────────────────────────
    def generate_physics_description(self, true_val, pred_val,
                                     err_teff, err_logg, err_feh,
                                     learned_wr, nominal_wr):
        def spectral_type(t):
            for cls, lo, hi in [("O",30000,60000),("B",10000,30000),
                                  ("A",7500,10000),("F",6000,7500),
                                  ("G",5200,6000),("K",3700,5200),("M",2400,3700)]:
                if t >= lo:
                    sub = min(9, int((hi - t) / (hi - lo) * 10))
                    return f"{cls}{sub}"
            return "M9"
        lum = lambda g: "V (dwarf)" if g >= 3.8 else "III (giant)"
        pop = "solar-abundance" if true_val[2] >= -0.3 else "metal-poor (Pop II)"
        mode = "ABLATED (45D zeroed)" if self.show_ablated else "NORMAL"
        ratio_str = f"{learned_wr:.2f}% / {nominal_wr:.2f}% (x{learned_wr/max(nominal_wr,1e-8):.2f})"
        return (
            f" XAI Mode: {mode}\n\n"
            f" Spectral Class\n"
            f"   True : {spectral_type(true_val[0])} {lum(true_val[1])}\n"
            f"   Pred : {spectral_type(pred_val[0])} {lum(pred_val[1])}\n\n"
            f" T_eff\n"
            f"   Catalog {true_val[0]:.0f} K  |  Model {pred_val[0]:.0f} K"
            f"  |  Err {err_teff:.1f}%\n\n"
            f" log g\n"
            f"   Catalog {true_val[1]:.3f}  |  Model {pred_val[1]:.3f}"
            f"  |  Err {err_logg:.1f}%\n\n"
            f" [Fe/H]\n"
            f"   Catalog {true_val[2]:+.3f} ({pop})\n"
            f"   Model   {pred_val[2]:+.3f}  |  Delta {err_feh:+.4f} dex\n\n"
            f" 45D Branch Weight:\n"
            f"   Learned / Nominal = {ratio_str}\n\n"
            f" Per-Line Weight (Top 5):\n"
            + "".join(f"   {n:<12s} {p:5.1f}%\n" for n, p in self.per_line_attr[:5])
        )

    # ── UI ────────────────────────────────────────────────────────────────────
    def setup_ui(self):
        left = tk.Frame(self.root, bg="#12121f", width=480, bd=1, relief=tk.SOLID)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=18, pady=18)
        left.pack_propagate(False)

        right = tk.Frame(self.root, bg="#151522")
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=18, pady=18)

        tk.Label(left, text="GALAH DR4 Stellar Evaluation",
                 font=("Helvetica", 13, "bold"),
                 bg="#12121f", fg="#ffffff").pack(pady=12)

        self.path_lbl = tk.Label(
            left, text=f"Validation split: {self.n_val} stars",
            font=("Helvetica", 9, "italic"), bg="#12121f", fg="#8888aa")
        self.path_lbl.pack(pady=4)

        # Navigation
        nav = tk.LabelFrame(left, text="Navigation", font=("Helvetica", 10),
                            bg="#12121f", fg="#ff8888", padx=8, pady=8)
        nav.pack(fill=tk.X, padx=18, pady=4)
        tk.Button(nav, text="<< Prev", command=self.move_prev,
                  bg="#1a1a2e", fg="#ffffff", font=("Helvetica", 9)).pack(side=tk.LEFT, padx=4)
        self.idx_lbl = tk.Label(nav, text="--", font=("Helvetica", 10, "bold"),
                                bg="#12121f", fg="#ffffff")
        self.idx_lbl.pack(side=tk.LEFT, expand=True)
        tk.Button(nav, text="Next >>", command=self.move_next,
                  bg="#1a1a2e", fg="#ffffff", font=("Helvetica", 9)).pack(side=tk.RIGHT, padx=4)

        # Weight share
        self.weight_share_lbl = tk.Label(
            left, text="45D Feature Weight: --",
            font=("Consolas", 11, "bold"),
            bg="#221133", fg="#ff77ff", bd=1, relief=tk.RIDGE, pady=4)
        self.weight_share_lbl.pack(fill=tk.X, padx=18, pady=4)

        # Per-line attribution
        line_frame = tk.LabelFrame(left, text="Per-Line Weight Attribution (45D)",
                                   font=("Helvetica", 9), bg="#12121f",
                                   fg="#aaaaff", padx=6, pady=4)
        line_frame.pack(fill=tk.X, padx=18, pady=4)
        self.line_attr_labels = []
        for lname, pct in self.per_line_attr[:10]:
            lbl = tk.Label(line_frame, text=f"{lname:<12s} {pct:5.1f}%",
                           font=("Consolas", 8), bg="#12121f", fg="#ccccff", anchor="w")
            lbl.pack(fill=tk.X)
            self.line_attr_labels.append(lbl)

        # Ablation toggle
        self.toggle_btn = tk.Button(
            left, text="Toggle: NORMAL Jacobian",
            font=("Helvetica", 10, "bold"),
            bg="#112233", fg="#77ddff",
            activebackground="#223344", activeforeground="#ffffff",
            bd=1, relief=tk.RIDGE, pady=6,
            command=self.toggle_ablated)
        self.toggle_btn.pack(fill=tk.X, padx=18, pady=4)

        # Parameter table
        tbl = tk.LabelFrame(left, text="Parameter Comparison",
                            font=("Helvetica", 10), bg="#12121f", fg="#88ff88",
                            padx=8, pady=8)
        tbl.pack(fill=tk.X, padx=18, pady=4)
        for col, (txt, fg) in enumerate([
            ("Model Pred", "#88ccff"), ("Catalog GT", "#ffaaee"), ("Error", "#ffffaa")
        ], start=1):
            tk.Label(tbl, text=txt, font=("Helvetica", 9, "bold"),
                     bg="#12121f", fg=fg).grid(row=0, column=col, padx=6, pady=4)
        self.pred_labels, self.true_labels, self.error_labels = [], [], []
        for i, name in enumerate(["T_eff (K)", "log g (dex)", "[Fe/H] (dex)"]):
            tk.Label(tbl, text=name, font=("Helvetica", 10, "bold"),
                     bg="#12121f", fg="#ffffff").grid(row=i+1, column=0, sticky="w", padx=2, pady=8)
            for j, (lst, fg) in enumerate([(self.pred_labels, "#88ffcc"),
                                            (self.true_labels, "#ff88aa"),
                                            (self.error_labels, "#ffff77")]):
                lbl = tk.Label(tbl, text="-", font=("Consolas", 11, "bold"),
                               bg="#12121f", fg=fg)
                lbl.grid(row=i+1, column=j+1, padx=6)
                lst.append(lbl)

        # Physics description
        desc = tk.LabelFrame(left, text="Astrophysical Telemetry",
                             font=("Helvetica", 10), bg="#12121f", fg="#ffff77",
                             padx=8, pady=4)
        desc.pack(fill=tk.BOTH, expand=True, padx=18, pady=4)
        self.desc_text = tk.Text(desc, bg="#181826", fg="#ddddff",
                                 font=("Helvetica", 9), wrap=tk.WORD,
                                 bd=0, highlightthickness=1,
                                 highlightbackground="#252538")
        self.desc_text.pack(fill=tk.BOTH, expand=True)

        # Matplotlib — 5 subplots: 4 arms + 1 XAI
        self.fig, axes = plt.subplots(5, 1, figsize=(8.5, 9.0),
                                      facecolor="#151522")
        self.ax_arms = axes[:4]
        self.ax_xai  = axes[4]
        self.canvas  = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ── Navigation ────────────────────────────────────────────────────────────
    def move_prev(self):
        if self.current_idx > 0:
            self.current_idx -= 1
            self.update_profile()
        self.root.focus_set()

    def move_next(self):
        if self.current_idx < self.n_val - 1:
            self.current_idx += 1
            self.update_profile()
        self.root.focus_set()

    def toggle_ablated(self):
        self.show_ablated = not self.show_ablated
        mode = "ABLATED (45D zeroed)" if self.show_ablated else "NORMAL"
        self.toggle_btn.configure(
            text=f"Toggle: {mode} Jacobian",
            bg="#331122" if self.show_ablated else "#112233",
            fg="#ff7777" if self.show_ablated else "#77ddff")
        if self.norm_flux_4arm is not None:
            self.execute_automated_jacobian()

    # ── Inference + plot ──────────────────────────────────────────────────────
    def update_profile(self):
        self.idx_lbl.configure(
            text=f"Star #{self.current_idx + 1} / {self.n_val}")

        raw_flux_4arm = self.X_flux[self.current_idx]    # (4, 4000)
        real_true     = self.Y_labels[self.current_idx]

        # Per-arm z-score normalisation (identical to dataset.py)
        f_mean = np.mean(raw_flux_4arm, axis=1, keepdims=True)
        f_std  = np.std(raw_flux_4arm,  axis=1, keepdims=True) + 1e-8
        self.norm_flux_4arm = np.clip((raw_flux_4arm - f_mean) / f_std, -3.0, 3.0)

        # 45D features
        raw_feat  = extract_45d_features_single_star(WAVE_GRID, raw_flux_4arm)
        norm_feat = (raw_feat - FEATURE_MEAN) / (FEATURE_STD + 1e-8)
        self.physical_features = norm_feat.astype(np.float32)

        t_flux = torch.from_numpy(self.norm_flux_4arm).float().unsqueeze(0).to(self.device)
        t_feat = torch.from_numpy(self.physical_features).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            norm_pred = self.model(t_flux, t_feat).cpu().numpy()[0]
        real_pred = norm_pred * LABEL_STD + LABEL_MEAN

        err_teff = abs(real_true[0] - real_pred[0]) / (real_true[0] + 1e-8) * 100
        err_logg = abs(real_true[1] - real_pred[1]) / (abs(real_true[1]) + 1e-8) * 100
        err_feh  = real_pred[2] - real_true[2]

        fmt = [f"{real_pred[0]:.1f} K", f"{real_pred[1]:.3f}", f"{real_pred[2]:.3f}"]
        tru = [f"{real_true[0]:.1f} K", f"{real_true[1]:.3f}", f"{real_true[2]:.3f}"]
        err = [f"{err_teff:.2f}%", f"{err_logg:.2f}%", f"{abs(err_feh):.4f} dex"]
        for i in range(3):
            self.pred_labels[i].configure(text=fmt[i])
            self.true_labels[i].configure(text=tru[i])
            self.error_labels[i].configure(text=err[i])

        learned_wr, nominal_wr = self.calculate_weight_attribution_ratio()
        self.weight_share_lbl.configure(
            text=f"45D: {learned_wr:.2f}% (nominal {nominal_wr:.2f}%)")

        self.desc_text.config(state=tk.NORMAL)
        self.desc_text.delete("1.0", tk.END)
        self.desc_text.insert(tk.END,
            self.generate_physics_description(
                real_true, real_pred, err_teff, err_logg, err_feh,
                learned_wr, nominal_wr))
        self.desc_text.config(state=tk.DISABLED)

        # Plot 4 arms
        arm_colors = ["#7ab4f5", "#7de8a8", "#f58a7a", "#c87af5"]
        for arm_idx, (ax, color) in enumerate(zip(self.ax_arms, arm_colors)):
            ax.clear()
            ax.set_facecolor("#12121f")
            wave = WAVE_GRID[arm_idx]
            ax.plot(wave, raw_flux_4arm[arm_idx], color=color,
                    linewidth=0.6, alpha=0.85)
            wmin, wmax, label = ARM_RANGES[arm_idx]
            ax.set_title(label, color="#ffffff", fontsize=7, pad=2)
            ax.set_xlim(wmin, wmax)
            ax.tick_params(colors="#ffffff", labelsize=5)
            ax.grid(True, color="#252538", linestyle="--", alpha=0.3)
            for aline_name, a_arm, a_center, a_color in ABSORPTION_LINES:
                if a_arm == arm_idx:
                    ax.axvline(a_center, color=a_color, alpha=0.5,
                               linewidth=0.8, linestyle="--")
                    ax.text(a_center, ax.get_ylim()[1] * 0.9,
                            aline_name, color=a_color, fontsize=4,
                            ha='center', rotation=90, alpha=0.8)

        self.execute_automated_jacobian()

    def execute_automated_jacobian(self):
        if self.norm_flux_4arm is None:
            return

        t_flux = torch.from_numpy(self.norm_flux_4arm).float().unsqueeze(0).to(self.device)
        t_flux.requires_grad_(True)

        if self.show_ablated:
            t_feat = torch.zeros(1, len(self.physical_features), device=self.device)
            label  = r"Jacobian (45D ABLATED) $\partial \log g / \partial \lambda$"
            color  = "#ff9944"
        else:
            t_feat = torch.from_numpy(
                self.physical_features).float().unsqueeze(0).to(self.device)
            label  = r"Jacobian (Normal) $\partial \log g / \partial \lambda$"
            color  = "#ff77ff"

        # Clear accumulated gradient before backward
        if t_flux.grad is not None:
            t_flux.grad.zero_()

        pred = self.model(t_flux, t_feat)
        grad_out = torch.zeros_like(pred)
        grad_out[0, 1] = 1.0
        pred.backward(grad_out)

        # Flatten (4, 4000) gradient → concat for display
        jac     = np.abs(t_flux.grad.cpu().numpy()[0])  # (4, 4000)
        jac_cat = np.concatenate([jac[i] for i in range(4)])
        jac_sm  = np.convolve(jac_cat, np.ones(15) / 15, mode='same')
        x_axis  = np.arange(len(jac_cat))

        self.ax_xai.clear()
        self.ax_xai.set_facecolor("#12121f")
        self.ax_xai.plot(x_axis, jac_sm, color=color,
                         linewidth=0.7, alpha=0.9, label=label)

        # Arm boundary markers
        for i in range(1, 4):
            self.ax_xai.axvline(i * 4000, color="#555566",
                                linewidth=0.8, linestyle=":")
        arm_labels = ["CCD1\nBlue", "CCD2\nGreen", "CCD3\nRed", "CCD4\nNIR"]
        y_top = np.max(jac_sm) if np.max(jac_sm) > 0 else 0.035
        for i, al in enumerate(arm_labels):
            self.ax_xai.text((i + 0.5) * 4000, y_top * 0.92,
                             al, color="#888899", fontsize=5, ha='center')

        # Absorption line markers
        for line_name, px_global, line_color in XAI_LINE_MARKERS:
            self.ax_xai.axvline(px_global, color=line_color,
                                linewidth=0.7, linestyle="--", alpha=0.55)
            self.ax_xai.text(px_global, y_top * 0.72, line_name,
                             color=line_color, fontsize=3.8,
                             ha='center', va='bottom',
                             rotation=90, alpha=0.85)

        mode_str = "ABLATED" if self.show_ablated else "Normal"
        self.ax_xai.set_title(f"XAI Jacobian Sensitivity — {mode_str}",
                              color="#ffffff", fontsize=9, fontweight="bold")
        self.ax_xai.set_ylabel("XAI Sensitivity", color="#ffffff", fontsize=7)
        self.ax_xai.set_xlabel("Pixel index (4 arms concatenated)",
                               color="#ffffff", fontsize=7)
        self.ax_xai.tick_params(colors="#ffffff", labelsize=5)
        self.ax_xai.grid(True, color="#252538", linestyle="--", alpha=0.4)
        self.ax_xai.legend(facecolor="#151522", edgecolor="#333344",
                           labelcolor="#ffffff", loc="upper right", fontsize=6)

        self.fig.tight_layout()
        self.fig.subplots_adjust(hspace=0.35)
        self.canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app  = StellarValidatorGUI(root)
    root.mainloop()
