# StellarParameterHybridNet Project Workspace

## 🔭 Project Overview
This project is a PyTorch-based Machine Learning pipeline for astrophysics. The goal is to predict fundamental stellar parameters from stellar spectra:
- **Effective Temperature ($T_{\text{eff}}$)** in Kelvin (K)
- **Surface Gravity ($\log g$)** in dex
- **Metallicity ($[\text{Fe}/\text{H}]$)** in dex

The architecture, **`StellarParameterHybridNet`**, fuses two data streams:
1. **1D CNN Branch**: Extracts features from raw spectral flux (aligned to a standard linear wavelength grid of 4563 pixels from 3650.0 Å to 10250.0 Å).
2. **Dense Branch**: Processes 30-dimensional extracted physical features (Equivalent Width, FWHM, and Depth for 10 key stellar absorption lines).

The model is trained using the **MaStar (MaNGA Stellar Library)** dataset from SDSS DR17 (6,085 spectra).

---

> [!IMPORTANT]
> **Antigravity Instructions**: Antigravity must always strictly follow this redesigned directory structure. Any new modular libraries must be placed in `src/`, run scripts in `scripts/`, report files in `report/`, and weights in `weights/`. Additionally, whenever code behaviors or file structural changes are introduced, Antigravity must immediately update `README.md` to keep documentation fully aligned.

## 📂 Directory Structure & Workflow

The codebase has been refactored into a modern, production-ready research structure:

### 1. Library Code ([`src/`](file:///Users/devmeko/Documents/KSA/3rdSem/GenAstro/TermProject/TermProject/src/))
All core Python sub-modules live here, structured as a package:
*   **[`src/data/`](file:///Users/devmeko/Documents/KSA/3rdSem/GenAstro/TermProject/TermProject/src/data/)**: Handles loading and preprocessing.
    *   [`preprocess_flux.py`](file:///Users/devmeko/Documents/KSA/3rdSem/GenAstro/TermProject/TermProject/src/data/preprocess_flux.py): Applies pixel masking and continuum normalization.
    *   [`extract_features.py`](file:///Users/devmeko/Documents/KSA/3rdSem/GenAstro/TermProject/TermProject/src/data/extract_features.py): Fits Gaussian profiles to extract 30D absorption line features.
    *   [`extract_labels.py`](file:///Users/devmeko/Documents/KSA/3rdSem/GenAstro/TermProject/TermProject/src/data/extract_labels.py): Extracts catalog target values.
    *   [`dataset.py`](file:///Users/devmeko/Documents/KSA/3rdSem/GenAstro/TermProject/TermProject/src/data/dataset.py): PyTorch `Dataset` that slices, normalizes, and packages data.
*   **[`src/models/`](file:///Users/devmeko/Documents/KSA/3rdSem/GenAstro/TermProject/TermProject/src/models/)**: Defines neural network layers.
    *   [`hybrid_net.py`](file:///Users/devmeko/Documents/KSA/3rdSem/GenAstro/TermProject/TermProject/src/models/hybrid_net.py): Combines all branch components.
    *   [`cnn_branch.py`](file:///Users/devmeko/Documents/KSA/3rdSem/GenAstro/TermProject/TermProject/src/models/cnn_branch.py): 1D CNN with Residual blocks (`ResBlock1D`).
    *   [`dense_branch.py`](file:///Users/devmeko/Documents/KSA/3rdSem/GenAstro/TermProject/TermProject/src/models/dense_branch.py): Dense layer stream for 30D features.
    *   [`fusion.py`](file:///Users/devmeko/Documents/KSA/3rdSem/GenAstro/TermProject/TermProject/src/models/fusion.py): Merges latent layers.
    *   [`output_branch.py`](file:///Users/devmeko/Documents/KSA/3rdSem/GenAstro/TermProject/TermProject/src/models/output_branch.py): Computes `CrossModalAttention` gating and final parameter projection.
*   **[`src/training/`](file:///Users/devmeko/Documents/KSA/3rdSem/GenAstro/TermProject/TermProject/src/training/)**:
    *   [`engine.py`](file:///Users/devmeko/Documents/KSA/3rdSem/GenAstro/TermProject/TermProject/src/training/engine.py): Executes the model optimization routine.
*   **[`src/validation/`](file:///Users/devmeko/Documents/KSA/3rdSem/GenAstro/TermProject/TermProject/src/validation/)**:
    *   [`eval_core.py`](file:///Users/devmeko/Documents/KSA/3rdSem/GenAstro/TermProject/TermProject/src/validation/eval_core.py): Resamples and normalizes external spectrograph FITS files.
    *   [`error_calculator.py`](file:///Users/devmeko/Documents/KSA/3rdSem/GenAstro/TermProject/TermProject/src/validation/error_calculator.py): Evaluates uncalibrated and calibrated predictions.
    *   [`xai_analyzer.py`](file:///Users/devmeko/Documents/KSA/3rdSem/GenAstro/TermProject/TermProject/src/validation/xai_analyzer.py): Calculates cumulative Jacobian sensitivity metrics.
    *   [`eval_core_mastar.py`](file:///Users/devmeko/Documents/KSA/3rdSem/GenAstro/TermProject/TermProject/src/validation/eval_core_mastar.py): Splits validation folds of the training set.
    *   [`error_calculation_mastar.py`](file:///Users/devmeko/Documents/KSA/3rdSem/GenAstro/TermProject/TermProject/src/validation/error_calculation_mastar.py): Cross-validates metrics against MaStar targets.
*   **[`src/utils/`](file:///Users/devmeko/Documents/KSA/3rdSem/GenAstro/TermProject/TermProject/src/utils/)**:
    *   [`config.py`](file:///Users/devmeko/Documents/KSA/3rdSem/GenAstro/TermProject/TermProject/src/utils/config.py): Hyperparameters and hardware configuration.
    *   [`loss_opt.py`](file:///Users/devmeko/Documents/KSA/3rdSem/GenAstro/TermProject/TermProject/src/utils/loss_opt.py): Loss function and optimization wrappers.

### 2. Run Scripts ([`scripts/`](file:///Users/devmeko/Documents/KSA/3rdSem/GenAstro/TermProject/TermProject/scripts/))
Lightweight executable scripts mapping to package modules:
*   [`train.py`](file:///Users/devmeko/Documents/KSA/3rdSem/GenAstro/TermProject/TermProject/scripts/train.py): Runs training.
*   [`evaluate.py`](file:///Users/devmeko/Documents/KSA/3rdSem/GenAstro/TermProject/TermProject/scripts/evaluate.py): Evaluates predictions on real spec FITS files.
*   [`evaluate_mastar.py`](file:///Users/devmeko/Documents/KSA/3rdSem/GenAstro/TermProject/TermProject/scripts/evaluate_mastar.py): Runs cross-validation against MaStar validation dataset folds.
*   [`xai_analysis.py`](file:///Users/devmeko/Documents/KSA/3rdSem/GenAstro/TermProject/TermProject/scripts/xai_analysis.py): Performs Jacobian analysis on real specs.
*   [`gui.py`](file:///Users/devmeko/Documents/KSA/3rdSem/GenAstro/TermProject/TermProject/scripts/gui.py): GUI validator for SDSS spec FITS.
*   [`gui_mastar.py`](file:///Users/devmeko/Documents/KSA/3rdSem/GenAstro/TermProject/TermProject/scripts/gui_mastar.py): GUI validator for MaStar validation sets.
*   [`compare_domains.py`](file:///Users/devmeko/Documents/KSA/3rdSem/GenAstro/TermProject/TermProject/scripts/compare_domains.py): Compares flux normalization distributions.
*   [`download_spec.py`](file:///Users/devmeko/Documents/KSA/3rdSem/GenAstro/TermProject/TermProject/scripts/download_spec.py): Downloads SDSS specs listed in catalog CSV.
*   [`spec_analyzer.py`](file:///Users/devmeko/Documents/KSA/3rdSem/GenAstro/TermProject/TermProject/scripts/spec_analyzer.py): Displays FITS HDU headers and properties.

### 3. Data Store ([`data/`](file:///Users/devmeko/Documents/KSA/3rdSem/GenAstro/TermProject/TermProject/data/))
Organizes dataset structures on disk:
*   **[`data/raw/`](file:///Users/devmeko/Documents/KSA/3rdSem/GenAstro/TermProject/TermProject/data/raw/)**: Large catalog files (e.g. `mastar-goodspec-v2_4_3-v1_0_2.fits`).
*   **[`data/processed/`](file:///Users/devmeko/Documents/KSA/3rdSem/GenAstro/TermProject/TermProject/data/processed/)**: Continuum-normalized matrices, labels, and statistics.
*   **[`data/validation_dataset/`](file:///Users/devmeko/Documents/KSA/3rdSem/GenAstro/TermProject/TermProject/data/validation_dataset/)**: External SDSS spectra FITS files and catalog index files.

### 4. Weights & Checkpoints ([`weights/`](file:///Users/devmeko/Documents/KSA/3rdSem/GenAstro/TermProject/TermProject/weights/))
Contains `.pth` weight checkpoints and training loss convergence visualizations.

### 5. Report Records ([`report/`](file:///Users/devmeko/Documents/KSA/3rdSem/GenAstro/TermProject/TermProject/report/))
Presents evaluation summaries for catalog comparisons:
*   [`dataset_error_report.txt`](file:///Users/devmeko/Documents/KSA/3rdSem/GenAstro/TermProject/TermProject/report/dataset_error_report.txt)
*   [`dataset_error_report_mastar.txt`](file:///Users/devmeko/Documents/KSA/3rdSem/GenAstro/TermProject/TermProject/report/dataset_error_report_mastar.txt)
*   [`xai_physics_report.txt`](file:///Users/devmeko/Documents/KSA/3rdSem/GenAstro/TermProject/TermProject/report/xai_physics_report.txt)

---

## 🐛 Recent Bugs & Learnings

### 1. The "Mean Collapse" Phenomenon
*   **Problem**: Model collapsed to statistical mean shortcuts during early optimization.
*   **Cause**: High magnitude of raw temperature labels ($T_{\text{eff}}$) dominated standard MSE loss calculations, ignoring $\log g$ and $[\text{Fe}/\text{H}]$.
*   **Solution**: Target parameters are scaled to normalized space $[0, 1]$ inside the dataset pipeline, and denormalized only during metrics compilation.

### 2. Validation Domain Mismatch
*   **Problem**: Predictions on external specs produced negative $R^2$ scores.
*   **Cause**: Discrepancies between raw spectral fluxes (huge values) and normalized fluxes oscillating around $1.0$.
*   **Solution**: Implemented continuum normalization on validation spectra using a 201-pixel median filter and resolution matching before passing fluxes to inference.
