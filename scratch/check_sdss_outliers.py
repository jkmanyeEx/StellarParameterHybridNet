import os
import numpy as np
import torch
import sys

sys.path.append("/Users/devmeko/Documents/KSA/3rdSem/GenAstro/TermProject/TermProject")

from src.validation.error_calculator import collect_spec_fits_files, load_spectra_from_fits_list

BASE_DIR = "/Users/devmeko/Documents/KSA/3rdSem/GenAstro/TermProject/TermProject"
dataset_dir = os.path.join(BASE_DIR, "data", "validation_dataset")
csv_path = os.path.join(dataset_dir, "Skyserver_SQL6_1_2026 10_51_26 PM.csv")

all_spec_files = collect_spec_fits_files(dataset_dir)
if all_spec_files:
    X_flux, Y_true, _ = load_spectra_from_fits_list(all_spec_files, csv_path, dataset_dir)
    print(f"Loaded {X_flux.shape[0]} SDSS validation spectra.")
    maxs = np.max(X_flux, axis=1)
    mins = np.min(X_flux, axis=1)
    print(f"SDSS max values range: {maxs.min():.3f} to {maxs.max():.3f}")
    print(f"SDSS min values range: {mins.min():.3f} to {mins.max():.3f}")
    
    outliers_max = np.sum(maxs > 3.0)
    outliers_min = np.sum(mins < 0.0)
    print(f"SDSS spectra with max > 3.0: {outliers_max}")
    print(f"SDSS spectra with min < 0.0: {outliers_min}")
else:
    print("No files found")
