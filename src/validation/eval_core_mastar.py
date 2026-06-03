import os
import numpy as np
import torch
from torch.utils.data import random_split


def load_mastar_spectra():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    flux_path = os.path.join(base_dir, "data", "processed", "X_flux_clean.npy")
    label_path = os.path.join(base_dir, "data", "processed", "Y_labels.npy")

    if not os.path.exists(flux_path):
        raise FileNotFoundError(f"Flux file not found: {flux_path}")
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Label file not found: {label_path}")

    X_flux_all   = np.load(flux_path)
    Y_labels_all = np.load(label_path)

    n = min(len(X_flux_all), len(Y_labels_all))
    X_flux_all   = X_flux_all[:n]
    Y_labels_all = Y_labels_all[:n]

    valid_mask    = (Y_labels_all[:, 0] > -900) & (Y_labels_all[:, 1] > -900) & (Y_labels_all[:, 2] > -900)
    valid_indices = np.where(valid_mask)[0]
    X_flux_clean  = X_flux_all[valid_mask]
    Y_labels_clean = Y_labels_all[valid_mask]

    total = len(X_flux_clean)
    train_size = int(0.8 * total)
    val_size   = total - train_size

    indices = list(range(total))
    _, val_subset = random_split(
        indices,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    val_idx      = list(val_subset)
    X_flux_val   = X_flux_clean[val_idx]
    Y_labels_val = Y_labels_clean[val_idx]

    orig_indices  = valid_indices[val_idx]
    valid_paths   = [f"MaStar_Sample_{idx:06d}.npy" for idx in orig_indices]

    print(f"[MaStar Val] Total clean samples: {total} | "
          f"Train: {train_size} | Val (test set): {val_size}")

    return X_flux_val, Y_labels_val, valid_paths
