"""
Reconstruct the MaStar validation split used during training.

The split is reproduced by applying the identical RNG seed and shuffling
procedure as src/training/mastar/engine.py, operating on the same
valid_indices array (absolute positions in the original flux matrix).
This guarantees that the validation set evaluated here is disjoint from
the training set the model was fitted on.
"""

import os
import numpy as np


def load_mastar_spectra():
    """
    Load the MaStar held-out validation split.

    Reproduces the exact train/val partition from engine.py:
      1. Align flux and label arrays to the same length n.
      2. Filter rows where any label is -999 (no VAC match).
      3. Record the absolute indices of valid rows (valid_indices).
      4. Shuffle valid_indices with np.random.default_rng(seed=42).
      5. Take the last 20 % as the validation set.

    Returns
    -------
    X_flux_val   : np.ndarray shape (n_val, 4563)
    Y_labels_val : np.ndarray shape (n_val, 3)  — [T_eff, log g, [Fe/H]]
    orig_indices : list of int — original row positions (for reporting)
    """
    base_dir   = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    flux_path  = os.path.join(base_dir, "data", "mastar", "processed", "X_flux_clean.npy")
    label_path = os.path.join(base_dir, "data", "mastar", "processed", "Y_labels.npy")

    if not os.path.exists(flux_path):
        raise FileNotFoundError(
            f"MaStar flux matrix not found at: {flux_path}\n"
            "Execute src/data/mastar/preprocess_flux.py first."
        )
    if not os.path.exists(label_path):
        raise FileNotFoundError(
            f"MaStar label matrix not found at: {label_path}\n"
            "Execute src/data/mastar/extract_labels.py first."
        )

    X_flux_all   = np.load(flux_path,  mmap_mode='r')
    Y_labels_all = np.load(label_path, mmap_mode='r')

    n = min(len(X_flux_all), len(Y_labels_all))

    # Step 3: valid_indices — absolute positions, identical to engine.py
    raw_labels   = Y_labels_all[:n]
    valid_mask   = (raw_labels[:, 0] > -900) & \
                   (raw_labels[:, 1] > -900) & \
                   (raw_labels[:, 2] > -900)
    valid_indices = np.where(valid_mask)[0]   # absolute positions in [:n]

    # Step 4: shuffle with the same RNG as engine.py
    rng = np.random.default_rng(42)
    rng.shuffle(valid_indices)

    # Step 5: last 20 % → validation set
    train_size    = int(0.8 * len(valid_indices))
    val_indices   = valid_indices[train_size:]   # absolute positions

    X_flux_val    = X_flux_all[val_indices]
    Y_labels_val  = Y_labels_all[val_indices]
    orig_indices  = val_indices.tolist()

    total_valid = len(valid_indices)
    print(f"[MaStar Val] Total valid samples : {total_valid} | "
          f"Train : {train_size} | Validation : {len(val_indices)}")

    return np.array(X_flux_val), np.array(Y_labels_val), orig_indices
