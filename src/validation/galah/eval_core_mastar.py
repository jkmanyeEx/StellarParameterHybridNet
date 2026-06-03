import os
import numpy as np


def load_mastar_spectra():
    base_dir   = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    flux_path  = os.path.join(base_dir, "data", "mastar", "processed", "X_flux_clean.npy")
    label_path = os.path.join(base_dir, "data", "mastar", "processed", "Y_labels.npy")

    if not os.path.exists(flux_path):
        raise FileNotFoundError(f"Flux file not found: {flux_path}")
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Label file not found: {label_path}")

    X_flux_all   = np.load(flux_path)
    Y_labels_all = np.load(label_path)

    n = min(len(X_flux_all), len(Y_labels_all))
    X_flux_all   = X_flux_all[:n]
    Y_labels_all = Y_labels_all[:n]

    valid_mask    = (Y_labels_all[:, 0] > -900) & \
                    (Y_labels_all[:, 1] > -900) & \
                    (Y_labels_all[:, 2] > -900)
    valid_indices = np.where(valid_mask)[0]   # 원본 배열 기준 위치
    X_flux_clean  = X_flux_all[valid_mask]
    Y_labels_clean = Y_labels_all[valid_mask]

    total      = len(X_flux_clean)
    train_size = int(0.8 * total)

    # engine.py와 동일한 RNG (np.random.default_rng(42)) + 동일한 shuffle 로직
    # → eval 시 val split이 훈련 val split과 정확히 일치 보장
    indices = np.arange(total)
    rng = np.random.default_rng(42)
    rng.shuffle(indices)
    val_local_idx = indices[train_size:]   # valid_mask 기준 내부 인덱스

    X_flux_val    = X_flux_clean[val_local_idx]
    Y_labels_val  = Y_labels_clean[val_local_idx]
    orig_indices  = valid_indices[val_local_idx]   # 원본 배열 위치 (리포트용)
    valid_paths   = [f"MaStar_Sample_{idx:06d}.npy" for idx in orig_indices]

    print(f"[MaStar Val] Total clean samples: {total} | "
          f"Train: {train_size} | Val (test set): {total - train_size}")

    return X_flux_val, Y_labels_val, valid_paths
