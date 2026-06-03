import os
import sys
import numpy as np

# Resolve paths relative to project root
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
LABEL_PATH   = os.path.join(BASE_DIR, "data", "apogee", "processed", "Y_labels.npy")
FLUX_PATH    = os.path.join(BASE_DIR, "data", "apogee", "processed", "X_flux_clean.npy")
FEATURE_PATH = os.path.join(BASE_DIR, "data", "apogee", "processed", "X_features_physical.npy")

def main():
    if not os.path.exists(LABEL_PATH):
        print(f"[Error] Labels path not found: {LABEL_PATH}")
        return

    labels   = np.load(LABEL_PATH)
    fluxes   = np.load(FLUX_PATH)
    features = np.load(FEATURE_PATH)

    n = min(len(labels), len(fluxes), len(features))
    labels, fluxes, features = labels[:n], fluxes[:n], features[:n]

    mask = (labels[:, 0] > -900) & (labels[:, 1] > -900) & (labels[:, 2] > -900)
    clean = labels[mask]

    print("=== TRUE APOGEE TRAINING LABEL STATS (copy these into error_calculator.py) ===")
    print(f"Clean samples: {clean.shape[0]}")
    print(f"LABEL_MEAN = np.array([{np.mean(clean,axis=0)[0]:.6f}, "
          f"{np.mean(clean,axis=0)[1]:.6f}, {np.mean(clean,axis=0)[2]:.6f}])")
    print(f"LABEL_STD  = np.array([{np.std(clean,axis=0)[0]:.6f}, "
          f"{np.std(clean,axis=0)[1]:.6f}, {np.std(clean,axis=0)[2]:.6f}])")

    clean_feat = features[mask]
    print(f"\n=== TRUE APOGEE {clean_feat.shape[1]}D FEATURE STATS ===")
    print("FEATURE_MEAN =", np.array2string(np.mean(clean_feat, axis=0), precision=6, separator=', '))
    print("FEATURE_STD  =", np.array2string(np.std(clean_feat, axis=0) + 1e-8, precision=6, separator=', '))

if __name__ == "__main__":
    main()
