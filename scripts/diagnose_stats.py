import os
import sys
import numpy as np
from astropy.io import fits

# Resolve paths relative to project root
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LABEL_PATH   = os.path.join(BASE_DIR, "data", "processed", "Y_labels.npy")
FLUX_PATH    = os.path.join(BASE_DIR, "data", "processed", "X_flux_telluric.npy")
FEATURE_PATH = os.path.join(BASE_DIR, "data", "processed", "X_features_physical.npy")

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

    print("=== TRUE TRAINING LABEL STATS (copy these into error_calculator.py) ===")
    print(f"Clean samples: {clean.shape[0]}")
    print(f"LABEL_MEAN = np.array([{np.mean(clean,axis=0)[0]:.6f}, "
          f"{np.mean(clean,axis=0)[1]:.6f}, {np.mean(clean,axis=0)[2]:.6f}])")
    print(f"LABEL_STD  = np.array([{np.std(clean,axis=0)[0]:.6f}, "
          f"{np.std(clean,axis=0)[1]:.6f}, {np.std(clean,axis=0)[2]:.6f}])")

    clean_feat = features[mask]
    print("\n=== TRUE 18D FEATURE STATS ===")
    print("FEATURE_MEAN =", np.array2string(np.mean(clean_feat, axis=0), precision=6, separator=', '))
    print("FEATURE_STD  =", np.array2string(np.std(clean_feat, axis=0) + 1e-8, precision=6, separator=', '))

    print("\n=== SPEC FITS GROUND-TRUTH AVAILABILITY ===")
    dataset_dir = os.path.join(BASE_DIR, "data", "validation_dataset")
    for f_name in ["spec-3615-55179-0010.fits",
                   "spec-3615-55179-0022.fits",
                   "spec-3615-55179-0045.fits"]:
        fn = os.path.join(dataset_dir, f_name)
        if not os.path.exists(fn):
            print(f"{f_name}: MISSING in {dataset_dir}")
            continue
        with fits.open(fn) as h:
            s = h[2].data
            print(f"\n{f_name}:")
            print(f"   ELODIE_TEFF={float(s['ELODIE_TEFF'][0]):.1f}  "
                  f"ELODIE_LOGG={float(s['ELODIE_LOGG'][0]):.2f}  "
                  f"ELODIE_FEH={float(s['ELODIE_FEH'][0]):.2f}")
            print(f"   CLASS={str(s['CLASS'][0]).strip()}  "
                  f"SUBCLASS={str(s['SUBCLASS'][0]).strip()}  "
                  f"Z={float(s['Z'][0]):.5f}")

if __name__ == "__main__":
    main()
