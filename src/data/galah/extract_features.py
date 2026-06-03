import numpy as np
from scipy.optimize import curve_fit
import os
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

try:
    from src.utils.galah.config import CPU_WORKERS_PREPROCESS
except ImportError:
    CPU_WORKERS_PREPROCESS = max(1, cpu_count() - 1)


def gaussian_profile(x, a, x0, sigma, c):
    return c - a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))


def extract_45d_features_single_star(wave_2d, norm_flux_2d):
    """
    15개 흡수선 x 3값 (EW, FWHM, depth) = 45D 피처 벡터 추출.
    각 흡수선이 해당하는 CCD arm (0~3)에서 피처를 추출합니다.
    """
    target_lines = {
        # CCD1 (Blue): 4713 - 4903
        "H_beta":      (4861.3, 15, 0),
        "Fe_I_4882":   (4882.1, 10, 0),
        "Mg_I_4703":   (4703.0, 10, 0),
        "Ba_II_4897":  (4897.4, 10, 0),
        # CCD2 (Green): 5648 - 5873
        "Fe_I_5662":   (5662.5, 10, 1),
        "Mg_I_5711":   (5711.1, 10, 1),
        "Fe_I_5782":   (5782.1, 10, 1),
        "Fe_I_5862":   (5862.4, 10, 1),
        # CCD3 (Red): 6478 - 6737
        "H_alpha":     (6562.8, 15, 2),
        "Fe_I_6495":   (6494.9, 10, 2),
        "Ca_I_6499":   (6499.7, 10, 2),
        "Li_I_6708":   (6707.8, 10, 2),
        # CCD4 (NIR): 7585 - 7887
        "Fe_I_7748":   (7748.3, 10, 3),
        "K_I_7699":    (7699.0, 10, 3),
        "O_I_7772":    (7772.0, 10, 3),
    }

    feature_vector = []

    for line_name, (center_wave, window_half, arm_idx) in target_lines.items():
        wave = wave_2d[arm_idx]
        flux = norm_flux_2d[arm_idx]

        mask  = (wave >= center_wave - window_half) & (wave <= center_wave + window_half)
        w_sub = wave[mask]
        f_sub = flux[mask]

        if len(w_sub) < 5:
            feature_vector.extend([0.0, 0.0, 0.0])
            continue

        p0     = [1.0 - np.min(f_sub), center_wave, 1.5, 1.0]
        bounds = ([0.0, center_wave - 5, 0.1, 0.8],
                  [1.0, center_wave + 5, 10.0, 1.2])

        try:
            popt, _ = curve_fit(gaussian_profile, w_sub, f_sub,
                                p0=p0, bounds=bounds, maxfev=800)
            a, x0, sigma, c = popt
            fwhm             = 2.355 * np.abs(sigma)
            depth            = a
            dw               = np.gradient(w_sub)
            equivalent_width = float(max(0.0, np.sum((1.0 - f_sub / c) * dw)))
            feature_vector.extend([equivalent_width, fwhm, depth])

        except (RuntimeError, ValueError):
            # 비모수적 fallback
            depth = float(max(0.0, 1.0 - np.min(f_sub)))
            dw    = np.gradient(w_sub)
            equivalent_width = float(max(0.0, np.sum((1.0 - f_sub) * dw)))

            half_val   = 1.0 - (depth / 2.0)
            below_half = np.where(f_sub < half_val)[0]
            if len(below_half) > 1:
                fwhm = float(w_sub[below_half[-1]] - w_sub[below_half[0]])
            else:
                fwhm = 3.0

            feature_vector.extend([equivalent_width, fwhm, depth])

    return np.array(feature_vector, dtype=np.float32)


def _extract_worker(args):
    wave, flux = args
    return extract_45d_features_single_star(wave, flux)


def main():
    print("Loading exported data for GALAH...")
    base_dir  = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    processed_dir = os.path.join(base_dir, "data", "galah", "processed")
    
    flux_path = os.path.join(processed_dir, "X_flux_clean.npy")
    wave_path = os.path.join(processed_dir, "standard_wave.npy")
    out_path  = os.path.join(processed_dir, "X_features_physical.npy")

    if not os.path.exists(flux_path):
        raise FileNotFoundError(f"No flux file found at: {flux_path}")

    X_flux_clean  = np.load(flux_path)
    standard_wave  = np.load(wave_path)
    total_stars    = X_flux_clean.shape[0]

    print(f"   > Flux matrix shape : {X_flux_clean.shape}")
    print(f"   > Wave grid shape   : {standard_wave.shape}")
    print(f"\nExtracting 45D features (15 lines x 3) "
          f"with {CPU_WORKERS_PREPROCESS} workers...")

    args = [(standard_wave, X_flux_clean[i]) for i in range(total_stars)]

    with Pool(processes=CPU_WORKERS_PREPROCESS) as pool:
        X_features_list = list(tqdm(
            pool.imap(_extract_worker, args, chunksize=100),
            total=total_stars,
            desc="45D GALAH Feature Extraction",
            unit="star"
        ))

    X_features = np.array(X_features_list, dtype=np.float32)
    np.save(out_path, X_features)
    print(f"Success! GALAH Feature matrix saved. Shape: {X_features.shape}")


if __name__ == "__main__":
    main()
