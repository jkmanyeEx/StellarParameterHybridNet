import numpy as np
from scipy.optimize import curve_fit
import os

try:
    from src.utils.mastar.config import CPU_WORKERS_PREPROCESS
except ImportError:
    try:
        from utils.config import CPU_WORKERS_PREPROCESS
    except ImportError:
        from multiprocessing import cpu_count
        CPU_WORKERS_PREPROCESS = max(1, cpu_count() - 1)


def gaussian_profile(x, a, x0, sigma, c):
    return c - a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))


def extract_30d_features_single_star(wave, norm_flux):
    """
    10개 흡수선 x 3값 (EW, FWHM, depth) = 30D 피처 벡터 추출.

    선 구성 (10개):
      Balmer:    H_alpha, H_beta, H_gamma, H_delta  (+-20A)
      Calcium:   Ca_II_K, Ca_II_H                   (+-15A)
      Magnesium: Mg_I_b                             (+-20A)
      Iron:      Fe_I_5270, Fe_I_4383               (+-15A)
      Sodium:    Na_I                               (+-15A)

    Gaussian 실패 시 비모수적 fallback (xai_analyzer.py와 동일):
      depth: 1 - min(flux)
      EW:    직접 적분
      FWHM:  반값 폭 추정
    -> 훈련 피처와 평가 피처의 fallback이 동일해 train-inference 일치
    """
    target_lines = {
        # Balmer series (Teff)
        "H_alpha":   (6563.0, 20),
        "H_beta":    (4861.0, 20),
        "H_gamma":   (4340.0, 20),
        "H_delta":   (4102.0, 20),
        # Calcium (log g + [Fe/H])
        "Ca_II_K":   (3934.0, 15),
        "Ca_II_H":   (3968.0, 15),
        # Magnesium (log g 전진 지표, XAI gravity sensitivity 1위)
        "Mg_I_b":    (5175.0, 20),
        # Iron ([Fe/H])
        "Fe_I_5270": (5270.0, 15),
        "Fe_I_4383": (4383.0, 15),
        # Sodium (log g + [Fe/H])
        "Na_I":      (5892.0, 15),
    }

    feature_vector = []

    for line_name, (center_wave, window_half) in target_lines.items():
        mask  = (wave >= center_wave - window_half) & (wave <= center_wave + window_half)
        w_sub = wave[mask]
        f_sub = norm_flux[mask]

        if len(w_sub) < 5:
            feature_vector.extend([0.0, 0.0, 0.0])
            continue

        p0     = [1.0 - np.min(f_sub), center_wave, 2.0, 1.0]
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
            # 비모수적 fallback (xai_analyzer.py와 동일)
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
    """multiprocessing.Pool에서 호출 가능하도록 모듈 최상위레벨에 정의."""
    wave, flux = args
    return extract_30d_features_single_star(wave, flux)


def main():
    from multiprocessing import Pool, cpu_count
    from tqdm import tqdm

    print("Loading exported data...")
    base_dir  = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    flux_path = os.path.join(base_dir, "data", "mastar", "processed", "X_flux_clean.npy")
    wave_path = os.path.join(base_dir, "data", "mastar", "processed", "standard_wave.npy")
    out_path  = os.path.join(base_dir, "data", "mastar", "processed", "X_features_physical.npy")

    if not os.path.exists(flux_path):
        raise FileNotFoundError(f"No flux file found at: {flux_path}")

    X_flux_clean  = np.load(flux_path)
    standard_wave  = np.load(wave_path)
    total_stars    = X_flux_clean.shape[0]

    print(f"   > Flux matrix shape : {X_flux_clean.shape}")
    print(f"   > Wave grid shape   : {standard_wave.shape}")
    print(f"\nExtracting 30D features (10 lines x 3) "
          f"with {CPU_WORKERS_PREPROCESS} workers (of {cpu_count()} total cores)...")

    args = [(standard_wave, X_flux_clean[i]) for i in range(total_stars)]

    with Pool(processes=CPU_WORKERS_PREPROCESS) as pool:
        X_features_list = list(tqdm(
            pool.imap(_extract_worker, args, chunksize=500),
            total=total_stars,
            desc="30D Feature Extraction",
            unit="star"
        ))

    X_features = np.array(X_features_list, dtype=np.float32)
    np.save(out_path, X_features)
    print(f"Success! Feature matrix saved. Shape: {X_features.shape}")
    print(f"   > Expected: ({total_stars}, 30)")


if __name__ == "__main__":
    main()
