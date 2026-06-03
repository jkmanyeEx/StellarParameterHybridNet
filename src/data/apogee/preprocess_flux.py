import numpy as np
import os
from multiprocessing import Pool, cpu_count
from scipy.ndimage import median_filter

try:
    from src.utils.apogee.config import CPU_WORKERS_PREPROCESS
except ImportError:
    CPU_WORKERS_PREPROCESS = max(1, cpu_count() - 1)


def _normalize_single_spectrum(args):
    raw_flux, pixel_mask = args
    raw_flux = raw_flux.astype(float)
    safe_flux = np.copy(raw_flux)
    if pixel_mask is not None:
        safe_flux[pixel_mask != 0] = np.nan

    fill = np.nanmedian(safe_flux) if np.any(np.isfinite(safe_flux)) else 1.0
    bg_continuum = median_filter(np.nan_to_num(safe_flux, nan=fill), size=201)
    bg_continuum = np.where(bg_continuum <= 0, 1e-5, bg_continuum)
    norm_flux = safe_flux / bg_continuum

    norm_median = np.nanmedian(norm_flux)
    norm_std    = np.nanstd(norm_flux)
    spike_mask  = np.abs(norm_flux - norm_median) > 5.0 * norm_std
    spike_mask |= ~np.isfinite(norm_flux)

    if spike_mask.any():
        px    = np.arange(len(norm_flux))
        valid = ~spike_mask
        norm_flux = np.interp(px, px[valid], norm_flux[valid]) if valid.sum() > 10 \
                    else np.ones_like(norm_flux)

    return norm_flux.astype(np.float32)


def generate_synthetic_apogee_dataset(out_dir, num_stars=120):
    """
    FITS 파일이 없을 경우, 10개 흡수선이 포함된 합성 3-Chip 데이터를 생성합니다.
    """
    print(f"⚠️  Raw FITS not found. Generating {num_stars} synthetic APOGEE spectra...")
    
    # 3 arms standard wave grids
    wave_grids = [
        np.linspace(15140, 15810, 2800), # Blue chip
        np.linspace(15850, 16430, 2800), # Green chip
        np.linspace(16470, 16960, 2800)  # Red chip
    ]
    standard_wave = np.stack(wave_grids, axis=0) # (3, 2800)

    # 10 target lines
    target_lines = [
        (15200.0, 1.2, 0.25), (15648.5, 1.5, 0.35), (15749.0, 1.0, 0.2), # Blue chip
        (15886.2, 1.1, 0.3), (15960.0, 1.3, 0.25), (15884.9, 1.6, 0.4), (16040.0, 0.8, 0.15), # Green chip
        (16680.0, 1.2, 0.25), (16754.7, 1.4, 0.3), (16811.1, 1.8, 0.45) # Red chip
    ]

    np.random.seed(42)
    fluxes = []
    star_ids = []

    for idx in range(num_stars):
        star_id = f"A{idx:05d}"
        star_ids.append(star_id)

        teff_scale = np.random.uniform(0.7, 1.3)
        logg_scale = np.random.uniform(0.8, 1.2)
        feh_scale = np.random.uniform(0.5, 1.5)

        star_arms = []
        for arm_idx, wave in enumerate(wave_grids):
            slope = np.random.uniform(-0.02, 0.02)
            flux = 1.0 + slope * np.linspace(-1.0, 1.0, 2800)
            
            for center, width, depth in target_lines:
                if wave[0] <= center <= wave[-1]:
                    if "16811" in f"{center:.1f}" or "15884" in f"{center:.1f}":
                        # Brackett lines: Teff dependent
                        act_depth = depth * teff_scale
                    else:
                        # Metal lines
                        act_depth = depth * feh_scale * logg_scale
                    act_depth = np.clip(act_depth, 0.0, 0.95)
                    
                    profile = act_depth * np.exp(-(wave - center)**2 / (2 * width**2))
                    flux -= profile
            
            flux += np.random.normal(0, 0.01, 2800)
            star_arms.append(flux)
            
        fluxes.append(np.stack(star_arms, axis=0))

    X_flux = np.array(fluxes, dtype=np.float32)
    
    np.save(os.path.join(out_dir, "X_flux_clean.npy"), X_flux)
    np.save(os.path.join(out_dir, "star_ids.npy"), np.array(star_ids))
    np.save(os.path.join(out_dir, "standard_wave.npy"), standard_wave)
    print(f"Saved synthetic APOGEE data to {out_dir}")
    print(f"   > X_flux_clean shape: {X_flux.shape}")
    print(f"   > standard_wave shape: {standard_wave.shape}")


def run_apogee_preprocessing_pipeline(raw_dir, out_dir):
    """
    APOGEE FITS가 있을 경우 전처리, 없으면 합성 데이터 생성.
    """
    os.makedirs(out_dir, exist_ok=True)
    apogee_catalog = os.path.join(raw_dir, "allStar-dr17.csv")
    spectra_dir = os.path.join(raw_dir, "spectra")
    
    if not os.path.exists(apogee_catalog) or not os.path.exists(spectra_dir):
        generate_synthetic_apogee_dataset(out_dir)
        return

    import astropy.io.fits as fits
    from scipy.interpolate import interp1d
    import csv
    
    print(f"[APOGEE] Reading catalog from: {apogee_catalog}")
    stars = []
    with open(apogee_catalog, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stars.append(row["apogee_id"].strip())
            
    print(f"Total stars in catalog: {len(stars)}")
    
    # 3 arms standard wave grids
    wave_grids = [
        np.linspace(15140, 15810, 2800), # Blue chip
        np.linspace(15850, 16430, 2800), # Green chip
        np.linspace(16470, 16960, 2800)  # Red chip
    ]
    standard_wave = np.stack(wave_grids, axis=0) # (3, 2800)
    
    fluxes = []
    star_ids = []
    
    for idx, sid in enumerate(stars):
        fpath = os.path.join(spectra_dir, f"{sid}.fits")
        if not os.path.exists(fpath):
            continue
            
        try:
            with fits.open(fpath) as h:
                flux = h[1].data.astype(float)
                hdr = h[1].header
                naxis1 = hdr['NAXIS1']
                crval1 = hdr['CRVAL1']
                cdelt1 = hdr['CDELT1']
                wave = 10 ** (crval1 + np.arange(naxis1) * cdelt1)
                
                star_arms = []
                for arm_idx, target_wave in enumerate(wave_grids):
                    f_interp = interp1d(wave, flux, kind='linear', bounds_error=False, fill_value=np.nanmedian(flux))
                    resampled_flux = f_interp(target_wave)
                    
                    norm_flux = _normalize_single_spectrum((resampled_flux, None))
                    star_arms.append(norm_flux)
                    
                fluxes.append(np.stack(star_arms, axis=0))
                star_ids.append(sid)
                
        except Exception as e:
            print(f"Error processing star {sid}: {e}")
            continue
            
    if len(fluxes) == 0:
        print("No valid real stars processed. Falling back to synthetic.")
        generate_synthetic_apogee_dataset(out_dir)
        return
        
    X_flux = np.array(fluxes, dtype=np.float32)
    np.save(os.path.join(out_dir, "X_flux_clean.npy"), X_flux)
    np.save(os.path.join(out_dir, "star_ids.npy"), np.array(star_ids))
    np.save(os.path.join(out_dir, "standard_wave.npy"), standard_wave)
    
    print(f"Saved real preprocessed APOGEE data to {out_dir}")
    print(f"   > X_flux_clean shape: {X_flux.shape}")
    print(f"   > standard_wave shape: {standard_wave.shape}")


if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    raw_dir  = os.path.join(base_dir, "data", "apogee", "raw")
    out_dir  = os.path.join(base_dir, "data", "apogee", "processed")
    
    run_apogee_preprocessing_pipeline(raw_dir, out_dir)
