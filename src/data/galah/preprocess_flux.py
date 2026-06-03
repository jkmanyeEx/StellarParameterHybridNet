import numpy as np
import os
from multiprocessing import Pool, cpu_count
from scipy.ndimage import median_filter

try:
    from src.utils.galah.config import CPU_WORKERS_PREPROCESS
except ImportError:
    CPU_WORKERS_PREPROCESS = max(1, cpu_count() - 1)


def _normalize_single_spectrum(args):
    """
    단일 스펙트럼 전처리.
    """
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


def generate_synthetic_galah_dataset(out_dir, num_stars=120):
    """
    FITS 파일이 없을 경우, 15개 흡수선이 포함된 합성 4-Arm 데이터셋을 생성합니다.
    """
    print(f"⚠️  Raw FITS not found. Generating {num_stars} synthetic GALAH spectra...")
    
    # 4 arms standard wave grids
    wave_grids = [
        np.linspace(4713, 4903, 4000), # CCD1
        np.linspace(5648, 5873, 4000), # CCD2
        np.linspace(6478, 6737, 4000), # CCD3
        np.linspace(7585, 7887, 4000)  # CCD4
    ]
    standard_wave = np.stack(wave_grids, axis=0) # (4, 4000)

    # 15 target lines
    target_lines = [
        (4861.3, 1.5, 0.4), (4882.1, 1.0, 0.2), (4703.0, 1.2, 0.3), (4897.4, 0.8, 0.15), # CCD1
        (5662.5, 1.0, 0.25), (5711.1, 1.2, 0.35), (5782.1, 0.9, 0.2), (5862.4, 0.7, 0.15), # CCD2
        (6562.8, 2.0, 0.5), (6494.9, 1.1, 0.3), (6499.7, 1.0, 0.2), (6707.8, 0.8, 0.15), # CCD3
        (7748.3, 1.2, 0.3), (7699.0, 1.5, 0.4), (7772.0, 1.1, 0.25) # CCD4
    ]

    np.random.seed(42)
    fluxes = []
    star_ids = []

    for idx in range(num_stars):
        star_id = f"G{idx:05d}"
        star_ids.append(star_id)

        # Teff, logg, [Fe/H]에 따라 흡수 깊이 조절을 모사
        teff_scale = np.random.uniform(0.7, 1.3)
        logg_scale = np.random.uniform(0.8, 1.2)
        feh_scale = np.random.uniform(0.5, 1.5)

        star_arms = []
        for arm_idx, wave in enumerate(wave_grids):
            # Flat continuum with small slope
            slope = np.random.uniform(-0.02, 0.02)
            flux = 1.0 + slope * np.linspace(-1.0, 1.0, 4000)
            
            # Add absorption lines falling in this arm
            for center, width, depth in target_lines:
                if wave[0] <= center <= wave[-1]:
                    # Adjust depth based on scales
                    if "6562" in f"{center:.1f}" or "4861" in f"{center:.1f}":
                        # Balmer lines: Teff dependent
                        act_depth = depth * teff_scale
                    else:
                        # Metal lines: feh dependent
                        act_depth = depth * feh_scale * logg_scale
                    act_depth = np.clip(act_depth, 0.0, 0.95)
                    
                    profile = act_depth * np.exp(-(wave - center)**2 / (2 * width**2))
                    flux -= profile
            
            # Add small noise before z-score
            flux += np.random.normal(0, 0.01, 4000)
            star_arms.append(flux)
            
        fluxes.append(np.stack(star_arms, axis=0))

    X_flux = np.array(fluxes, dtype=np.float32)
    
    np.save(os.path.join(out_dir, "X_flux_clean.npy"), X_flux)
    np.save(os.path.join(out_dir, "star_ids.npy"), np.array(star_ids))
    np.save(os.path.join(out_dir, "standard_wave.npy"), standard_wave)
    print(f"Saved synthetic GALAH data to {out_dir}")
    print(f"   > X_flux_clean shape: {X_flux.shape}")
    print(f"   > standard_wave shape: {standard_wave.shape}")


def run_galah_preprocessing_pipeline(raw_dir, out_dir):
    """
    GALAH 원본 FITS 파일이 존재하는 경우 이를 파싱하고 전처리합니다.
    없으면 합성 데이터로 대체합니다.
    """
    os.makedirs(out_dir, exist_ok=True)
    galah_catalog = os.path.join(raw_dir, "galah_dr4_allstar.csv")
    spectra_dir = os.path.join(raw_dir, "spectra")
    
    if not os.path.exists(galah_catalog) or not os.path.exists(spectra_dir):
        generate_synthetic_galah_dataset(out_dir)
        return

    import astropy.io.fits as fits
    from scipy.interpolate import interp1d
    import csv
    
    print(f"[GALAH] Reading catalog from: {galah_catalog}")
    stars = []
    with open(galah_catalog, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stars.append(row["sobject_id"].strip())
            
    print(f"Total stars in catalog: {len(stars)}")
    
    # 4 arms standard wave grids
    wave_grids = [
        np.linspace(4713, 4903, 4000), # CCD1
        np.linspace(5648, 5873, 4000), # CCD2
        np.linspace(6478, 6737, 4000), # CCD3
        np.linspace(7585, 7887, 4000)  # CCD4
    ]
    standard_wave = np.stack(wave_grids, axis=0) # (4, 4000)
    
    fluxes = []
    star_ids = []
    
    for idx, sid in enumerate(stars):
        # Check if all 4 arms exist for this star
        arms_exist = True
        arm_files = {}
        for filt in ["B", "G", "R", "I"]:
            fpath = os.path.join(spectra_dir, f"{sid}_{filt}.fits")
            if not os.path.exists(fpath):
                arms_exist = False
                break
            arm_files[filt] = fpath
            
        if not arms_exist:
            continue
            
        star_arms = []
        try:
            for arm_idx, filt in enumerate(["B", "G", "R", "I"]):
                fpath = arm_files[filt]
                with fits.open(fpath) as h:
                    hdr = h[0].header
                    flux = h[0].data.astype(float)
                    crval1 = hdr['CRVAL1']
                    cdelt1 = hdr['CDELT1']
                    crpix1 = hdr.get('CRPIX1', 1.0)
                    wave = crval1 + (np.arange(len(flux)) - crpix1 + 1) * cdelt1
                    
                    target_wave = wave_grids[arm_idx]
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
        generate_synthetic_galah_dataset(out_dir)
        return
        
    X_flux = np.array(fluxes, dtype=np.float32)
    np.save(os.path.join(out_dir, "X_flux_clean.npy"), X_flux)
    np.save(os.path.join(out_dir, "star_ids.npy"), np.array(star_ids))
    np.save(os.path.join(out_dir, "standard_wave.npy"), standard_wave)
    
    print(f"Saved real preprocessed GALAH data to {out_dir}")
    print(f"   > X_flux_clean shape: {X_flux.shape}")
    print(f"   > standard_wave shape: {standard_wave.shape}")


if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    raw_dir  = os.path.join(base_dir, "data", "galah", "raw")
    out_dir  = os.path.join(base_dir, "data", "galah", "processed")
    
    run_galah_preprocessing_pipeline(raw_dir, out_dir)
