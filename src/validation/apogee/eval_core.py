import os
import numpy as np
from astropy.io import fits
from scipy.interpolate import interp1d
from scipy.ndimage import median_filter, gaussian_filter1d


def continuum_normalize(flux, ivar=None, window=201):
    """
    Continuum-normalize a 1-D flux array using a median filter,
    IDENTICAL to the MaStar training pipeline (src/data/preprocess_flux.py).

    Parameters
    ----------
    flux   : 1-D array of raw flux values
    ivar   : 1-D inverse-variance array (optional, SDSS spec HDU1 'ivar').
             Pixels with ivar <= 0 are treated as bad and interpolated over,
             mirroring the MASK != 0 logic used in the MaStar pipeline.
    window : median filter kernel size for continuum estimation

    Pipeline (kept strictly identical to preprocess_flux.py):
      1. Mark bad pixels (ivar <= 0 or non-finite flux) as NaN
      2. Compute bg_continuum on a NaN-filled copy (nan_to_num)
      3. Divide flux by continuum  →  norm_flux (NaNs propagate)
      4. Detect spikes via 5σ clipping + non-finite check
      5. Replace spike/NaN pixels by linear interpolation from valid neighbours
    """
    flux = np.array(flux, dtype=float)
    safe_flux = flux.copy()

    # Step 1 — mask bad pixels → NaN (mirrors MaStar MASK != 0 logic)
    if ivar is not None:
        ivar = np.array(ivar, dtype=float)
        safe_flux[ivar <= 0] = np.nan
    safe_flux[~np.isfinite(safe_flux)] = np.nan

    # Step 2 — continuum on a NaN-free copy
    fill_value  = np.nanmedian(safe_flux) if np.any(np.isfinite(safe_flux)) else 1.0
    bg_continuum = median_filter(
        np.nan_to_num(safe_flux, nan=fill_value), size=window
    )
    bg_continuum = np.where(bg_continuum <= 0, 1e-5, bg_continuum)

    # Step 3 — normalise (NaN pixels stay NaN)
    norm_flux = safe_flux / bg_continuum

    # Step 4 & 5 — spike detection + linear interpolation
    norm_median = np.nanmedian(norm_flux)
    norm_std    = np.nanstd(norm_flux)
    spike_mask  = np.abs(norm_flux - norm_median) > 5.0 * norm_std
    spike_mask |= ~np.isfinite(norm_flux)   # NaN/Inf 포함

    if spike_mask.any():
        pixel_indices = np.arange(len(norm_flux))
        valid_mask    = ~spike_mask
        if valid_mask.sum() > 10:
            norm_flux = np.interp(
                pixel_indices,
                pixel_indices[valid_mask],
                norm_flux[valid_mask]
            )
        else:
            norm_flux = np.ones_like(norm_flux)

    return norm_flux


def align_wavelength_resolution(loglam, flux, target_pixel_size=4563,
                                target_wave_grid=None):
    """
    Convert SDSS spec (loglam array, flux array) onto a fixed linear grid.
    loglam           : 1-D array of log10(wavelength / Angstrom)
    flux             : 1-D flux array matching loglam (already continuum-normalized)
    target_wave_grid : pre-loaded wave grid array (pass to avoid repeated disk I/O).
                       If None, loads standard_wave.npy from disk.
    Returns aligned flux (1, target_pixel_size), or None if data is bad.
    """
    if target_wave_grid is not None:
        target_wave_grid = np.asarray(target_wave_grid)
    else:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        wave_path = os.path.join(base_dir, "data", "apogee", "processed", "standard_wave.npy")
        target_wave_grid = np.load(wave_path) if os.path.exists(wave_path) \
                           else np.linspace(3650.0, 10250.0, target_pixel_size)

    loglam = np.atleast_1d(loglam).flatten().astype(float)
    flux   = np.atleast_1d(flux).flatten().astype(float)

    if len(flux) < 10:
        print(f"   [SKIP] Flux too short ({len(flux)} pixels).")
        return None
    if np.all(flux == 0.0) or np.max(np.abs(flux)) == 0:
        print("   [SKIP] Flux is all zeros.")
        return None

    # Convert log10(lambda) -> linear lambda
    wave = 10.0 ** loglam

    clean = np.isfinite(flux) & np.isfinite(wave) & (flux > -900)
    if clean.sum() < 10:
        print(f"   [SKIP] Only {clean.sum()} valid pixels after masking.")
        return None

    try:
        f = interp1d(wave[clean], flux[clean],
                     kind='linear', bounds_error=False, fill_value=1.0)
        aligned = f(target_wave_grid)
        
        # ── Resolution matching: SDSS R≈2000 → MaStar R≈1800 ───────────────────
        # SDSS has slightly higher resolution than MaStar, so absorption lines
        # appear narrower in the eval data than in training data.
        # We apply a small Gaussian blur to match MaStar's instrumental profile.
        #
        # Exact Derivation (constant in log-space grid d(log10 lambda) = 0.0001):
        #   FWHM_MaStar = lambda / 1800
        #   FWHM_SDSS   = lambda / 2000
        #   FWHM_kernel = lambda * sqrt(1/1800^2 - 1/2000^2) ≈ 0.000242 * lambda
        #   Pixel scale: d(lambda)/d(pixel) = lambda * ln(10) * 0.0001 ≈ 0.000230 * lambda
        #   FWHM_kernel_pixels = 0.000242 / 0.000230 ≈ 1.05 pixels
        #   sigma_pixels = 1.05 / 2.355 ≈ 0.45 pixels
        aligned = gaussian_filter1d(aligned, sigma=0.45)
        
        aligned = aligned.reshape(1, -1)
    except Exception as e:
        print(f"   [SKIP] Interpolation failed: {e}")
        return None

    return aligned


def read_sdss_spec(fits_path, apply_continuum_norm=True):
    """
    Read an SDSS spec FITS file and return:
      - flux   : 1-D numpy array  (continuum-normalized to match MaStar domain)
      - loglam : 1-D numpy array  (log10 wavelength)
      - is_star: boolean (True if CLASS == 'STAR')
    """
    if not os.path.exists(fits_path):
        raise FileNotFoundError(f"File not found: {fits_path}")

    with fits.open(fits_path, memmap=True) as hdul:
        # HDU 1 = COADD  — one row per pixel
        coadd  = hdul[1].data
        flux   = np.array(coadd['flux'],   dtype=float)
        loglam = np.array(coadd['loglam'], dtype=float)
        ivar   = np.array(coadd['ivar'],   dtype=float)   # inverse variance mask

        # HDU 2 = SPALL  — one row, CLASS
        spall     = hdul[2].data
        obj_class = str(spall['CLASS'][0]).strip().upper()

    # Reject non-stellar objects
    is_star = (obj_class == 'STAR')
    if not is_star:
        print(f"   [SKIP] {os.path.basename(fits_path)}: CLASS={obj_class} (not a STAR).")
        return None, None, False

    # ── CRITICAL: bring flux into the MaStar training domain ──────────────────
    # ivar를 함께 전달해 bad pixel을 MaStar MASK 방식과 동일하게 처리
    if apply_continuum_norm:
        flux = continuum_normalize(flux, ivar=ivar, window=201)

    return flux, loglam, True
