import numpy as np
import astropy.io.fits as fits
from scipy.ndimage import median_filter
import os
from multiprocessing import Pool, cpu_count

try:
    from src.utils.mastar.config import CPU_WORKERS_PREPROCESS
except ImportError:
    try:
        from utils.config import CPU_WORKERS_PREPROCESS
    except ImportError:
        CPU_WORKERS_PREPROCESS = max(1, cpu_count() - 1)


def _normalize_single_spectrum(args):
    """
    단일 스펙트럼에 전처리 파이프라인 적용.
    multiprocessing.Pool에서 호출 가능하도록 args를 tuple로 받음.
    """
    raw_flux, pixel_mask = args
    raw_flux = raw_flux.astype(float)
    safe_flux = np.copy(raw_flux)
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


def _process_fits(file_path, label):
    """
    단일 FITS 파일(goodspec 또는 combspec)을 읽고 병렬 전처리합니다.
    반환값: (flux_list, star_id_list, standard_wave)
    """
    print(f"[{label}] Reading: {os.path.basename(file_path)}")
    hdul = fits.open(file_path)
    table = hdul[1].data

    standard_wave = table["WAVE"][0]
    total_rows = len(table)
    print(f"   > {total_rows} spectra found. "
          f"Preprocessing with {CPU_WORKERS_PREPROCESS} workers "
          f"(of {cpu_count()} total cores)...")

    args = [(table["FLUX"][i].copy(), table["MASK"][i].copy())
            for i in range(total_rows)]
    id_list = [table["MANGAID"][i] for i in range(total_rows)]
    hdul.close()

    with Pool(processes=CPU_WORKERS_PREPROCESS) as pool:
        flux_list = list(pool.imap(_normalize_single_spectrum, args,
                                   chunksize=max(1, total_rows // (CPU_WORKERS_PREPROCESS * 4))))

    print(f"   > Done: {total_rows} spectra processed.")
    return flux_list, id_list, standard_wave


def run_mastar_preprocessing_pipeline(goodspec_path, combspec_path=None):
    """
    goodspec + (선택) combspec 두 파일을 합산해 전처리합니다.
    combspec의 MANGAID 중 goodspec에 이미 존재하는 별은 제외합니다 (데이터 침복 방지).
    """
    all_flux, all_ids = [], []

    # ── goodspec (per-visit) ──────────────────────────────────────────────────
    g_flux, g_ids, standard_wave = _process_fits(goodspec_path, "goodspec")
    all_flux.extend(g_flux)
    all_ids.extend(g_ids)
    goodspec_id_set = set(str(sid).strip() for sid in g_ids)
    print(f"   > goodspec unique MANGAIDs: {len(goodspec_id_set)}")

    # ── combspec (per-star combined, 선택) ────────────────────────────────────
    if combspec_path and os.path.exists(combspec_path):
        c_flux, c_ids, c_wave = _process_fits(combspec_path, "combspec")

        # ── wave grid 일치 검증 ─────────────────────────────────────────
        if len(c_wave) == len(standard_wave) and np.allclose(c_wave, standard_wave, atol=0.1):
            print("   > combspec wave grid: MATCH ✔")
        else:
            print(f"   > combspec wave grid MISMATCH — "
                  f"goodspec {len(standard_wave)}px vs combspec {len(c_wave)}px. "
                  f"combspec spectra will be interpolated onto goodspec grid.")
            from scipy.interpolate import interp1d
            aligned = []
            for flux in c_flux:
                f_interp = interp1d(c_wave, flux, kind='linear',
                                    bounds_error=False, fill_value=1.0)
                aligned.append(f_interp(standard_wave).astype(np.float32))
            c_flux = aligned

        # ── 데이터 침복 제거: goodspec에 이미 있는 MANGAID 제외 ────────────────
        added, skipped = 0, 0
        for flux, sid in zip(c_flux, c_ids):
            if str(sid).strip() in goodspec_id_set:
                skipped += 1
            else:
                all_flux.append(flux)
                all_ids.append(sid)
                added += 1
        print(f"   > combspec: {added} 고유 별 추가, {skipped} 중복 제외 (goodspec에 이미 존재)")
    else:
        print("   > combspec not provided — goodspec only.")

    X_flux = np.array(all_flux, dtype=np.float32)
    print(f"\nTotal combined spectra: {X_flux.shape[0]} "
          f"(goodspec {len(g_flux)} + combspec unique {X_flux.shape[0] - len(g_flux)})")

    out_dir = os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")),
        "data", "mastar", "processed"
    )
    os.makedirs(out_dir, exist_ok=True)

    np.save(os.path.join(out_dir, "X_flux_clean.npy"),  X_flux)
    np.save(os.path.join(out_dir, "star_ids.npy"),       np.array(all_ids))
    np.save(os.path.join(out_dir, "standard_wave.npy"),  standard_wave)

    print(f"Saved → {out_dir}")
    print(f"Final Matrix Shape: {X_flux.shape}")
    return standard_wave, X_flux


if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    raw_dir  = os.path.join(base_dir, "data", "mastar", "raw")
    goodspec = os.path.join(raw_dir, "mastar-goodspec-v3_1_1-v1_7_7.fits")
    combspec = os.path.join(raw_dir, "mastar-combspec-v3_1_1-v1_7_7-lsfpercent60.0.fits")


    if not os.path.exists(goodspec):
        print(f"Error: goodspec not found at {goodspec}")
    else:
        run_mastar_preprocessing_pipeline(goodspec, combspec)
