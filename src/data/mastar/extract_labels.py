import numpy as np
from astropy.io import fits
import os

def main():
    print("Aligning Catalog Labels with Spectrum ID sequence...")
    print("[Label Source] MaStar Stellar Parameter VAC v2")
    print("               Teff/logg: TEFF_MED / LOGG_MED (4-method median)")
    print("               [Fe/H]   : FEH_NOAPP_MED (non-APOGEE-calibrated median)")
    print("               → SSPP 검증셋 스케일과 일치하는 광학 기반 [Fe/H] 사용\n")

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    vac_path      = os.path.join(base_dir, "data", "raw", "mastar-goodstars-v3_1_1-v1_7_7-params-v2.fits")
    star_ids_path = os.path.join(base_dir, "data", "processed", "star_ids.npy")
    save_dir      = os.path.join(base_dir, "data", "processed")

    if not os.path.exists(vac_path):
        print(f"Error: VAC file not found at '{vac_path}'")
        return
    if not os.path.exists(star_ids_path):
        print(f"Error: '{star_ids_path}' not found. Run preprocess_flux.py first.")
        return

    # ── 1. VAC에서 MANGAID → (Teff, logg, [Fe/H]) 맵 빌드 ──────────────────────
    # NGROUPS == 0 → 유효한 median을 계산할 방법이 없음 → 제외
    # FEH_NOAPP_MED: APOGEE calibration 미적용 중간값
    #   → 광학 스펙트럼 기반이라 SSPP *adop 검증 스케일과 일치
    with fits.open(vac_path) as hdul:
        vac = hdul[1].data
        col_names = vac.columns.names
        print(f"   > VAC columns: {col_names}")

        # FEH_NOAPP_MED 없으면 FEH_MED로 fallback
        feh_col = "FEH_NOAPP_MED" if "FEH_NOAPP_MED" in col_names else "FEH_MED"
        print(f"   > [Fe/H] column used: {feh_col}")

        vac_map = {}
        skipped_ngroups = 0
        for row in vac:
            manga_id = str(row["MANGAID"]).strip()
            ngroups  = int(row["NGROUPS"])
            if ngroups == 0:
                skipped_ngroups += 1
                continue
            vac_map[manga_id] = {
                "TEFF": float(row["TEFF_MED"]),
                "LOGG": float(row["LOGG_MED"]),
                "FEH":  float(row[feh_col]),
            }

    print(f"   > VAC: {len(vac_map)} valid stars loaded "
          f"({skipped_ngroups} skipped, NGROUPS==0)")

    # ── 2. flux 순서(star_ids)를 기준으로 VAC 레이블 정렬 ───────────────────────
    # goodspec은 per-visit이라 MANGAID 중복 가능.
    # VAC는 per-star이라 MANGAID 유일 — 동일 별의 여러 visit은 같은 레이블을 받음.
    star_ids = np.load(star_ids_path, allow_pickle=True)

    aligned_teff, aligned_logg, aligned_feh = [], [], []
    null_count = 0
    vac_hit    = 0

    for sid in star_ids:
        spec_id = str(sid).strip()
        if spec_id in vac_map:
            aligned_teff.append(vac_map[spec_id]["TEFF"])
            aligned_logg.append(vac_map[spec_id]["LOGG"])
            aligned_feh.append(vac_map[spec_id]["FEH"])
            vac_hit += 1
        else:
            aligned_teff.append(-999.0)
            aligned_logg.append(-999.0)
            aligned_feh.append(-999.0)
            null_count += 1

    Y_labels = np.column_stack((aligned_teff, aligned_logg, aligned_feh))

    print(f"   > Alignment complete.")
    print(f"     Total spectra      : {len(Y_labels)}")
    print(f"     VAC 매칭 성공    : {vac_hit}")
    print(f"     매칭 실패 (-999)  : {null_count}")
    print(f"     유효 비율         : {vac_hit / len(Y_labels) * 100:.1f}%")

    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "Y_labels.npy"), Y_labels)
    print(f"[Success] Shape: {Y_labels.shape} → {save_dir}/Y_labels.npy")


if __name__ == "__main__":
    main()
