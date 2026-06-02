import torch
from torch.utils.data import Dataset
import numpy as np
import os


class StellarHybridDataset(Dataset):
    """
    훈련/검증 분리 후 정규화 (leakage 방지).

    처리 순서:
      1. [:n] 슬라이싱 (세 파일 길이 맞추기)
      2. indices 적용  ← engine이 valid 필터 전 원본 위치로 넘김
      3. -999 필터     ← indices 적용 후 남은 invalid 제거
      4. 정규화

    fit_stats=True  → train split에서만 mean/std 계산 후 저장
    fit_stats=False → 전달받은 통계로 정규화 (val/test에서 사용)
    """

    def __init__(self,
                 flux_path,
                 feature_path,
                 label_path,
                 indices=None,
                 fit_stats=True,
                 feature_stats=None,
                 label_stats=None,
                 stats_save_dir=None):

        raw_fluxes   = np.load(flux_path)
        raw_features = np.load(feature_path)
        raw_labels   = np.load(label_path)

        # ── 1. 샘플 수 맞추기 ─────────────────────────────────────────────────
        n = min(raw_fluxes.shape[0], raw_features.shape[0], raw_labels.shape[0])
        raw_fluxes   = raw_fluxes[:n]
        raw_features = raw_features[:n]
        raw_labels   = raw_labels[:n]

        # ── 2. indices 적용 (engine이 원본 위치 기준으로 넘긴 split 인덱스) ────
        if indices is not None:
            raw_fluxes   = raw_fluxes[indices]
            raw_features = raw_features[indices]
            raw_labels   = raw_labels[indices]

        # ── 3. -999 필터 (indices 적용 이후) ──────────────────────────────────
        valid = (raw_labels[:, 0] > -900) & \
                (raw_labels[:, 1] > -900) & \
                (raw_labels[:, 2] > -900)
        raw_fluxes   = raw_fluxes[valid]
        raw_features = raw_features[valid]
        raw_labels   = raw_labels[valid]

        self.final_samples = len(raw_labels)

        # ── 4. flux: per-spectrum z-score (누출 없음 — 각 스펙트럼 독립) ───────
        flux_mean = np.mean(raw_fluxes, axis=1, keepdims=True)
        flux_std  = np.std(raw_fluxes,  axis=1, keepdims=True) + 1e-8
        norm_flux = np.clip((raw_fluxes - flux_mean) / flux_std, -3.0, 3.0)

        # ── 5. feature / label: train에서만 fit, val에는 적용만 ───────────────
        if fit_stats:
            self.feature_mean = np.mean(raw_features, axis=0)
            self.feature_std  = np.std(raw_features,  axis=0) + 1e-8
            self.label_mean   = np.mean(raw_labels,   axis=0)
            self.label_std    = np.std(raw_labels,    axis=0) + 1e-8

            if stats_save_dir:
                os.makedirs(stats_save_dir, exist_ok=True)
                np.save(os.path.join(stats_save_dir, "feature_stats.npy"),
                        np.stack([self.feature_mean, self.feature_std]))
                np.save(os.path.join(stats_save_dir, "label_stats.npy"),
                        np.stack([self.label_mean, self.label_std]))
                print(f"   > Normalization stats saved → {stats_save_dir}")
        else:
            assert feature_stats is not None and label_stats is not None, \
                "fit_stats=False일 때 feature_stats, label_stats를 전달해야 합니다."
            self.feature_mean, self.feature_std = feature_stats
            self.label_mean,   self.label_std   = label_stats

        norm_features = (raw_features - self.feature_mean) / (self.feature_std + 1e-8)
        norm_labels   = (raw_labels   - self.label_mean)   / (self.label_std   + 1e-8)

        print(f"   [Dataset] {self.final_samples} samples | "
              f"flux [{norm_flux.min():.2f}, {norm_flux.max():.2f}] | "
              f"T_eff mean={self.label_mean[0]:.0f}K  "
              f"logg mean={self.label_mean[1]:.2f}")

        self.fluxes   = torch.from_numpy(norm_flux).float().unsqueeze(1)
        self.features = torch.from_numpy(norm_features).float()
        self.labels   = torch.from_numpy(norm_labels).float()

    def __len__(self):
        return self.final_samples

    def __getitem__(self, idx):
        return self.fluxes[idx], self.features[idx], self.labels[idx]
