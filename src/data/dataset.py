import torch
from torch.utils.data import Dataset
import numpy as np
import os


class StellarHybridDataset(Dataset):
    """
    훈련/검증 분리 후 정규화 (leakage 방지).

    처리 순서:
      1. [:n] 슬라이싱 (세 파일 길이 맞추기)
      2. indices 적용  <- engine이 valid 필터 전 원본 위치로 넘김
      3. -999 필터     <- indices 적용 후 남은 invalid 제거
      4. 정규화

    fit_stats=True  -> train split에서만 mean/std 계산 후 저장
    fit_stats=False -> 전달받은 통계로 정규화 (val/test에서 사용)

    augment=True (훈련 시에만):
      배치마다 3가지 물리적으로 타당한 변환을 랜덤 적용해
      도메인 과적합을 방지하고 cross-domain 강건성을 향상시킵니다.
        1. Gaussian noise      sigma ~ Uniform(0, 0.05)
        2. Continuum tilt      저차 다항식 × flux (기기별 색 응답 차이 모사)
        3. Radial velocity shift  +-2픽셀 이내 정수 shift (RV 불확실도 모사)
    """

    def __init__(self,
                 flux_path,
                 feature_path,
                 label_path,
                 indices=None,
                 fit_stats=True,
                 feature_stats=None,
                 label_stats=None,
                 stats_save_dir=None,
                 augment=False):

        self.augment = augment

        raw_fluxes   = np.load(flux_path)
        raw_features = np.load(feature_path)
        raw_labels   = np.load(label_path)

        # 1. 샘플 수 맞추기
        n = min(raw_fluxes.shape[0], raw_features.shape[0], raw_labels.shape[0])
        raw_fluxes   = raw_fluxes[:n]
        raw_features = raw_features[:n]
        raw_labels   = raw_labels[:n]

        # 2. indices 적용
        if indices is not None:
            raw_fluxes   = raw_fluxes[indices]
            raw_features = raw_features[indices]
            raw_labels   = raw_labels[indices]

        # 3. -999 필터
        valid = (raw_labels[:, 0] > -900) & \
                (raw_labels[:, 1] > -900) & \
                (raw_labels[:, 2] > -900)
        raw_fluxes   = raw_fluxes[valid]
        raw_features = raw_features[valid]
        raw_labels   = raw_labels[valid]

        self.final_samples = len(raw_labels)

        # 4. flux: per-spectrum z-score
        flux_mean = np.mean(raw_fluxes, axis=1, keepdims=True)
        flux_std  = np.std(raw_fluxes,  axis=1, keepdims=True) + 1e-8
        norm_flux = np.clip((raw_fluxes - flux_mean) / flux_std, -3.0, 3.0)

        # 5. feature / label 정규화
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
                print(f"   > Normalization stats saved -> {stats_save_dir}")
        else:
            assert feature_stats is not None and label_stats is not None, \
                "fit_stats=False일 때 feature_stats, label_stats를 전달해야 합니다."
            self.feature_mean, self.feature_std = feature_stats
            self.label_mean,   self.label_std   = label_stats

        norm_features = (raw_features - self.feature_mean) / (self.feature_std + 1e-8)
        norm_labels   = (raw_labels   - self.label_mean)   / (self.label_std   + 1e-8)

        print(f"   [Dataset] {self.final_samples} samples | "
              f"flux [{norm_flux.min():.2f}, {norm_flux.max():.2f}] | "
              f"augment={augment} | "
              f"T_eff mean={self.label_mean[0]:.0f}K  "
              f"logg mean={self.label_mean[1]:.2f}")

        # float32 tensor로 저장 (augment는 __getitem__에서 실시간 적용)
        self.fluxes   = torch.from_numpy(norm_flux).float()       # (N, L) — unsqueeze는 __getitem__에서
        self.features = torch.from_numpy(norm_features).float()
        self.labels   = torch.from_numpy(norm_labels).float()
        self.n_pixels = norm_flux.shape[1]

    def _augment_flux(self, flux_1d):
        """
        flux_1d: torch.Tensor shape (L,)
        세 가지 물리적 augmentation을 랜덤 적용 후 클리핑.
        """
        # 1. Gaussian noise: sigma ~ U(0, 0.05)
        sigma = torch.empty(1).uniform_(0.0, 0.05).item()
        flux_1d = flux_1d + torch.randn_like(flux_1d) * sigma

        # 2. Continuum tilt: 저차 다항식 (선형 + 2차) 곱하기
        #    실제 기기별 색 응답 차이 / 기울어진 sky subtraction 모사
        x = torch.linspace(-1.0, 1.0, self.n_pixels)
        a = torch.empty(1).uniform_(-0.05, 0.05).item()   # 선형 기울기
        b = torch.empty(1).uniform_(-0.02, 0.02).item()   # 2차 곡률
        tilt = 1.0 + a * x + b * (x ** 2)
        flux_1d = flux_1d * tilt

        # 3. Radial velocity shift: +-2픽셀 정수 shift
        shift = int(torch.randint(-2, 3, (1,)).item())
        if shift != 0:
            flux_1d = torch.roll(flux_1d, shift)
            # 롤로 감긴 끝단 픽셀을 경계값으로 채워 artifact 방지
            if shift > 0:
                flux_1d[:shift] = flux_1d[shift]
            else:
                flux_1d[shift:] = flux_1d[shift - 1]

        return torch.clamp(flux_1d, -3.0, 3.0)

    def __len__(self):
        return self.final_samples

    def __getitem__(self, idx):
        flux = self.fluxes[idx]                # shape (L,)
        if self.augment:
            flux = self._augment_flux(flux.clone())
        return flux.unsqueeze(0), self.features[idx], self.labels[idx]
