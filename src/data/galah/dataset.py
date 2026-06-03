import torch
from torch.utils.data import Dataset
import numpy as np
import os


class StellarHybridDataset(Dataset):
    """
    GALAH 4-arm dataset with split-then-normalize and per-arm physical augmentation.
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

        # 3. -999 필터 (TEFF, LOGG, FEH)
        valid = (raw_labels[:, 0] > -900) & \
                (raw_labels[:, 1] > -900) & \
                (raw_labels[:, 2] > -900)
        raw_fluxes   = raw_fluxes[valid]
        raw_features = raw_features[valid]
        raw_labels   = raw_labels[valid]

        self.final_samples = len(raw_labels)
        self.num_arms      = raw_fluxes.shape[1]  # 4
        self.n_pixels      = raw_fluxes.shape[2]  # 4000

        # 4. flux: per-spectrum, per-arm z-score (val/augment=False 전용)
        flux_mean = np.mean(raw_fluxes, axis=2, keepdims=True)
        flux_std  = np.std(raw_fluxes,  axis=2, keepdims=True) + 1e-8
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
            assert feature_stats is not None and label_stats is not None
            self.feature_mean, self.feature_std = feature_stats
            self.label_mean,   self.label_std   = label_stats

        norm_features = (raw_features - self.feature_mean) / (self.feature_std + 1e-8)
        norm_labels   = (raw_labels   - self.label_mean)   / (self.label_std   + 1e-8)

        print(f"   [GALAH Dataset] {self.final_samples} samples | "
              f"augment={augment} | "
              f"T_eff mean={self.label_mean[0]:.0f}K  "
              f"logg mean={self.label_mean[1]:.2f}")

        # 정규화된 flux (val + augment=False에서 사용)
        self.fluxes   = torch.from_numpy(norm_flux).float()
        self.features = torch.from_numpy(norm_features).float()
        self.labels   = torch.from_numpy(norm_labels).float()

        # raw flux: augment=True일 때 tilt 적용 후 z-score
        if augment:
            self.raw_fluxes_for_aug = torch.from_numpy(raw_fluxes).float()

    def _augment_flux(self, raw_arms):
        """
        raw_arms: torch.Tensor shape (num_arms, n_pixels)
        각 arm별로 물리적 데이터 증강을 개별 수행.
        """
        augmented_arms = []
        for i in range(self.num_arms):
            raw_1d = raw_arms[i]
            # 1. Continuum tilt: z-score 전 적용
            x    = torch.linspace(-1.0, 1.0, self.n_pixels)
            a    = torch.empty(1).uniform_(-0.02, 0.02).item()
            b    = torch.empty(1).uniform_(-0.01, 0.01).item()
            flux = raw_1d * (1.0 + a * x + b * (x ** 2))

            # 2. Z-score normalize
            f_mean = flux.mean()
            f_std  = flux.std() + 1e-8
            flux   = torch.clamp((flux - f_mean) / f_std, -3.0, 3.0)

            # 3. Gaussian noise
            sigma = torch.empty(1).uniform_(0.0, 0.05).item()
            flux  = flux + torch.randn_like(flux) * sigma

            # 4. RV shift
            shift = int(torch.randint(-2, 3, (1,)).item())
            if shift != 0:
                flux = torch.roll(flux, shift)
                if shift > 0: flux[:shift]  = flux[shift]
                else:         flux[shift:]  = flux[shift - 1]

            flux = torch.clamp(flux, -3.0, 3.0)
            augmented_arms.append(flux)

        return torch.stack(augmented_arms, dim=0)

    def __len__(self):
        return self.final_samples

    def __getitem__(self, idx):
        if self.augment:
            flux = self._augment_flux(self.raw_fluxes_for_aug[idx].clone())
            feat = self.features[idx] + torch.randn_like(self.features[idx]) * 0.05
        else:
            flux = self.fluxes[idx]
            feat = self.features[idx]
        return flux, feat, self.labels[idx]
