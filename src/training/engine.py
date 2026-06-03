import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from ..utils.config import DEVICE, BATCH_SIZE, EPOCHS, LEARNING_RATE, \
                          CPU_WORKERS_DATALOADER, print_config
from ..data.dataset import StellarHybridDataset
from ..models.hybrid_net import StellarParameterHybridNet


def save_loss_curve(train_losses, val_losses, save_path):
    epochs = range(1, len(train_losses) + 1)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, train_losses, label="Train MSE", color="#1f77b4",
            linewidth=2.5, marker='o', markersize=3)
    ax.plot(epochs, val_losses,   label="Val MSE",   color="#ff7f0e",
            linewidth=2.5, marker='s', markersize=3)
    ax.set_title("StellarParameterHybridNet — Loss Curve",
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("MSE Loss", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=11)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def main(resume=False):
    print_config()

    base_dir     = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    flux_path    = os.path.join(base_dir, "data", "processed", "X_flux_clean.npy")
    feature_path = os.path.join(base_dir, "data", "processed", "X_features_physical.npy")
    label_path   = os.path.join(base_dir, "data", "processed", "Y_labels.npy")
    save_dir     = os.path.join(base_dir, "weights")
    stats_dir    = os.path.join(base_dir, "data", "processed")

    # ── 1. 인덱스 먼저 분리 (split-then-normalize, leakage 방지) ───────────────
    print("Building train/val split...")

    # 유효 샘플 수 파악 — 세 파일의 최솟값으로 n 결정 후 valid 필터
    _flux_n    = np.load(flux_path,    mmap_mode='r').shape[0]
    _feat_n    = np.load(feature_path, mmap_mode='r').shape[0]
    raw_labels = np.load(label_path)
    n = min(_flux_n, _feat_n, raw_labels.shape[0])
    raw_labels = raw_labels[:n]

    valid_mask    = (raw_labels[:, 0] > -900) & \
                    (raw_labels[:, 1] > -900) & \
                    (raw_labels[:, 2] > -900)
    # valid_indices는 [:n] 슬라이싱 이후의 위치 — dataset.py 내부 슬라이싱과 동일
    valid_indices = np.where(valid_mask)[0]

    rng = np.random.default_rng(42)
    rng.shuffle(valid_indices)
    train_size  = int(0.8 * len(valid_indices))
    train_idx   = valid_indices[:train_size]
    val_idx     = valid_indices[train_size:]
    print(f"   > Train: {len(train_idx)}  Val: {len(val_idx)}")

    # ── 2. train Dataset (fit_stats=True → 통계 저장, augment=True) ────────────
    print("\nBuilding train dataset and fitting normalization stats...")
    train_dataset = StellarHybridDataset(
        flux_path, feature_path, label_path,
        indices=train_idx,
        fit_stats=True,
        stats_save_dir=stats_dir,
        augment=True,          # Gaussian noise + continuum tilt + RV shift
    )

    # ── 3. val Dataset (fit_stats=False → train 통계 재사용, augment=False) ─────
    print("Building val dataset with train stats...")
    val_dataset = StellarHybridDataset(
        flux_path, feature_path, label_path,
        indices=val_idx,
        fit_stats=False,
        feature_stats=(train_dataset.feature_mean, train_dataset.feature_std),
        label_stats=(train_dataset.label_mean,   train_dataset.label_std),
        augment=False,         # val은 augmentation 없이 원본으로 평가
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=CPU_WORKERS_DATALOADER,
                              pin_memory=False, persistent_workers=True,
                              prefetch_factor=2)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=CPU_WORKERS_DATALOADER,
                              pin_memory=False, persistent_workers=True,
                              prefetch_factor=2)

    # ── 4. 모델 / 옵티마이저 ──────────────────────────────────────────────────
    model = StellarParameterHybridNet().to(DEVICE)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
    )

    os.makedirs(save_dir, exist_ok=True)
    best_ckpt_path = os.path.join(save_dir, "stellar_hybrid_model.pth")

    # ── 5. Resume (선택) ───────────────────────────────────────────────────────
    start_epoch   = 0
    best_val_loss = float('inf')
    train_losses, val_losses = [], []

    if resume and os.path.exists(best_ckpt_path):
        checkpoint = torch.load(best_ckpt_path, map_location=DEVICE)
        # checkpoint가 state_dict만 저장된 경우와 full checkpoint인 경우 모두 처리
        if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
            model.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            scheduler.load_state_dict(checkpoint['scheduler_state'])
            start_epoch   = checkpoint['epoch'] + 1
            best_val_loss = checkpoint['best_val_loss']
            train_losses  = checkpoint.get('train_losses', [])
            val_losses    = checkpoint.get('val_losses', [])
            print(f"   [Resume] Loaded full checkpoint from epoch {checkpoint['epoch']+1}")
            print(f"   [Resume] Best val loss so far: {best_val_loss:.4f}")
        else:
            # state_dict만 저장된 기존 포맷 (현재 저장 방식)
            model.load_state_dict(checkpoint)
            print(f"   [Resume] Loaded weights only (no optimizer state).")
            print(f"   [Resume] Optimizer/scheduler reset to initial state.")
    elif resume:
        print(f"   [Resume] No checkpoint found at {best_ckpt_path} — starting fresh.")

    print("\nTraining Started!")
    for epoch in range(start_epoch, EPOCHS):
        # train
        model.train()
        running_train = 0.0
        for flux, feat, labels in train_loader:
            flux, feat, labels = flux.to(DEVICE), feat.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(flux, feat), labels)
            loss.backward()
            optimizer.step()
            running_train += loss.item()

        # val
        model.eval()
        running_val = 0.0
        with torch.no_grad():
            for flux, feat, labels in val_loader:
                flux, feat, labels = flux.to(DEVICE), feat.to(DEVICE), labels.to(DEVICE)
                running_val += criterion(model(flux, feat), labels).item()
        model.train()

        avg_train = running_train / len(train_loader)
        avg_val   = running_val   / len(val_loader)
        scheduler.step(avg_val)
        current_lr = scheduler.get_last_lr()[0]

        train_losses.append(avg_train)
        val_losses.append(avg_val)

        # best checkpoint (full checkpoint 저장 — resume 가능)
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save({
                'epoch':            epoch,
                'model_state':      model.state_dict(),
                'optimizer_state':  optimizer.state_dict(),
                'scheduler_state':  scheduler.state_dict(),
                'best_val_loss':    best_val_loss,
                'train_losses':     train_losses,
                'val_losses':       val_losses,
            }, best_ckpt_path)

        if (epoch + 1) % 5 == 0 or epoch == 0 or (epoch + 1) == EPOCHS:
            print(f"Epoch [{epoch+1:3d}/{EPOCHS}] "
                  f"train={avg_train:.4f}  val={avg_val:.4f}  "
                  f"lr={current_lr:.2e}  "
                  f"Best val loss: {best_val_loss:.4f}"
                  f"{' ★' if avg_val == best_val_loss else ''}")

    print(f"\nBest val loss: {best_val_loss:.4f}")
    print(f"Best weights saved → {best_ckpt_path}")

    save_loss_curve(train_losses, val_losses, save_path=os.path.join(save_dir, "loss_curve.png"))


if __name__ == "__main__":
    main()
