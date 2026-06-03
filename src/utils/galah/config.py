import torch
import os
from multiprocessing import cpu_count

# ── GPU 감지 ─────────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# ── CPU 코어 감지 및 작업별 할당 ─────────────────────────────────────────────
# Apple Silicon 구조:
#   M2 Pro: 성능 코어 8 + 효율 코어 4 = 12 logical
#   M4:     성능 코어 4 + 효율 코어 6 = 10 logical
#
# 할당 전략:
#   - CPU_WORKERS_PREPROCESS: 전처리/피처 추출용 (연산 집약적, 코어 최대 활용)
#     → 전체 코어 - 1 (OS + 메인 스레드용 1개 남김)
#   - CPU_WORKERS_DATALOADER: DataLoader prefetch용 (I/O 바운드)
#     → 코어 수에 따라 적절히 제한 (과도하면 메모리 압박)
#   - BATCH_SIZE: GPU 메모리에 따라 자동 조정
_total_cores = cpu_count()

# 전처리/피처 추출: 코어 전부 사용 (OS용 1개 예비)
CPU_WORKERS_PREPROCESS = max(1, _total_cores - 1)

# DataLoader: 코어가 많아도 8개로 캡 (그 이상은 오버헤드)
CPU_WORKERS_DATALOADER = min(8, max(2, _total_cores // 2))

# 배치 크기: GPU 메모리 기반 자동 설정
def _detect_batch_size():
    if DEVICE.type == "mps":
        # Apple Unified Memory — psutil로 실제 할당 메모리 추정
        try:
            import psutil
            total_gb = psutil.virtual_memory().total / (1024 ** 3)
            if total_gb >= 30:    return 256   # 32GB (M2 Pro)
            elif total_gb >= 20:  return 192   # 24GB
            else:                 return 128   # 16GB (M4)
        except ImportError:
            return 128
    elif DEVICE.type == "cuda":
        try:
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            if vram_gb >= 16:   return 512
            elif vram_gb >= 8:  return 256
            else:               return 128
        except Exception:
            return 128
    return 64  # CPU fallback

BATCH_SIZE    = _detect_batch_size()
EPOCHS        = 150
LEARNING_RATE = 0.0002

# GALAH Specific Constants
NUM_ARMS = 4
OUTPUT_DIM_PER_ARM = 1600
FEATURE_DIM = 45
LABEL_DIM = 3


def print_config():
    print("[GALAH Config] Hardware and Training Settings:")
    print(f"   > Compute Device       : {DEVICE}")
    print(f"   > Total CPU Cores      : {_total_cores}")
    print(f"   > Preprocess Workers   : {CPU_WORKERS_PREPROCESS}")
    print(f"   > DataLoader Workers   : {CPU_WORKERS_DATALOADER}")
    print(f"   > Batch Size           : {BATCH_SIZE}  (auto-detected)")
    print(f"   > Total Epochs         : {EPOCHS}")
    print(f"   > Learning Rate        : {LEARNING_RATE}")
    print(f"   > Number of Arms (CCD) : {NUM_ARMS}")
    print(f"   > Output Dim Per Arm   : {OUTPUT_DIM_PER_ARM}")
    print(f"   > Feature Dimension    : {FEATURE_DIM}")
    print(f"   > Label Dimension      : {LABEL_DIM}\n")

