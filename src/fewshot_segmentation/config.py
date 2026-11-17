"""Project-wide configuration and reproducibility settings.

Aligned with SSP paper (Fan et al., ECCV 2022) EXCEPT:
  - IMAGE_SIZE = 224 (paper: 473)
  - FEATURE_DIM = 512 (paper: 1024)
  - TRAIN_ITERATIONS = 1000 (paper: 6000)
"""
import os
import random
from pathlib import Path
import numpy as np
import torch

# ============================================================
# Reproducibility
# ============================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ============================================================
# Device
# ============================================================
if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
DEVICE = device

# ============================================================
# Paths
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ============================================================
# USER-REQUESTED VALUES (differ from paper)
# ============================================================
IMAGE_SIZE = 473                        # ← USER: 224 (paper: 473)
FEATURE_DIM = 1024                       # ← USER: 512 (paper: 1024)
TRAIN_ITERATIONS = 6000                 # ← USER: 1000 (paper: 6000)

# ============================================================
# PAPER HYPERPARAMETERS (unchanged)
# ============================================================
K_SHOT = 5
MIN_FG_PIXELS = 20
BATCH_PAIRS = 4
EPISODES_PER_TEST_FOLD = int(os.environ.get("FEWSHOT_EPISODES_PER_TEST_FOLD", "100"))

# SGD (paper Section 4)
LEARNING_RATE = 1e-3
MOMENTUM = 0.9
LR_DECAY_STEP = 2000
LR_DECAY_GAMMA = 0.1
WEIGHT_DECAY = 1e-4

# SSP-specific (paper Section 3)
TAU_FG = 0.7
TAU_BG = 0.6
ALPHA1 = 0.5
ALPHA2 = 0.5
LAMBDA_M = 1.0
LAMBDA_Q = 1.0
LAMBDA_S = 0.2

# ============================================================
# 4-fold split (PASCAL-5^i)
# ============================================================
NUM_FOLDS = 4
FOLDS = [
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10],
    [11, 12, 13, 14, 15],
    [16, 17, 18, 19, 20],
]

VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
    "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
    "train", "tvmonitor",
]
NUM_CLASSES = 20