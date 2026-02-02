import os
import torch
import random
import numpy as np


class Config:
    """
    Global configuration for the Salt Segmentation Task.
    Implements the 'High-Capacity Deep Residual U-Net with Stochastic Depth' strategy.
    """

    # -------------------------------------------------------------------------
    # 1. General System Settings
    # -------------------------------------------------------------------------
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 4  # Number of data loading workers

    # -------------------------------------------------------------------------
    # 2. File Paths and Directories
    # -------------------------------------------------------------------------
    # Input Data (Read-Only)
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
    DEPTHS_CSV = os.path.join(INPUT_DIR, "depths.csv")

    # Working Directory (Write Access)
    # Using 'idea_12' as the experiment identifier
    WORKING_DIR = "./working/idea_12"

    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = os.path.join(WORKING_DIR, "submission")
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    PREDICTIONS_DIR = os.path.join(WORKING_DIR, "predictions")

    # Ensure working directories exist
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # 3. Data Parameters
    # -------------------------------------------------------------------------
    ORIG_IMG_SIZE = 101
    IMG_SIZE = 128  # Padded size for U-Net (divisible by 32)
    CHANNELS = 1  # Grayscale seismic image

    # Normalization (calculated from dataset analysis)
    MEAN = 0.5
    STD = 0.5

    # -------------------------------------------------------------------------
    # 4. Model Architecture Hyperparameters
    # -------------------------------------------------------------------------
    # High-Capacity Encoder
    ENCODER_FILTERS = [64, 128, 256, 512, 1024]

    # Regularization
    DROP_PATH_RATE = 0.2  # Stochastic Depth probability
    DROPOUT_RATE = 0.0  # Standard dropout (usually 0 if using DropPath)

    # Decoder & Heads
    USE_SCSE = True  # Concurrent Spatial and Channel Squeeze & Excitation
    DEEP_SUPERVISION = True  # Aux heads at 32, 64, 128

    # -------------------------------------------------------------------------
    # 5. Training Hyperparameters
    # -------------------------------------------------------------------------
    BATCH_SIZE = 32

    # Cyclic Schedule: 3 cycles of 50 epochs = 150 total epochs
    CYCLES = 3
    EPOCHS_PER_CYCLE = 50
    EPOCHS = CYCLES * EPOCHS_PER_CYCLE

    # Optimizer
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2  # AdamW standard

    # Loss Weights (Consistent Compound Loss)
    # L = L_BCE + L_Dice_Sample + 0.1 * L_Lovasz
    WEIGHT_BCE = 1.0
    WEIGHT_DICE = 1.0
    WEIGHT_LOVASZ = 0.1

    # -------------------------------------------------------------------------
    # 6. Evaluation & Inference
    # -------------------------------------------------------------------------
    # Metric IoU Thresholds
    IOU_THRESHOLDS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    # Snapshot Ensembling
    # We will ensemble the best models from the last N cycles
    ENSEMBLE_CYCLES = [2, 3]  # 1-based index of cycles to use

    # Test Time Augmentation
    TTA_FLIP = True  # Horizontal flip TTA

    # -------------------------------------------------------------------------
    # 7. Debugging / Development
    # -------------------------------------------------------------------------
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 100  # Number of samples to use in debug mode

    @staticmethod
    def setup_reproducibility(seed=SEED):
        """
        Sets the seed for all random number generators to ensure reproducibility.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
