import os
import torch
import numpy as np
import random


class Config:
    # -------------------------------------------------------------------------
    # 1. General Configuration
    # -------------------------------------------------------------------------
    IDEA_NAME = "idea_7"
    SEED = 42

    # Working directory for this specific experiment
    WORKING_DIR = os.path.join("./working", IDEA_NAME)
    os.makedirs(WORKING_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # 2. Data Paths
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Metadata files (Pre-generated)
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Submission output
    SUBMISSION_DIR = "./submission"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # -------------------------------------------------------------------------
    # 3. Data Processing Parameters
    # -------------------------------------------------------------------------
    # Original image dimensions
    ORIG_HEIGHT = 101
    ORIG_WIDTH = 101

    # Model input dimensions (Padded with reflection)
    # 128 is divisible by 32 (2^5), suitable for U-Net depth 5
    INPUT_HEIGHT = 128
    INPUT_WIDTH = 128

    # Input Channels: 1 Grayscale Image + 1 Depth Channel
    INPUT_CHANNELS = 2

    # Debugging flags to control dataset size
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 50  # Only use 50 samples if DEBUG is True

    # -------------------------------------------------------------------------
    # 4. Model Architecture Hyperparameters
    # -------------------------------------------------------------------------
    # Backbone
    ENCODER_FILTERS = 64  # Starting filters for the custom residual encoder

    # Decoder & Head
    USE_HYPERCOLUMNS = True  # Aggregate features from all decoder levels
    USE_SCSE = True  # Spatial and Channel Squeeze & Excitation
    DEEP_SUPERVISION = True  # Auxiliary losses at intermediate levels

    # -------------------------------------------------------------------------
    # 5. Training Hyperparameters
    # -------------------------------------------------------------------------
    BATCH_SIZE = 64
    NUM_WORKERS = 4  # 12 vCPUs available, 4 is a safe balance

    # Optimization
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # Cyclic Cosine Annealing Schedule
    # Total Epochs = 150 (3 cycles * 50 epochs)
    NUM_EPOCHS = 150
    CYCLE_LEN = 50

    # Two-Stage Optimization Curriculum
    # Phase 1: BCE + Dice (Robust convergence) -> Epochs 0-99 (Cycles 1 & 2)
    # Phase 2: Lovasz-Hinge (Metric fine-tuning) -> Epochs 100-149 (Cycle 3)
    PHASE_1_EPOCHS = 100

    # -------------------------------------------------------------------------
    # 6. Compute & Checkpointing
    # -------------------------------------------------------------------------
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Paths for snapshot ensembling
    CYCLE_2_BEST_MODEL = os.path.join(CHECKPOINT_DIR, "best_cycle_2.pth")
    CYCLE_3_BEST_MODEL = os.path.join(CHECKPOINT_DIR, "best_cycle_3.pth")

    @staticmethod
    def set_seed(seed=42):
        """Sets the random seed for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
