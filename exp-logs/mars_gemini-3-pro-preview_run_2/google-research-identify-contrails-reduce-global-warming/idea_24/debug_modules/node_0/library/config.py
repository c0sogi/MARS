import os
import torch
import random
import numpy as np


class Config:
    # ==========================================
    # Project & Paths
    # ==========================================
    PROJECT_NAME = "idea_24"
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Metadata files (Pre-generated)
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "validation_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Output Directories
    WORKING_DIR = os.path.join("./working", PROJECT_NAME)
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Model Artifacts
    BEST_MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # Ensure directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # ==========================================
    # Data Hyperparameters
    # ==========================================
    SEED = 42
    IMG_SIZE = 256
    NUM_WORKERS = 4  # Optimized for 12 vCPUs

    # Temporal Sequence
    N_TIMES_BEFORE = 4
    N_TIMES_AFTER = 3
    # Total frames in sequence = 4 + 3 + 1 = 8

    # Satellite Bands (GOES-16 ABI)
    BAND_IDS = [8, 9, 10, 11, 12, 13, 14, 15, 16]

    # Input Engineering
    # Channels 1-3: Ash Color Composite
    # Channels 4-6: Temporal Differences (Band 11, 14, 15)
    IN_CHANNELS = 6

    # ==========================================
    # Model Architecture
    # ==========================================
    BACKBONE = "convnext_tiny"
    ENCODER_PRETRAINED = True

    # Progressive Kernel Expansion for Decoder
    # Logic: Stride 16 (7x7) -> Stride 8 (9x9) -> Stride 4 (11x11) -> Stride 2 (13x13) -> Stride 1 (15x15)
    # This maintains a consistent physical receptive field during upsampling.
    DECODER_KERNEL_SIZES = [7, 9, 11, 13, 15]

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 32
    EPOCHS = 30
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2  # Standard for AdamW

    # Scheduler (Cosine Annealing)
    T_MAX = EPOCHS
    ETA_MIN = 1e-6

    # Early Stopping
    PATIENCE = 6

    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ==========================================
    # Inference & Post-Processing
    # ==========================================
    THRESHOLD = 0.5
    USE_TTA = True  # Test Time Augmentation (Flip/Rotate)

    @staticmethod
    def set_seed(seed=42):
        """Sets the seed for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        # Ensure deterministic behavior for cuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
