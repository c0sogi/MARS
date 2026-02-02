import os
import torch


class Config:
    """
    Global configuration for the Cactus Classification task.
    Implements settings for 'Custom Narrow SE-ResNet with Dual-Stream Multi-Scale Aggregation'.
    """

    # ==========================================
    # Directories and Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    # Working directory for Idea 11
    WORK_DIR = "./working/idea_11"

    # Ensure the working directory exists
    os.makedirs(WORK_DIR, exist_ok=True)

    # Metadata File Paths (Pre-generated)
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Submission Paths
    SUBMISSION_DIR = os.path.join(WORK_DIR, "submission")
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Data Configuration
    # ==========================================
    IMAGE_SIZE = (32, 32)
    NUM_CLASSES = 1

    # Normalization constants (if needed, though typically handled in transforms)
    # Using standard ImageNet means/stds or 0.5/0.5 is common.
    # Since we use a custom narrow net from scratch, 0.5/0.5 or simple [0,1] scaling is sufficient.

    # ==========================================
    # Model Architecture: Narrow SE-ResNet
    # ==========================================
    # Backbone channel configuration for 3 stages
    BACKBONE_CHANNELS = [16, 32, 64]

    # Use Squeeze-and-Excitation blocks
    USE_SE_BLOCKS = True

    # Head Configuration: Dual-Stream (GAP + GMP) Aggregation
    USE_DUAL_STREAM_HEAD = True

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    # Homogeneous Seed Averaging
    NUM_SEEDS = 5
    SEEDS = [0, 1, 2, 3, 4]

    EPOCHS = 20
    BATCH_SIZE = 128

    # Optimization
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2  # AdamW standard

    # Scheduler (Cosine Annealing)
    T_MAX = EPOCHS
    ETA_MIN = 1e-6

    # ==========================================
    # Compute Configuration
    # ==========================================
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 12 vCPUs available; 4 workers is usually a sweet spot for low-overhead dataloading
    NUM_WORKERS = 4

    # ==========================================
    # Debugging and Development
    # ==========================================
    # Flags to control dataset size for rapid prototyping
    DEBUG = False
    DEBUG_SUBSET_SIZE = 100  # Number of samples to use if DEBUG is True
