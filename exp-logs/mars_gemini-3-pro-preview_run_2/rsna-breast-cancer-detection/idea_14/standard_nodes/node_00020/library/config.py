import os
import torch


class Config:
    # =========================================================================
    # Global System Settings
    # =========================================================================
    SEED = 42
    NUM_WORKERS = 4  # Number of dataloader workers
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # =========================================================================
    # File Paths and Directories
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for caching and checkpoints
    # Specifically for Idea 14 as requested
    WORK_DIR = "./working/idea_14"

    # Submission output
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Metadata Files
    TRAIN_META = os.path.join(METADATA_DIR, "train.csv")
    VAL_META = os.path.join(METADATA_DIR, "val.csv")
    TEST_META = os.path.join(METADATA_DIR, "test.csv")

    # =========================================================================
    # Model Hyperparameters (DS-GEHN)
    # =========================================================================
    MODEL_NAME = "efficientnet_v2_s"  # Torchvision backbone
    PRETRAINED = True

    # Input Engineering
    # Input is 1 channel (DICOM), expanded to 3 internally via GPU layer
    IN_CHANNELS = 1

    # Resolution: 640x640 as per strategy
    IMG_SIZE = (640, 640)

    # Tabular Features
    # Features to use for the hybrid branch
    TABULAR_COLS = ["age", "implant", "site_id", "laterality", "view", "machine_id"]
    TABULAR_EMBED_DIM = 64  # Dimension after processing tabular data

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    # Full dataset utilization is prioritized
    USE_FULL_DATA = True

    # Batch Size: A100 40GB can handle larger batches, but 640x640 is large.
    # Setting to 24 to be safe and efficient with gradients.
    BATCH_SIZE = 24

    # Epochs: 10 is usually sufficient with OneCycleLR
    EPOCHS = 10

    # Optimizer settings
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 1e-2

    # Loss Function Settings
    # High positive weight to handle class imbalance (approx 1:50)
    POS_WEIGHT = 20.0

    # Deep Supervision
    # Weighted sum: L_total = L_final + 0.4 * L_aux
    AUX_LOSS_WEIGHT = 0.4

    # =========================================================================
    # Setup Logic
    # =========================================================================
    @classmethod
    def setup(cls):
        """Creates necessary directories."""
        os.makedirs(cls.WORK_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Execute setup immediately when module is imported to ensure paths exist
Config.setup()
