import os
import torch


class Config:
    # ==========================================
    # Global System Settings
    # ==========================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    # ==========================================
    # File Paths & Directories
    # ==========================================
    # Input Directories
    INPUT_DIR = "./input"
    TRAIN_DIR = os.path.join(INPUT_DIR, "train")
    TEST_DIR = os.path.join(INPUT_DIR, "test")

    # Metadata Paths (Pre-generated)
    METADATA_DIR = "./metadata"
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test.parquet")

    # Working Directory for Caching & Outputs
    # Specific to this experiment idea
    WORKING_DIR = "./working/idea_17"
    MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # Submission
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Data Processing Hyperparameters
    # ==========================================
    IMG_SIZE = 256

    # Volumetric Settings
    TOTAL_SLICES = 32  # Total slices sampled from the volume (High Density)
    SLICES_PER_VIEW = 16  # Slices fed to the model in a single pass (Stability)
    NUM_MODALITIES = 4  # FLAIR, T1w, T1wCE, T2w

    # Input Channels = Slices per View * Modalities
    # 16 * 4 = 64 channels
    IN_CHANS = SLICES_PER_VIEW * NUM_MODALITIES

    # ==========================================
    # Model Architecture
    # ==========================================
    BACKBONE = "efficientnet_b0"
    DROP_PATH_RATE = 0.2  # Stochastic Depth regularization
    NUM_CLASSES = 1

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 8  # Conservative batch size for 64-channel 256x256 input
    NUM_EPOCHS = 15  # Standard convergence timeframe
    LEARNING_RATE = 1e-4  # Adam LR
    WEIGHT_DECAY = 0.0  # Explicitly set to 0.0 as per instructions
    PATIENCE = 5  # Early stopping patience

    @classmethod
    def setup(cls):
        """
        Ensures necessary directories exist.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Execute setup immediately upon import to guarantee directory existence
Config.setup()
