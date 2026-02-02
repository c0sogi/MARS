import os
import torch


class Config:
    """
    Configuration for Idea 14: 2.5D HRNet with Physical Space Normalization.
    """

    # =========================================================================
    # Random Seed & Reproducibility
    # =========================================================================
    SEED = 42

    # =========================================================================
    # File System Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_14"

    # Metadata locations (Pre-generated)
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Output paths
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # Data Preprocessing & Physical Normalization
    # =========================================================================
    # Target physical spacing in mm (Physical Space Normalization)
    TARGET_SPACING = 1.5

    # Spatial dimensions
    # We use random crops for training to handle variable image sizes after resampling
    TRAIN_CROP_SIZE = (320, 320)  # (Height, Width)

    # Inference sliding window size
    INFERENCE_WINDOW_SIZE = (320, 320)
    INFERENCE_STRIDE = 160  # 50% overlap

    # Input channels: 3 for 2.5D (Slice i-1, Slice i, Slice i+1)
    IN_CHANNELS = 3

    # Class definitions
    CLASSES = ["large_bowel", "small_bowel", "stomach"]
    NUM_CLASSES = len(CLASSES)

    # =========================================================================
    # Model Architecture
    # =========================================================================
    ARCH = "hrnet_w32"  # High-Resolution Network
    PRETRAINED = True

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    # Batch size (A100 40GB can handle larger batches for 320x320)
    BATCH_SIZE = 32

    # Optimization
    EPOCHS = 15
    LR = 3e-4
    WEIGHT_DECAY = 1e-2

    # Scheduler (Cosine Annealing)
    T_MAX = EPOCHS
    ETA_MIN = 1e-6

    # Loss Function Weights
    # Tversky Loss parameters (Beta > Alpha to emphasize recall)
    TVERSKY_ALPHA = 0.3
    TVERSKY_BETA = 0.7
    TVERSKY_SMOOTH = 1.0

    # Combined Loss Weights
    WEIGHT_BCE = 0.5
    WEIGHT_TVERSKY = 0.5

    # =========================================================================
    # Hardware & Compute
    # =========================================================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # Use available vCPUs for data loading
    NUM_WORKERS = 12

    # =========================================================================
    # Debugging & Development
    # =========================================================================
    # If set to an integer (e.g., 1000), only use that many samples for training/val
    SAMPLE_SIZE = None
    DEBUG = False

    @classmethod
    def setup(cls):
        """
        Ensures that necessary working directories exist.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Initialize directories on import
Config.setup()
