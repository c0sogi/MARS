import os
import torch


class Config:
    """
    Global configuration for the CA-ResDnCNN Denoising Pipeline.
    """

    # -------------------------------------------------------------------------
    # General Settings
    # -------------------------------------------------------------------------
    PROJECT_NAME = "CA-ResDnCNN_Denoising"
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # Utilizing 12 vCPUs efficiently for data loading
    NUM_WORKERS = 12

    # -------------------------------------------------------------------------
    # File Paths & Directories
    # -------------------------------------------------------------------------
    # Read-only Input Directories
    INPUT_DIR = "./input"
    TRAIN_IMG_DIR = os.path.join(INPUT_DIR, "train")
    TRAIN_CLEANED_DIR = os.path.join(INPUT_DIR, "train_cleaned")
    TEST_IMG_DIR = os.path.join(INPUT_DIR, "test")

    # Metadata Paths (Pre-generated)
    METADATA_DIR = "./metadata"
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Working Directory for Checkpoints and Cache
    # Using idea_11 as the designated workspace for this run
    WORKING_DIR = "./working/idea_11"

    # Submission Output
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Model Checkpoint Path
    CHECKPOINT_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # -------------------------------------------------------------------------
    # Data Processing & Curriculum
    # -------------------------------------------------------------------------
    PATCH_SIZE = 50

    # Two-Stage Curriculum Strides
    # Stage 1: Convergence Phase (Lower density, faster epochs)
    STRIDE_STAGE_1 = 20
    # Stage 2: Refinement Phase (High density, maximum data saturation)
    STRIDE_STAGE_2 = 10

    # Cache Filenames for deterministic loading
    CACHE_FILE_STAGE_1 = os.path.join(WORKING_DIR, "train_patches_s1.npy")
    CACHE_TARGETS_STAGE_1 = os.path.join(WORKING_DIR, "train_targets_s1.npy")
    CACHE_FILE_STAGE_2 = os.path.join(WORKING_DIR, "train_patches_s2.npy")
    CACHE_TARGETS_STAGE_2 = os.path.join(WORKING_DIR, "train_targets_s2.npy")

    # Validation Cache (Fixed stride, e.g., 20 or 50, to save time during val)
    VAL_STRIDE = 20
    CACHE_FILE_VAL = os.path.join(WORKING_DIR, "val_patches.npy")
    CACHE_TARGETS_VAL = os.path.join(WORKING_DIR, "val_targets.npy")

    # -------------------------------------------------------------------------
    # Model Architecture: CA-ResDnCNN
    # -------------------------------------------------------------------------
    IN_CHANNELS = 1
    OUT_CHANNELS = 1  # Network predicts the noise residual
    NUM_FEATURES = 64  # Number of feature maps in hidden layers
    NUM_BLOCKS = 20  # Depth of the residual stack
    KERNEL_SIZE = 3  # 3x3 Convolutions (Dense, no dilation)
    PADDING = 1

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    # A100 40GB allows for large batches even with float32
    BATCH_SIZE = 128

    # Optimization
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    ETA_MIN = 1e-7  # Minimum LR for Cosine Annealing

    # Training Duration limits (Soft limits, runtime takes precedence)
    MAX_EPOCHS_STAGE_1 = 100
    MAX_EPOCHS_STAGE_2 = 100

    # Early Stopping
    PATIENCE = 15

    # -------------------------------------------------------------------------
    # Runtime Management
    # -------------------------------------------------------------------------
    # Total budget is 24 hours. We reserve 30 mins for inference and overhead.
    MAX_RUNTIME_HOURS = 23.5
    MAX_RUNTIME_SECONDS = MAX_RUNTIME_HOURS * 3600

    # -------------------------------------------------------------------------
    # Inference Strategy
    # -------------------------------------------------------------------------
    # Geometric Self-Ensemble (8 variants: original + flips/rotations)
    TTA_ENABLED = True

    @classmethod
    def setup(cls):
        """
        Create necessary directories for output and caching.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Initialize directories on module import
Config.setup()
