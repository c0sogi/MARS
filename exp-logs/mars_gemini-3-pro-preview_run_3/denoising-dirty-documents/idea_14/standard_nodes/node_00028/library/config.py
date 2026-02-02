import os
import torch


class Config:
    """
    Configuration for the Ensemble of Curriculum-Trained Zero-Initialized
    Deep Residual Networks (ECT-ZI-ResDnCNN).
    """

    # =========================================================================
    # File Paths & Directories
    # =========================================================================
    INPUT_DIR = "./input"
    TRAIN_DIR = os.path.join(INPUT_DIR, "train")
    TRAIN_CLEANED_DIR = os.path.join(INPUT_DIR, "train_cleaned")
    TEST_DIR = os.path.join(INPUT_DIR, "test")

    METADATA_DIR = "./metadata"
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Working directory for caching intermediate files (patches) and model checkpoints
    WORKING_DIR = "./working/idea_14"

    # Submission output
    SUBMISSION_DIR = "./submission"
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # Data Pipeline & Curriculum Parameters
    # =========================================================================
    PATCH_SIZE = 50

    # Phase 1: Sparse extraction for fast initial convergence (High throughput)
    STRIDE_SPARSE = 20

    # Phase 2: Dense extraction for refinement (High capacity saturation)
    STRIDE_DENSE = 10

    # Augmentation
    USE_AUGMENTATION = True  # Random flips and 90-degree rotations

    # Normalization
    PIXEL_MIN = 0.0
    PIXEL_MAX = 1.0

    # =========================================================================
    # Model Architecture (ZI-ResDnCNN)
    # =========================================================================
    IN_CHANNELS = 1
    OUT_CHANNELS = 1  # Network predicts the noise residual
    NUM_FEATURES = 64
    NUM_RES_BLOCKS = 20  # Deep stack of linear residual blocks

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    ENSEMBLE_SIZE = 5
    SEED = 42

    BATCH_SIZE = 128
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4

    # Curriculum Schedule
    # Stage 1: Train on sparse data
    STAGE_1_EPOCHS = 30

    # Stage 2: Fine-tune on dense data
    STAGE_2_EPOCHS = 50

    # Optimization
    PATIENCE = 10  # Early stopping patience

    # =========================================================================
    # Hardware & Runtime
    # =========================================================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # Use available vCPUs for data loading
    NUM_WORKERS = 12

    # =========================================================================
    # Debugging / Development
    # =========================================================================
    # Set DEBUG to True to run on a tiny subset of data to verify pipeline integrity
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 500  # Max patches to use if DEBUG is True

    @classmethod
    def setup(cls):
        """Creates necessary working directories."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

    @classmethod
    def get_ensemble_seeds(cls):
        """Returns a list of seeds for the ensemble members."""
        return [cls.SEED + i for i in range(cls.ENSEMBLE_SIZE)]


# Execute setup on import to ensure environment is ready
Config.setup()
