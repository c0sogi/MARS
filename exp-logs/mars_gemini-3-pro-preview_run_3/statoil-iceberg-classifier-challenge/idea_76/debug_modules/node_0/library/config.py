import os
import torch


class Config:
    """
    Centralized configuration for the Ship vs Iceberg classification task.
    Implements settings for the ATSI-CNN architecture and training pipeline.
    """

    # ==========================================
    # Reproducibility
    # ==========================================
    SEED = 42

    # ==========================================
    # Hardware & Compute
    # ==========================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # Number of subprocesses for data loading

    # ==========================================
    # File Paths & Directories
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"

    # Cache directory for deterministic data processing (e.g., numpy arrays)
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_76")

    # Submission output
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Raw Data Files
    TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
    TEST_JSON = os.path.join(INPUT_DIR, "test.json")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Metadata Files (Generated previously)
    TRAIN_META = os.path.join(METADATA_DIR, "train.csv")
    VAL_META = os.path.join(METADATA_DIR, "val.csv")
    TEST_META = os.path.join(METADATA_DIR, "test.csv")

    # ==========================================
    # Data Parameters
    # ==========================================
    IMAGE_SIZE = 75
    NUM_CHANNELS = 3  # Band 1 (HH), Band 2 (HV), Average ((HH+HV)/2)
    NUM_CLASSES = 1  # Binary classification (0: Ship, 1: Iceberg)

    # Debugging / Quick Test
    DEBUG = False
    DEBUG_SUBSET_SIZE = 50  # Number of samples to use when DEBUG is True

    # ==========================================
    # Model Hyperparameters (ATSI-CNN)
    # ==========================================
    # Backbone: Plain CNN with 4 blocks
    # Width strategy: Expand early, then cap
    BACKBONE_CHANNELS = [64, 128, 128, 128]

    # Squeeze-and-Excitation settings
    SE_REDUCTION_RATIO = 16

    # Asymmetric Isomorphic Readout
    # Projection dimension for Stages 3 and 4
    PROJECTION_DIM = 64

    # Feature Vector Composition:
    # Stage 3 (9x9): Global Max Pooling (64) + Global MAD Pooling (64) = 128
    # Stage 4 (4x4): Global Max Pooling (64) + Global Min Pooling (64) = 128
    # Total Image Features = 256
    IMG_FEATURE_DIM = 256

    # Classification Head
    USE_INC_ANGLE = True
    # Input to classifier: Image Features (256) + Incidence Angle (1)
    CLASSIFIER_INPUT_DIM = IMG_FEATURE_DIM + 1
    CLASSIFIER_HIDDEN_DIM = 256

    # Regularization
    DROPOUT_RATE = 0.5
    LEAKY_RELU_SLOPE = 0.1  # Preserves negative values (shadows)

    # Initialization
    INIT_METHOD = "kaiming_uniform"  # PyTorch default

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    NUM_FOLDS = 5
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3  # Constant learning rate
    NUM_EPOCHS = 75
    PATIENCE = 12  # Early stopping patience
    WEIGHT_DECAY = 1e-4  # L2 Regularization

    # ==========================================
    # Utility Methods
    # ==========================================
    @classmethod
    def setup(cls):
        """
        Creates necessary directories for the pipeline.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

    @classmethod
    def print_config(cls):
        """
        Prints the current configuration.
        """
        print("=" * 30)
        print("CONFIG")
        print("=" * 30)
        print(f"Device: {cls.DEVICE}")
        print(f"Image Size: {cls.IMAGE_SIZE}x{cls.IMAGE_SIZE}")
        print(f"Channels: {cls.NUM_CHANNELS}")
        print(f"Backbone Channels: {cls.BACKBONE_CHANNELS}")
        print(f"Batch Size: {cls.BATCH_SIZE}")
        print(f"Learning Rate: {cls.LEARNING_RATE}")
        print(f"Epochs: {cls.NUM_EPOCHS}")
        print(f"Patience: {cls.PATIENCE}")
        print(f"Debug Mode: {cls.DEBUG}")
        print("=" * 30)
