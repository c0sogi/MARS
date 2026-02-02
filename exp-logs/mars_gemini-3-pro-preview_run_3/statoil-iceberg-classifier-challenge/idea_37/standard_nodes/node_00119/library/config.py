import os
import torch


class Config:
    """
    Configuration for the Dual-Polarity DropBlock CNN with Non-Bottleneck Attention (DPDB-NBA-CNN).
    """

    # =========================================================================
    # Directories
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_37"
    SUBMISSION_DIR = "./submission"

    # Ensure necessary directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # =========================================================================
    # File Paths
    # =========================================================================
    # Raw Data
    TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
    TEST_JSON = os.path.join(INPUT_DIR, "test.json")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Metadata (Pre-generated CSVs)
    TRAIN_META_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_META_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_META_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Output Submission
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Cache Files (Numpy format for fast loading)
    # These paths are used by the data processing module to store/load processed tensors
    CACHE_X_TRAIN = os.path.join(WORKING_DIR, "X_train.npy")
    CACHE_Y_TRAIN = os.path.join(WORKING_DIR, "y_train.npy")
    CACHE_ANGLE_TRAIN = os.path.join(WORKING_DIR, "angle_train.npy")

    CACHE_X_VAL = os.path.join(WORKING_DIR, "X_val.npy")
    CACHE_Y_VAL = os.path.join(WORKING_DIR, "y_val.npy")
    CACHE_ANGLE_VAL = os.path.join(WORKING_DIR, "angle_val.npy")

    CACHE_X_TEST = os.path.join(WORKING_DIR, "X_test.npy")
    CACHE_ANGLE_TEST = os.path.join(WORKING_DIR, "angle_test.npy")
    CACHE_ID_TEST = os.path.join(WORKING_DIR, "id_test.npy")

    # =========================================================================
    # Data Parameters
    # =========================================================================
    IMAGE_SIZE = 75
    INPUT_CHANNELS = 3  # Band 1 (HH), Band 2 (HV), Average ((HH+HV)/2)
    NUM_CLASSES = 1  # Binary classification (Iceberg vs Ship)

    # =========================================================================
    # Model Hyperparameters (DPDB-NBA-CNN)
    # =========================================================================
    # Backbone: Plain CNN (4 Stages)
    BACKBONE_CHANNELS = [64, 128, 128, 128]

    # Activation
    LEAKY_RELU_SLOPE = 0.1

    # Regularization
    USE_DROPBLOCK = True
    DROPBLOCK_BLOCK_SIZE = 5
    DROPBLOCK_PROB = 0.1
    DROPOUT_RATE = 0.5  # Applied in the classification head

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    SEED = 42
    N_FOLDS = 5
    BATCH_SIZE = 32
    NUM_EPOCHS = 75
    PATIENCE = 12

    # Optimizer (AdamW)
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 0.01

    # =========================================================================
    # Compute
    # =========================================================================
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def get_checkpoint_path(fold_idx):
        """Returns the path to save/load the model checkpoint for a specific fold."""
        return os.path.join(Config.WORKING_DIR, f"model_fold_{fold_idx}.pth")
