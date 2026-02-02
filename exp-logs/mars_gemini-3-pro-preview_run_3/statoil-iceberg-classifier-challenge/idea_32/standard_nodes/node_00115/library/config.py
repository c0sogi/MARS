import os
import torch


class Config:
    """
    Configuration for the Biased Hybrid-Attentive ResNet (BHA-ResNet) experiment.
    Defines file paths, hyperparameters, and model settings.
    """

    # --------------------------------------------------------------------------
    # Identifiers and Directories
    # --------------------------------------------------------------------------
    IDEA_NAME = "idea_32"

    # Base Directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"
    SUBMISSION_DIR = "./submission"

    # Experiment Specific Directories
    # Used for caching processed numpy arrays to speed up repeated runs
    CACHE_DIR = os.path.join(WORKING_DIR, IDEA_NAME)
    # Used for saving model weights
    CHECKPOINT_DIR = os.path.join(CACHE_DIR, "checkpoints")

    # Ensure mutable directories exist
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # --------------------------------------------------------------------------
    # File Paths
    # --------------------------------------------------------------------------
    # Raw Data
    TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
    TEST_JSON = os.path.join(INPUT_DIR, "test.json")

    # Metadata (Pre-generated)
    TRAIN_META_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_META_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_META_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Output Submission
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Cached Data Paths (Numpy format for speed)
    CACHE_X_TRAIN = os.path.join(CACHE_DIR, "X_train.npy")
    CACHE_Y_TRAIN = os.path.join(CACHE_DIR, "y_train.npy")
    CACHE_ANGLE_TRAIN = os.path.join(CACHE_DIR, "angle_train.npy")
    CACHE_IDS_TRAIN = os.path.join(CACHE_DIR, "ids_train.npy")

    CACHE_X_TEST = os.path.join(CACHE_DIR, "X_test.npy")
    CACHE_ANGLE_TEST = os.path.join(CACHE_DIR, "angle_test.npy")
    CACHE_IDS_TEST = os.path.join(CACHE_DIR, "ids_test.npy")

    # --------------------------------------------------------------------------
    # Data Hyperparameters
    # --------------------------------------------------------------------------
    IMAGE_SIZE = 75
    # Input channels: HH, HV, and (HH+HV)/2
    INPUT_CHANNELS = 3
    NUM_CLASSES = 1

    # Augmentation Flags
    AUGMENT_HFLIP = True
    AUGMENT_VFLIP = True

    # --------------------------------------------------------------------------
    # Model Hyperparameters (BHA-ResNet)
    # --------------------------------------------------------------------------
    # Channel widths for the 4 residual stages
    MODEL_STAGES = [64, 128, 128, 128]

    # Structural Constraints
    USE_BIAS = True  # Retain bias in convolutions
    LEAKY_RELU_SLOPE = 0.1  # Preserve negative signal (shadows)
    DROPOUT_RATE = 0.5  # Applied in classification head
    USE_HYBRID_ATTENTION = True  # Use SE Module with GAP

    # --------------------------------------------------------------------------
    # Training Hyperparameters
    # --------------------------------------------------------------------------
    SEED = 42
    NUM_FOLDS = 5

    # Optimization
    BATCH_SIZE = 32
    NUM_EPOCHS = 75
    PATIENCE = 12  # Early stopping patience
    LEARNING_RATE = 1e-3  # Constant learning rate
    WEIGHT_DECAY = 1e-4  # L2 Regularization

    # --------------------------------------------------------------------------
    # Compute & Inference
    # --------------------------------------------------------------------------
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Test-Time Augmentation
    USE_TTA = False

    # --------------------------------------------------------------------------
    # Debugging
    # --------------------------------------------------------------------------
    # If set to an integer, limits the number of samples for rapid testing
    MAX_DEBUG_SAMPLES = None
