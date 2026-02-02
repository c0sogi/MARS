import os


class Config:
    # =========================================================================
    # Directories and File Paths
    # =========================================================================
    # Input Data (Read-Only)
    INPUT_DIR = "./input"
    TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
    TEST_JSON = os.path.join(INPUT_DIR, "test.json")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Metadata (Pre-generated)
    METADATA_DIR = "./metadata"
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Working Directory (Write Access)
    # Using a specific subdirectory for this idea to avoid conflicts
    WORKING_DIR = "./working/idea_65"

    # Cache for processed numpy arrays
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")

    # Checkpoints for model weights
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")

    # Final Submission
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Create directories if they don't exist
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # =========================================================================
    # Data Specifications
    # =========================================================================
    IMAGE_SIZE = 75
    NUM_BANDS_RAW = 2
    # We use 3 channels: HH, HV, and (HH+HV)/2
    NUM_INPUT_CHANNELS = 3
    NUM_CLASSES = 1

    # =========================================================================
    # Model Hyperparameters (CMSDI-CNN)
    # =========================================================================
    MODEL_NAME = "CMSDI_CNN"

    # Backbone (Plain CNN)
    BACKBONE_CHANNELS = [64, 128, 128, 128]
    LEAKY_RELU_SLOPE = 0.1

    # Head
    HIDDEN_DIM = 256
    DROPOUT_RATE = 0.5
    NUM_DROPOUT_SAMPLES = 5

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    SEED = 42
    NUM_FOLDS = 5

    # Optimization
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4  # L2 Regularization

    # Duration and Early Stopping
    NUM_EPOCHS = 75
    PATIENCE = 12

    # Hardware
    NUM_WORKERS = 4  # Adjust based on vCPU count (12 available)

    # =========================================================================
    # Debugging / Development
    # =========================================================================
    # Set to True to train on a small subset for quick pipeline verification
    DEBUG = False
    DEBUG_SUBSET_SIZE = 100
