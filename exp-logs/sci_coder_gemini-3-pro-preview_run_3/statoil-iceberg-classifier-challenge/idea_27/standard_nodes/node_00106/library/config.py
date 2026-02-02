import os
import torch


class Config:
    # ==========================================
    # Paths & Directories
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_27"
    SUBMISSION_DIR = "./submission"

    # Metadata Files
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Raw Data Files
    TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
    TEST_JSON = os.path.join(INPUT_DIR, "test.json")

    # Output Paths
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Data Configuration
    # ==========================================
    IMG_HEIGHT = 75
    IMG_WIDTH = 75
    IN_CHANNELS = 3  # HH, HV, Average((HH+HV)/2)

    # ==========================================
    # Model Architecture (Hybrid-Pooling SE-CNN)
    # ==========================================
    # Backbone structure: 4 stages
    # Width strategy: Early expansion to 128, then capped
    BLOCK_CHANNELS = [64, 128, 128, 128]

    # Activation
    LEAKY_RELU_SLOPE = 0.1  # Preserves negative signal values

    # Regularization
    DROPOUT_RATE = 0.5
    USE_BIAS = True  # Explicitly retain bias for initialization dynamics

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    SEED = 42
    N_FOLDS = 5

    BATCH_SIZE = 32
    EPOCHS = 75

    # Optimization
    LEARNING_RATE = 1e-3  # Constant LR
    WEIGHT_DECAY = 1e-3  # L2 Regularization to prevent overfitting on noise
    PATIENCE = 12  # Early stopping patience

    # ==========================================
    # Runtime & Debugging
    # ==========================================
    NUM_WORKERS = 2
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Debug flag: If True, uses a small subset of data for quick pipeline verification
    DEBUG = False
    DEBUG_SUBSET_SIZE = 100

    @classmethod
    def setup(cls):
        """Creates necessary working directories."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Initialize directories on import
Config.setup()
