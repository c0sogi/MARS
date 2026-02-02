import os
import torch


class Config:
    """
    Configuration class for the Whale Species Prediction task.
    Implements settings for a Progressive Resolution Ensemble of DenseNet121 models
    with ArcFace heads.
    """

    # -------------------------------------------------------------------------
    # Directories and Paths
    # -------------------------------------------------------------------------
    # Input data (Read-Only)
    INPUT_DIR = "./input"
    TRAIN_IMG_DIR = os.path.join(INPUT_DIR, "train")
    TEST_IMG_DIR = os.path.join(INPUT_DIR, "test")

    # Metadata (Generated in previous steps)
    METADATA_DIR = "./metadata"
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Output directories
    WORKING_DIR = "./working/idea_7"
    SUBMISSION_DIR = "./submission"
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # -------------------------------------------------------------------------
    # Global Settings
    # -------------------------------------------------------------------------
    # Random Seeds for the ensemble members (5 independent models)
    ENSEMBLE_SEEDS = [42, 2024, 777, 1234, 5678]

    # Compute
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # Number of data loading workers

    # Debugging
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 100  # Subset size when DEBUG is True

    # -------------------------------------------------------------------------
    # Model Architecture
    # -------------------------------------------------------------------------
    MODEL_NAME = "densenet121"
    PRETRAINED = True

    # Number of classes derived from metadata analysis (Train Set: 4029 unique classes)
    NUM_CLASSES = 4029

    # Projection Head
    EMBEDDING_SIZE = 512
    DROPOUT_RATE = 0.3

    # ArcFace Loss Hyperparameters
    ARCFACE_MARGIN = 0.50
    ARCFACE_SCALE = 30.0

    # -------------------------------------------------------------------------
    # Training Hyperparameters (Progressive Resizing)
    # -------------------------------------------------------------------------
    BATCH_SIZE = 32
    WEIGHT_DECAY = 1e-4

    # Stage 1: Coarse Tuning (Global Features)
    STAGE_1_IMG_SIZE = 256
    STAGE_1_EPOCHS = 10
    STAGE_1_LR = 1e-3

    # Stage 2: Fine Tuning (Refined Decision Boundaries)
    STAGE_2_IMG_SIZE = 320
    STAGE_2_EPOCHS = 6
    STAGE_2_LR = 1e-4

    # -------------------------------------------------------------------------
    # Inference / TTA
    # -------------------------------------------------------------------------
    TTA_FLIP = True  # Use Horizontal Flip Test-Time Augmentation
    TOP_K = 5  # Number of predictions per image

    @classmethod
    def setup(cls):
        """
        Ensures that the necessary working and submission directories exist.
        This is called automatically when the module is imported/class is defined.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Initialize directories
Config.setup()
