import os
import torch


class Config:
    """
    Configuration for the Spatial-Quadrant Wide-Body Network (SQ-WBN) pipeline.
    """

    # --------------------------------------------------------------------------
    # Global Seeding
    # --------------------------------------------------------------------------
    SEED = 42

    # --------------------------------------------------------------------------
    # Directory and File Paths
    # --------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"

    # Specific artifact directory for this idea
    ARTIFACT_DIR = os.path.join(WORKING_DIR, "idea_20")

    # Raw Data
    TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
    TEST_JSON = os.path.join(INPUT_DIR, "test.json")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Metadata (Stratified Splits)
    TRAIN_META = os.path.join(METADATA_DIR, "train.csv")
    VAL_META = os.path.join(METADATA_DIR, "val.csv")
    TEST_META = os.path.join(METADATA_DIR, "test.csv")

    # Outputs
    SUBMISSION_PATH = os.path.join(ARTIFACT_DIR, "submission.csv")
    CACHE_PATH = os.path.join(ARTIFACT_DIR, "processed_data.npz")

    # --------------------------------------------------------------------------
    # Data Processing Parameters
    # --------------------------------------------------------------------------
    IMG_HEIGHT = 75
    IMG_WIDTH = 75

    # Input Composition: Band 1 (HH), Band 2 (HV), Mean ((HH+HV)/2)
    INPUT_CHANNELS = 3

    # Augmentation Settings
    ROTATION_ANGLES = [0, 90, 180, 270]
    USE_HORIZONTAL_FLIP = True
    USE_VERTICAL_FLIP = False  # Per Lesson 15

    # --------------------------------------------------------------------------
    # Model Architecture: Spatial-Quadrant Wide-Body Network (SQ-WBN)
    # --------------------------------------------------------------------------
    # Wide Backbone Filters (Lesson 77)
    FILTER_SIZES = [64, 128, 128, 128]

    # Dual Pooling (Max+Min) doubles the channel depth at each stage
    # Final Conv Block Output: 128 filters -> Dual Pool -> 256 channels
    # Quadrant Pooling: 4x4 spatial grid -> 2x2 spatial grid
    # Flattened Vector: 2 * 2 * 256 = 1024
    LATENT_DIM = 1024

    DROPOUT_RATE = 0.5
    USE_CBAM = True

    # --------------------------------------------------------------------------
    # Training Hyperparameters
    # --------------------------------------------------------------------------
    NUM_FOLDS = 5
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    LEARNING_RATE = 2e-4  # Conservative start (Lesson 23)
    PATIENCE = 10  # Early Stopping

    # --------------------------------------------------------------------------
    # Debugging and Runtime Control
    # --------------------------------------------------------------------------
    # Set DEBUG to True to run on a small subset of data for testing
    DEBUG = False
    DEBUG_SUBSET_SIZE = 100

    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 2

    @classmethod
    def initialize(cls):
        """
        Sets up the necessary directories for artifacts.
        """
        os.makedirs(cls.ARTIFACT_DIR, exist_ok=True)
        print(f"Artifact directory initialized at: {cls.ARTIFACT_DIR}")
        print(f"Device: {cls.DEVICE}")
