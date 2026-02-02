import os
import torch


class Config:
    # ==========================================
    # Project & Experiment Identity
    # ==========================================
    PROJECT_NAME = "contrails_segmentation"
    IDEA_NAME = "idea_14"
    SEED = 42

    # Debugging flags
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 500  # Number of samples to use when DEBUG is True

    # ==========================================
    # Directories & File Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Metadata CSVs (Pre-generated)
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VALID_METADATA_PATH = os.path.join(METADATA_DIR, "validation_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Output Directories
    WORKING_DIR = os.path.join("./working", IDEA_NAME)
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Data Engineering & Input Specs
    # ==========================================
    IMAGE_SIZE = 256
    NUM_FRAMES = 8  # Total frames provided in the .npy files

    # Temporal Frame Indices (0-based)
    # The labeled mask corresponds to the 5th image (index 4).
    # We use the 4th image (index 3) for temporal differencing.
    LABELED_FRAME_IDX = 4
    PREV_FRAME_IDX = 3

    # Band Selection
    # Full ABI Bands available: 08, 09, 10, 11, 12, 13, 14, 15, 16
    # Ash False Color Composite uses Bands 11, 14, 15.
    ASH_BAND_IDS = [11, 14, 15]

    # Ash Normalization Bounds (Approximate physical values for scaling)
    # Channel 1: Band 15 - Band 14 (Temperature Difference)
    # Channel 2: Band 14 - Band 11 (Temperature Difference)
    # Channel 3: Band 14 (Brightness Temperature)
    ASH_MIN = [-4.0, -4.0, 243.0]
    ASH_MAX = [2.0, 5.0, 303.0]

    # ==========================================
    # Model Architecture
    # ==========================================
    MODEL_NAME = "HyperDenseConvNeXtUNet"
    ENCODER_NAME = "convnext_tiny"
    ENCODER_WEIGHTS = "imagenet"

    # Input Channels:
    # 3 channels (Ash Composite at t=4) +
    # 3 channels (Ash Composite at t=4 minus Ash Composite at t=3)
    IN_CHANNELS = 6
    OUT_CHANNELS = 1

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 32
    EPOCHS = 30
    LEARNING_RATE = 2e-4
    WEIGHT_DECAY = 0.01

    # Optimization
    OPTIMIZER = "AdamW"
    SCHEDULER = "CosineAnnealingLR"
    MIN_LR = 1e-6

    # Loss Function Strategy
    # Hybrid Loss: Binary Cross Entropy + Batch-Level Dice
    LOSS_FUNCTION = "BCE_BatchDice"

    # Compute
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ==========================================
    # Augmentation (Affine Only)
    # ==========================================
    # We avoid elastic/grid distortions to preserve linear contrail morphology.
    AUG_ROTATION_LIMIT = 15
    AUG_SCALE_LIMIT = (0.9, 1.1)
    AUG_SHIFT_LIMIT = 0.1
    AUG_PROB = 0.5

    # ==========================================
    # Inference & Post-Processing
    # ==========================================
    THRESHOLD = 0.5
    USE_TTA = True  # Enable Test-Time Augmentation (Flip/Rotate)

    @classmethod
    def setup(cls):
        """
        Creates the necessary working and submission directories.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
        print(f"Configuration setup complete. Working directory: {cls.WORKING_DIR}")
