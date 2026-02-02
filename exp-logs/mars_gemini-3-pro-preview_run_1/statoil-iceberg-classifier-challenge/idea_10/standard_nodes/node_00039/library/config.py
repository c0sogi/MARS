import os
import torch


class Config:
    # =========================================================================
    # File Paths and Directories
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_10"
    SUBMISSION_DIR = "./submission"

    # Raw Data Files
    TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
    TEST_JSON = os.path.join(INPUT_DIR, "test.json")

    # Metadata Files
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Output Paths
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # =========================================================================
    # Data Processing Parameters
    # =========================================================================
    # Image Dimensions
    ORIGINAL_IMG_SIZE = 75
    IMG_SIZE = 224  # Upsampling target

    # Global Statistics for Min-Max Normalization (from Data Analysis)
    # Band 1 (HH)
    BAND1_MIN = -45.5944
    BAND1_MAX = 32.1806
    # Band 2 (HV)
    BAND2_MIN = -45.6555
    BAND2_MAX = 17.8628

    # Augmentation Parameters
    ROTATION_LIMIT = 20  # degrees (+/-)

    # =========================================================================
    # Model Architecture
    # =========================================================================
    MODEL_NAME = "resnet18"
    PRETRAINED = True
    NUM_CLASSES = 1
    DROPOUT_RATE = 0.5

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    # General
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    # Batch Size
    BATCH_SIZE = 32  # Cite solution_lesson_node_00038: Smaller batch size for more gradient steps

    # Optimization
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 0.01

    # Scheduler Params
    SCHEDULER_FACTOR = (
        0.1  # Cite solution_lesson_node_00011: Aggressive decay for fine-tuning
    )
    SCHEDULER_PATIENCE = 5

    # Loss
    LABEL_SMOOTHING = 0.05

    # Phase 1: Calibration (Cross-Validation)
    N_FOLDS = 5
    CALIBRATION_EPOCHS = 50  # Increased to allow ReduceLROnPlateau to work

    # Phase 2: Full-Fit Ensemble
    # We will use these seeds to initialize 5 independent models trained on full data
    ENSEMBLE_SEEDS = [42, 2024, 777, 12345, 99]

    # =========================================================================
    # Inference
    # =========================================================================
    TTA_FLIPS = True  # Use Horizontal and Vertical flips during inference
