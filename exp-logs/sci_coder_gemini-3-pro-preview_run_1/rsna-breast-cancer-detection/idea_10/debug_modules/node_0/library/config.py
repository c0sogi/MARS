import os
import torch


class Config:
    # =========================================================================
    # Paths & Directories
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"

    # Specific cache directory for this idea (Idea 10)
    # Used for caching processed data/tensors
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_10")
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Metadata File Paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Image Directories
    TRAIN_IMAGES_DIR = os.path.join(INPUT_DIR, "train_images")
    TEST_IMAGES_DIR = os.path.join(INPUT_DIR, "test_images")

    # Submission
    SUBMISSION_DIR = "./submission"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # Data Configuration
    # =========================================================================
    # Input dimensions: (3, 768, 768) -> Image, Age Map, Implant Map
    IMG_SIZE = (768, 768)
    IN_CHANNELS = 3

    # Column names in metadata
    TARGET_COL = "cancer"
    ID_COL = "prediction_id"
    PATIENT_ID_COL = "patient_id"
    IMAGE_ID_COL = "image_id"
    FILE_PATH_COL = "file_path"
    LATERALITY_COL = "laterality"
    VIEW_COL = "view"

    # Auxiliary inputs for channel construction
    AGE_COL = "age"
    IMPLANT_COL = "implant"

    # =========================================================================
    # Model Hyperparameters
    # =========================================================================
    BACKBONE = "efficientnet_b2"
    PRETRAINED = True

    # Siamese Network Settings
    USE_SIAMESE = True

    # Feature Pyramid Levels to use for Difference Module
    # EfficientNet-B2 usually has features at indices corresponding to strides
    # We target P3, P4, P5
    PYRAMID_LEVELS = [3, 4, 5]

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    SEED = 42

    # Training Loop
    NUM_EPOCHS = 10
    BATCH_SIZE = 8  # Conservative for 768x768 on 40GB GPU

    # Optimization
    LR = 1e-4
    WEIGHT_DECAY = 1e-2

    # Loss Function Handling
    # Calculated as ~ (Negatives / Positives) -> 38551 / 816 ≈ 47.2
    POS_WEIGHT = 47.0

    # Gradient Handling
    MAX_GRAD_NORM = None  # Explicitly disabled as per Idea description

    # =========================================================================
    # Hardware & Performance
    # =========================================================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # Using 12 vCPUs available
    NUM_WORKERS = 12
    PIN_MEMORY = True

    # =========================================================================
    # Debugging & Development
    # =========================================================================
    # Set to True to run on a small subset of data for pipeline verification
    DEBUG = False
    DEBUG_SAMPLES = 500
