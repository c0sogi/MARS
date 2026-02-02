import os
import torch


class Config:
    """
    Configuration class for the Zero-Initialized Output-Space Residual Network (ZI-OSR Net).
    Centralizes all hyperparameters, file paths, and constants.
    """

    # ====================================================
    # General Settings
    # ====================================================
    PROJECT_NAME = "idea_33"
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # Number of workers for data loading

    # ====================================================
    # File Paths & Directories
    # ====================================================
    # Input Directories (Read-Only)
    INPUT_DIR = "./input"
    TRAIN_DICOM_DIR = os.path.join(INPUT_DIR, "train")
    TEST_DICOM_DIR = os.path.join(INPUT_DIR, "test")

    # Metadata Paths (Pre-generated)
    METADATA_DIR = "./metadata"
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Working Directories (Write Access)
    WORKING_DIR = f"./working/{PROJECT_NAME}"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure working directories exist
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # ====================================================
    # Data Preprocessing & Feature Engineering
    # ====================================================
    # Image Preprocessing
    IMG_SIZE = 260
    NUM_SLICES = 3  # Anchor slice + 2 boundary slices (RGB-like input)

    # Radiological Windowing (Lung Window)
    LUNG_WINDOW_LEVEL = -600
    LUNG_WINDOW_WIDTH = 1500

    # Clinical Features
    # Inputs: Baseline FVC, Relative Time, Age, Sex, SmokingStatus
    CLINICAL_INPUT_DIM = 5

    # Ordinal Encoding for SmokingStatus
    SMOKING_STATUS_MAP = {"Never smoked": 0, "Ex-smoker": 1, "Currently smokes": 2}

    # Sex Encoding
    SEX_MAP = {"Male": 0, "Female": 1}

    # ====================================================
    # Model Architecture
    # ====================================================
    BACKBONE_NAME = "efficientnet_b2"
    PRETRAINED = True

    # The clinical stream uses an over-parameterized MLP
    CLINICAL_HIDDEN_DIMS = [128, 64]

    # ====================================================
    # Training Hyperparameters
    # ====================================================
    EPOCHS = 50
    BATCH_SIZE = 32

    # Differential Learning Rates
    LR_BACKBONE = 1e-4  # Lower LR for the visual backbone
    LR_HEAD = 1e-3  # Higher LR for the MLPs

    # Optimizer & Scheduler
    WEIGHT_DECAY = 1e-2
    T_MAX = EPOCHS  # For Cosine Annealing
    ETA_MIN = 1e-6

    # Loss Function Constants
    METRIC_CONST_SQRT2 = 1.41421356  # sqrt(2)

    # ====================================================
    # Post-Processing & Metrics
    # ====================================================
    MIN_CONFIDENCE = 70.0
    MAX_ERROR_METRIC = 1000.0
