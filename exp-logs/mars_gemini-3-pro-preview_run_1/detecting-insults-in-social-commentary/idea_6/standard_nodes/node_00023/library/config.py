import os
import torch


class Config:
    # ====================================================
    # General Settings
    # ====================================================
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 4

    # ====================================================
    # File Paths
    # ====================================================
    # Input Metadata (Pre-split)
    METADATA_DIR = "./metadata"
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_PATH = os.path.join(METADATA_DIR, "validation.csv")
    TEST_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Output Directories
    WORKING_DIR = "./working/idea_7"
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Artifact Paths
    # Path to save/load the domain-adapted backbone after Stage 1
    MLM_MODEL_PATH = os.path.join(WORKING_DIR, "mlm_finetuned")

    # ====================================================
    # Model Architecture
    # ====================================================
    MODEL_NAME = "microsoft/deberta-v3-base"
    MAX_LEN = 128  # Sufficient for mean char count ~200

    # Structural Feature Extraction
    SVD_COMPONENTS = 256
    NGRAM_RANGE_WORD = (1, 2)
    NGRAM_RANGE_CHAR = (3, 5)

    # Variable-Rate Multi-Sample Dropout (VR-MSD)
    DROPOUT_RATES = [0.1, 0.2, 0.3, 0.4, 0.5]

    # ====================================================
    # Training: Stage 1 (Masked Language Modeling)
    # ====================================================
    MLM_EPOCHS = 3
    MLM_BATCH_SIZE = 16
    MLM_LR = 2e-5
    MLM_MASK_PROB = 0.15

    # ====================================================
    # Training: Stage 2 (Classification)
    # ====================================================
    N_FOLDS = 5
    CLS_EPOCHS = 5
    TRAIN_BATCH_SIZE = 16
    VALID_BATCH_SIZE = 32

    # Differential Learning Rates
    LR_BACKBONE = 2e-5
    LR_HEAD = 1e-3
    WEIGHT_DECAY = 0.01

    # Adversarial Weight Perturbation (AWP)
    AWP_ENABLED = True
    AWP_START_EPOCH = 1
    AWP_LR = 1e-4
    AWP_EPS = 1e-4

    # Optimization & Scheduling
    PATIENCE = 3  # Early stopping patience
    MAX_GRAD_NORM = 1.0  # Gradient clipping
    WARMUP_RATIO = 0.1
