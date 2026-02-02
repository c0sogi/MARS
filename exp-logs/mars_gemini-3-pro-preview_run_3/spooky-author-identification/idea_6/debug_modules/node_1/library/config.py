import os
import torch


class Config:
    """
    Global configuration for the Author Identification Task.
    Includes paths, hyperparameters, and model settings.
    """

    # =========================================================================
    # General Settings
    # =========================================================================
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset of data for debugging
    DEBUG_SAMPLE_SIZE = 100
    NUM_WORKERS = 4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =========================================================================
    # Paths
    # =========================================================================
    # Input directories (Read-Only)
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Output directories (Writeable)
    # Using 'idea_6' as the current working namespace
    WORKING_DIR = "./working/idea_6"
    SUBMISSION_DIR = "./submission"

    # Ensure writeable directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # File Paths
    TRAIN_DATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_DATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_DATA_PATH = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Artifact Paths (for caching)
    MLM_MODEL_DIR = os.path.join(WORKING_DIR, "mlm_models")
    FINETUNED_MODEL_DIR = os.path.join(WORKING_DIR, "finetuned_models")
    FEATURES_DIR = os.path.join(WORKING_DIR, "features")

    os.makedirs(MLM_MODEL_DIR, exist_ok=True)
    os.makedirs(FINETUNED_MODEL_DIR, exist_ok=True)
    os.makedirs(FEATURES_DIR, exist_ok=True)

    # =========================================================================
    # Data Processing
    # =========================================================================
    MAX_LEN = 256  # Sufficient for most sentences (mean char len ~148)
    LABEL_MAP = {"EAP": 0, "HPL": 1, "MWS": 2}
    ID2LABEL = {0: "EAP", 1: "HPL", 2: "MWS"}
    NUM_CLASSES = 3

    # =========================================================================
    # Model Hyperparameters
    # =========================================================================
    N_FOLDS = 5

    # Transformer Backbones
    # Using standard HuggingFace Hub identifiers
    MODEL_BACKBONES = ["microsoft/deberta-v3-base", "roberta-base"]

    # Training Parameters
    BATCH_SIZE = 16  # Adjusted for A100 usage with potential gradient accumulation
    LEARNING_RATE = 2e-5
    WEIGHT_DECAY = 0.01
    EPOCHS = 5
    PATIENCE = 2  # Early stopping patience
    MAX_GRAD_NORM = 1.0

    # Masked Language Modeling (DAPT)
    MLM_EPOCHS = 3
    MLM_BATCH_SIZE = 16
    MLM_LEARNING_RATE = 5e-5
    MLM_MASK_PROB = 0.15

    # =========================================================================
    # Statistical & Stylometric Features
    # =========================================================================
    TFIDF_PARAMS = {
        "ngram_range_word": (1, 2),
        "ngram_range_char": (3, 5),
        "max_features_word": 15000,
        "max_features_char": 25000,
        "sublinear_tf": True,
    }

    USE_STYLOMETRIC_FEATURES = True

    # =========================================================================
    # Semi-Supervised Learning (Pseudo-Labeling)
    # =========================================================================
    PSEUDO_LABEL_THRESHOLD = 0.95
    USE_PSEUDO_LABELING = True
