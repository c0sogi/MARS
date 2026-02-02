import os
import torch


class Config:
    """
    Configuration class for the Author Identification task.
    Centralizes hyperparameters, file paths, and model settings for the
    Heterogeneous Hybrid Ensemble (DeBERTa + RoBERTa + Statistical).
    """

    # =========================================================================
    # General Settings
    # =========================================================================
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 4  # Number of dataloader workers
    DEBUG = False  # Set to True for fast debugging runs on a subset
    DEBUG_SAMPLE_SIZE = 100  # Number of samples to use in debug mode

    # =========================================================================
    # File Paths & Directories
    # =========================================================================
    # Input Directories (Read-Only)
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working Directory (Write Allowed)
    # Using idea_5 folder for caching and model artifacts
    WORKING_DIR = "./working/idea_5"
    SUBMISSION_DIR = "./submission"

    # Ensure write directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Data File Paths
    # Using metadata files as primary source for training/validation
    TRAIN_DATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_DATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_DATA_PATH = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Paths
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # Data Configuration
    # =========================================================================
    TARGET_COL = "author"
    TEXT_COL = "text"
    ID_COL = "id"
    CLASS_LABELS = ["EAP", "HPL", "MWS"]
    NUM_CLASSES = 3

    # Tokenizer / Model Input Settings
    # Max length set to 512 based on model constraints and data analysis
    MAX_LENGTH = 512

    # =========================================================================
    # Model Architecture Settings
    # =========================================================================
    # Branch 1: DeBERTa-v3-base (Syntax-Aware)
    MODEL_DEBERTA = "microsoft/deberta-v3-base"

    # Branch 2: RoBERTa-base (Robust Positional Context)
    MODEL_ROBERTA = "roberta-base"

    # Head Architecture: Concatenate [CLS] from last 4 hidden layers
    USE_MULTI_LAYER_CONCAT = True

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    # Cross-Validation
    N_FOLDS = 5

    # Training Loop
    EPOCHS = 5
    # Batch size 16 fits comfortably on A100 (40GB) with 512 seq len
    BATCH_SIZE = 16
    GRAD_ACCUMULATION_STEPS = 1
    LEARNING_RATE = 2e-5
    WEIGHT_DECAY = 0.01
    MAX_GRAD_NORM = 1.0

    # Scheduler
    SCHEDULER_TYPE = "linear"
    WARMUP_RATIO = 0.1

    # Early Stopping
    PATIENCE = 3  # Stop if validation metric doesn't improve for 3 checks

    # =========================================================================
    # Advanced Regularization: Adversarial Weight Perturbation (AWP)
    # =========================================================================
    USE_AWP = True
    AWP_START_EPOCH = 1  # Start AWP after the 1st epoch (warmup)
    AWP_LR = 1e-4  # Learning rate for adversarial perturbation
    AWP_EPS = 1e-2  # Epsilon (magnitude) of perturbation

    # =========================================================================
    # Domain-Adaptive Pre-training (Masked Language Modeling)
    # =========================================================================
    PERFORM_MLM = True
    MLM_EPOCHS = 3
    MLM_BATCH_SIZE = 16
    MLM_LR = 5e-5
    MLM_MASK_PROB = 0.15
    # Directory to store the domain-adapted models
    MLM_MODEL_DIR = os.path.join(WORKING_DIR, "mlm_models")

    # =========================================================================
    # Statistical Model Settings (TF-IDF)
    # =========================================================================
    TFIDF_NGRAM_RANGE_WORD = (1, 2)
    TFIDF_NGRAM_RANGE_CHAR = (3, 5)
    TFIDF_MAX_FEATURES = 20000

    # =========================================================================
    # Caching
    # =========================================================================
    # Path to save processed datasets (parquet/npy)
    CACHE_DIR = WORKING_DIR
