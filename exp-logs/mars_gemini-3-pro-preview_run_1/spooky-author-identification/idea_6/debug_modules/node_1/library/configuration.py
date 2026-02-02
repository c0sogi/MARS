import os
import torch


class Config:
    """
    Configuration class for Multi-Scale Pooling Stacking with Dynamic Context.
    Centralizes all hyperparameters, file paths, and model settings.
    """

    # -------------------------------------------------------------------------
    # General Configuration
    # -------------------------------------------------------------------------
    SEED = 42
    DEBUG = (
        False  # Toggle for debugging (runs on subset if implemented in training loop)
    )
    NUM_WORKERS = 4

    # -------------------------------------------------------------------------
    # Directories & Paths
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_6"
    SUBMISSION_DIR = "./submission"

    # Ensure working and submission directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Data Files (using metadata splits)
    TRAIN_DATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_DATA_PATH = os.path.join(METADATA_DIR, "validation.csv")
    TEST_DATA_PATH = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Files
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # -------------------------------------------------------------------------
    # Data Processing
    # -------------------------------------------------------------------------
    # Max sequence length set to 99th percentile of training data to avoid context bloat
    MAX_LENGTH = 256
    NUM_FOLDS = 5

    # Target Mapping
    LABEL2ID = {"EAP": 0, "HPL": 1, "MWS": 2}
    ID2LABEL = {0: "EAP", 1: "HPL", 2: "MWS"}
    NUM_CLASSES = 3

    # -------------------------------------------------------------------------
    # Expert A: Deep Semantic Model (DeBERTa-v3-Large)
    # -------------------------------------------------------------------------
    MODEL_NAME = "microsoft/deberta-v3-large"

    # Architecture
    HIDDEN_DROPOUT_PROB = 0.1
    LAYER_NORM_EPS = 1e-7

    # Training Hyperparameters
    EPOCHS = 4
    TRAIN_BATCH_SIZE = 4  # Fits A100 40GB with large model
    VALID_BATCH_SIZE = 16
    GRADIENT_ACCUMULATION_STEPS = 4  # Effective Batch Size = 16

    LEARNING_RATE = 1e-5
    WEIGHT_DECAY = 0.01
    MAX_GRAD_NORM = 1.0

    # Optimization Strategies
    LLRD_DECAY = 0.9  # Layer-wise Learning Rate Decay
    WARMUP_RATIO = 0.1
    PATIENCE = 2  # Early Stopping

    # -------------------------------------------------------------------------
    # Expert B: Surface Stylometric Model (TF-IDF + Logistic Regression)
    # -------------------------------------------------------------------------
    # TF-IDF Settings
    TFIDF_WORD_NGRAM_RANGE = (1, 3)
    TFIDF_CHAR_NGRAM_RANGE = (3, 5)

    # Feature Limits (Sparse Matrix)
    TFIDF_MAX_FEATURES_WORD = 20000
    TFIDF_MAX_FEATURES_CHAR = 30000

    # Logistic Regression Settings
    LR_C = 1.0
    LR_SOLVER = "saga"
    LR_MAX_ITER = 1000

    # -------------------------------------------------------------------------
    # Meta-Learner: XGBoost
    # -------------------------------------------------------------------------
    XGB_PARAMS = {
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": "mlogloss",
        "learning_rate": 0.05,
        "max_depth": 4,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "n_estimators": 2000,
        "early_stopping_rounds": 50,
        "n_jobs": -1,
        "verbosity": 0,
        "seed": SEED,
        "tree_method": "hist",
    }

    # -------------------------------------------------------------------------
    # Hardware
    # -------------------------------------------------------------------------
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
