import os
import torch


class Config:
    """
    Configuration class for the Tri-View Stacking Ensemble solution.
    """

    # ==========================================
    # General Settings
    # ==========================================
    SEED = 42
    DEBUG = False  # Set to True to run on a subset of data for debugging
    N_FOLDS = 5  # Number of folds for stacking
    NUM_WORKERS = 2  # Number of dataloader workers

    # ==========================================
    # Paths & Directories
    # ==========================================
    # Input Metadata (Pre-split)
    METADATA_DIR = "./metadata"
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_PATH = os.path.join(METADATA_DIR, "test.parquet")

    # Working Directory for Caching (Idea 4)
    WORKING_DIR = "./working/idea_4"
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Submission Directory
    SUBMISSION_DIR = "./submission"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Data Columns
    # ==========================================
    # Use edit-aware text to prevent leakage (removes "EDIT: Thanks for pizza")
    TEXT_COL = "request_text_edit_aware"
    TITLE_COL = "request_title"
    TARGET_COL = "requester_received_pizza"
    ID_COL = "request_id"

    # Columns strictly excluded to prevent leakage (retrieval time features)
    LEAKAGE_KEYWORDS = ["_at_retrieval", "requester_user_flair"]

    # ==========================================
    # Feature Engineering Hyperparameters
    # ==========================================
    # Lexical View (TF-IDF)
    TFIDF_MAX_FEATURES = 3000
    TFIDF_NGRAM_RANGE = (1, 2)

    # Semantic View (Transformer)
    BERT_MODEL_NAME = "distilbert-base-uncased"
    MAX_SEQ_LEN = (
        256  # Truncate to 256 to save compute, usually sufficient for requests
    )

    # ==========================================
    # Model Hyperparameters
    # ==========================================

    # Level 1: Lexical Bagger (Random Forest)
    # Strong on sparse high-dimensional data
    RF_PARAMS = {
        "n_estimators": 300,
        "max_depth": None,
        "min_samples_leaf": 2,
        "min_samples_split": 5,
        "class_weight": "balanced",
        "random_state": SEED,
        "n_jobs": -1,
    }

    # Level 1: Semantic Fine-Tuner (DistilBERT)
    # End-to-end fine-tuning for deep semantic signal
    BERT_TRAIN_PARAMS = {
        "batch_size": 16,
        "learning_rate": 2e-5,
        "epochs": 3,
        "weight_decay": 0.01,
        "early_stopping_patience": 1,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    # Level 1: Contextual Booster (XGBoost)
    # Strong on dense metadata and engineered style features
    XGB_PARAMS = {
        "n_estimators": 200,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "random_state": SEED,
        "n_jobs": -1,
        "verbosity": 0,
    }

    # Level 2: Meta-Learner (Logistic Regression)
    # Calibrates the probabilities from the three views
    META_PARAMS = {
        "C": 1.0,
        "penalty": "l2",
        "solver": "lbfgs",
        "class_weight": None,  # Let the base models handle balance
        "random_state": SEED,
    }
