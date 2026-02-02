import os
import torch


class Config:
    # --------------------------------------------------------------------------
    # General & Reproducibility
    # --------------------------------------------------------------------------
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset of data for debugging
    DEBUG_SAMPLE_SIZE = 100  # Number of samples to use when DEBUG is True

    # --------------------------------------------------------------------------
    # Directory Paths
    # --------------------------------------------------------------------------
    # Input data (Read-Only)
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Output data (Write-Allowed)
    # Using 'idea_4' as the specific working directory for this iteration
    WORKING_DIR = "./working/idea_4"
    SUBMISSION_DIR = "./submission"

    # Ensure necessary write directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # --------------------------------------------------------------------------
    # File Paths
    # --------------------------------------------------------------------------
    # Using the pre-split metadata files for consistent training/validation
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_PATH = os.path.join(METADATA_DIR, "validation.csv")
    TEST_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Submission files
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --------------------------------------------------------------------------
    # Compute Environment
    # --------------------------------------------------------------------------
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # Number of workers for DataLoaders

    # --------------------------------------------------------------------------
    # Expert A: Semantic Model (DeBERTa) Hyperparameters
    # --------------------------------------------------------------------------
    DEBERTA_MODEL = "microsoft/deberta-v3-base"
    MAX_LEN = 512
    BATCH_SIZE = 16  # Increased for A100
    VAL_BATCH_SIZE = 32  # Increased for A100
    LEARNING_RATE = 2e-5
    WEIGHT_DECAY = 0.01
    EPOCHS = 3
    EARLY_STOPPING_PATIENCE = 1

    # --------------------------------------------------------------------------
    # Expert B: Lexical Model (TF-IDF) Hyperparameters
    # --------------------------------------------------------------------------
    TFIDF_NGRAMS = (1, 3)  # Word n-gram range
    TFIDF_CHAR_NGRAMS = (3, 5)  # Character n-gram range
    TFIDF_MAX_FEATURES = 20000  # Max features for vectorization

    # --------------------------------------------------------------------------
    # Expert C: Syntactic Model (POS) Hyperparameters
    # --------------------------------------------------------------------------
    POS_TAGS = True  # Enable POS feature extraction
    POS_NGRAMS = (1, 4)  # POS tag n-gram range
    SPACY_MODEL = "en_core_web_sm"
    POS_MAX_FEATURES = 10000

    NUM_FOLDS = 5  # K-Fold Cross Validation

    # --------------------------------------------------------------------------
    # Meta-Learner (XGBoost) Hyperparameters
    # --------------------------------------------------------------------------
    XGB_PARAMS = {
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": "mlogloss",
        "eta": 0.05,
        "max_depth": 4,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "verbosity": 0,
        "n_jobs": -1,
        "seed": SEED,
    }

    # --------------------------------------------------------------------------
    # Labels
    # --------------------------------------------------------------------------
    LABELS = ["EAP", "HPL", "MWS"]
    ID2LABEL = {0: "EAP", 1: "HPL", 2: "MWS"}
    LABEL2ID = {"EAP": 0, "HPL": 1, "MWS": 2}
