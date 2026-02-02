import os
import torch


class Config:
    """
    Global configuration for the Author Identification pipeline.
    Includes paths, hyperparameters for both Neural and Linear models,
    and general execution settings.
    """

    # -------------------------------------------------------------------------
    # General Settings
    # -------------------------------------------------------------------------
    SEED = 42
    NUM_WORKERS = 4
    # Check for GPU availability
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # -------------------------------------------------------------------------
    # File Paths & Directories
    # -------------------------------------------------------------------------
    # Base directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_2"
    SUBMISSION_DIR = "./submission"

    # Data files (using metadata splits as requested)
    TRAIN_DATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_DATA_PATH = os.path.join(METADATA_DIR, "validation.csv")
    TEST_DATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Submission output
    SUBMISSION_FILE_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Model saving paths
    TRANSFORMER_MODEL_DIR = os.path.join(WORKING_DIR, "transformer_model")
    LINEAR_MODEL_PATH = os.path.join(WORKING_DIR, "linear_model.joblib")
    VECTORIZER_PATH = os.path.join(WORKING_DIR, "vectorizers.joblib")

    # Cache directories for processed features
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")

    # -------------------------------------------------------------------------
    # Stylometric Branch (Linear Model) Hyperparameters
    # -------------------------------------------------------------------------
    # TF-IDF Settings
    # Word n-grams (1-3) and Character n-grams (3-5) as per idea description
    TFIDF_WORD_NGRAM_RANGE = (1, 3)
    TFIDF_CHAR_NGRAM_RANGE = (3, 5)

    # Max features per vectorizer (None = all features, or set integer limit)
    # Keeping it reasonably high but bounded to prevent memory explosion
    MAX_FEATURES_WORD = 20000
    MAX_FEATURES_CHAR = 30000

    # Logistic Regression Settings
    # 'saga' is efficient for large sparse datasets and supports multinomial loss
    LOGREG_PARAMS = {
        "solver": "saga",
        "C": 1.0,
        "penalty": "l2",
        "multi_class": "multinomial",
        "max_iter": 1000,
        "random_state": SEED,
        "n_jobs": -1,
        "tol": 1e-4,
    }

    # -------------------------------------------------------------------------
    # Contextual Branch (Transformer) Hyperparameters
    # -------------------------------------------------------------------------
    # Cite solution_lesson_node_00014: DeBERTa-v3 Superiority
    MODEL_NAME = "microsoft/deberta-v3-base"

    # Tokenizer settings
    MAX_LENGTH = 128  # Sufficient for sentence-level tasks

    # Training settings
    TRAIN_BATCH_SIZE = 16
    VAL_BATCH_SIZE = 32
    LEARNING_RATE = 2e-5
    WEIGHT_DECAY = 0.01
    # Cite solution_lesson_node_00019: OOF Stacking requires K-Fold
    N_FOLDS = 5
    EPOCHS = 3  # Reduced epochs per fold to fit runtime

    # Optimization
    EARLY_STOPPING_PATIENCE = 2

    # -------------------------------------------------------------------------
    # Ensemble Settings
    # -------------------------------------------------------------------------
    # Cite solution_lesson_node_00008: XGBoost Meta-Learner
    XGB_PARAMS = {
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": "mlogloss",
        "learning_rate": 0.05,
        "max_depth": 3,
        "n_estimators": 1000,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "verbosity": 0,
        "n_jobs": -1,
        "random_state": SEED,
    }

    # Target Mapping
    LABEL2ID = {"EAP": 0, "HPL": 1, "MWS": 2}
    ID2LABEL = {0: "EAP", 1: "HPL", 2: "MWS"}

    @classmethod
    def setup(cls):
        """
        Creates necessary working directories.
        Should be called at the start of the pipeline.
        """
        dirs = [
            cls.WORKING_DIR,
            cls.SUBMISSION_DIR,
            cls.TRANSFORMER_MODEL_DIR,
            cls.CACHE_DIR,
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)

    @classmethod
    def get_device(cls):
        return torch.device(cls.DEVICE)
