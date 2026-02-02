import os
import torch


class Config:
    """
    Configuration for the Tri-Modal Heterogeneous Stacking Essay Scoring System.

    This configuration manages:
    1. Semantic Branch: DeBERTa-v3-large with AWP and Concatenated Pooling.
    2. Lexical Branch: Sparse TF-IDF N-grams with Ridge Regression.
    3. Mechanics Branch: Hand-crafted linguistic features.
    4. Meta-Learner: Non-linear Gradient Boosting (LightGBM).
    """

    # --- General Settings ---
    SEED = 42
    EXP_NAME = "idea_6"
    DEBUG = False  # Flag to enable rapid debugging on subsets

    # --- File Paths ---
    # Metadata (Input)
    TRAIN_DATA_PATH = "./metadata/train.csv"
    VAL_DATA_PATH = "./metadata/val.csv"
    TEST_DATA_PATH = "./metadata/test.csv"

    # Working Directory (Output)
    WORKING_DIR = f"./working/{EXP_NAME}"
    MODEL_DIR = os.path.join(WORKING_DIR, "models")
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")

    # Submission
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --- Compute Resources ---
    # Detect GPU availability
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 4  # Number of dataloader workers

    # --- Semantic Branch (DeBERTa) ---
    MODEL_BACKBONE = "microsoft/deberta-v3-large"
    MAX_LENGTH = 1024

    # Training Hyperparameters
    TRAIN_BATCH_SIZE = 2  # Adjusted for A100 40GB VRAM with 1024 tokens
    VALID_BATCH_SIZE = 4
    LEARNING_RATE = 1e-5
    EPOCHS = 4
    WEIGHT_DECAY = 0.01
    MAX_GRAD_NORM = 1.0

    # Loss & Pooling
    LOSS_FN = "SmoothL1Loss"
    POOLING_TYPE = "concat_mean_max"  # Concatenated Mean and Max Pooling strategy

    # Adversarial Weight Perturbation (AWP)
    USE_AWP = True
    AWP_START_EPOCH = 2
    AWP_LR = 1e-4
    AWP_EPS = 1e-4

    # --- Lexical Branch (TF-IDF + Ridge) ---
    TFIDF_NGRAM_RANGE = (1, 3)  # Unigrams, Bigrams, Trigrams
    TFIDF_MIN_DF = 3
    RIDGE_ALPHA = 1.0

    # --- Mechanics Branch (Linguistic Features) ---
    # List of explicit features to compute
    MECHANICS_FEATURES = [
        "char_count",
        "word_count",
        "avg_sentence_length",
        "sentence_length_var",
        "vocab_richness",
        "flesch_kincaid_grade",
        "gunning_fog",
        "spelling_error_count",
    ]

    # --- Meta-Learner (Stacking) ---
    META_MODEL = "lightgbm"
    N_FOLDS = 5

    # LightGBM Hyperparameters
    LGB_PARAMS = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.01,
        "num_leaves": 31,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbosity": -1,
        "n_estimators": 2000,
        "early_stopping_rounds": 100,
        "random_state": SEED,
        "n_jobs": -1,
    }

    # --- Post-Processing ---
    USE_NELDER_MEAD = True  # Enable Nelder-Mead threshold optimization

    @classmethod
    def setup(cls):
        """
        Creates necessary directory structures for the experiment.
        This ensures ./working/idea_6/ and subdirectories exist before usage.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.MODEL_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

    @classmethod
    def set_debug_mode(cls, debug: bool = True):
        """
        Adjusts configuration for debugging.

        Args:
            debug (bool): If True, reduces epochs, batch sizes, and estimators
                          to allow for a quick end-to-end run.
        """
        cls.DEBUG = debug
        if debug:
            print(f"(!) DEBUG MODE ENABLED for {cls.EXP_NAME}")
            cls.EPOCHS = 1
            cls.TRAIN_BATCH_SIZE = 2
            cls.LGB_PARAMS["n_estimators"] = 50
            cls.LGB_PARAMS["early_stopping_rounds"] = 10
            # Note: Data loaders should check Config.DEBUG to subsample data


# Initialize directories on import
Config.setup()
