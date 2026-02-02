import os
import torch


class Config:
    """
    Global configuration for the Author Identification pipeline.
    Includes paths, model hyperparameters, and runtime settings.
    """

    # --------------------------------------------------------------------------
    # General Setup
    # --------------------------------------------------------------------------
    SEED = 42
    DEBUG = False  # Set to True for fast debugging runs on a subset
    DEBUG_SAMPLE_SIZE = 100

    # Compute Environment
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # Number of dataloader workers

    # --------------------------------------------------------------------------
    # Directory Paths
    # --------------------------------------------------------------------------
    # Base directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_7"
    SUBMISSION_DIR = "./submission"

    # Ensure writable directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Data Files (using Metadata as source of truth for splits)
    TRAIN_DATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_DATA_PATH = os.path.join(METADATA_DIR, "validation.csv")
    TEST_DATA_PATH = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Final Submission Output
    SUBMISSION_FILE_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --------------------------------------------------------------------------
    # Model Hyperparameters
    # --------------------------------------------------------------------------

    # Expert A: Deep Learning (DeBERTa-v3-Large)
    # Using 'microsoft/deberta-v3-large' as the backbone
    MODEL_NAME = "microsoft/deberta-v3-large"
    MAX_LENGTH = 256  # 99th percentile of training data length

    # Training Dynamics
    # A100 40GB allows reasonable batch size, but we use accumulation for stability
    TRAIN_BATCH_SIZE = 4
    VALID_BATCH_SIZE = 8
    GRADIENT_ACCUMULATION_STEPS = 8  # Effective batch size ~32

    # Optimization
    LEARNING_RATE = 1e-5
    WEIGHT_DECAY = 0.01
    MAX_GRAD_NORM = 1.0
    EPOCHS = 4
    EARLY_STOPPING_PATIENCE = 2
    LAYER_WISE_LR_DECAY = 0.9  # Decay LR for lower layers

    # Expert B: Surface Stylometric (TF-IDF + Logistic Regression)
    TFIDF_WORD_NGRAM_RANGE = (1, 3)
    TFIDF_CHAR_NGRAM_RANGE = (3, 5)
    TFIDF_MIN_DF = 3

    # Meta-Learner: XGBoost
    XGB_PARAMS = {
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": "mlogloss",
        "learning_rate": 0.05,
        "max_depth": 4,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "n_jobs": -1,
        "random_state": SEED,
        "verbosity": 0,
    }

    # Stacking Strategy
    N_FOLDS = 5

    # --------------------------------------------------------------------------
    # Mappings & Constants
    # --------------------------------------------------------------------------
    LABEL2ID = {"EAP": 0, "HPL": 1, "MWS": 2}
    ID2LABEL = {0: "EAP", 1: "HPL", 2: "MWS"}

    # --------------------------------------------------------------------------
    # Caching Paths (Artifacts)
    # --------------------------------------------------------------------------
    # Paths for saving intermediate OOF predictions and features
    CACHE_EXPERT_A_OOF = os.path.join(WORKING_DIR, "expert_a_oof.npy")
    CACHE_EXPERT_A_TEST = os.path.join(WORKING_DIR, "expert_a_test.npy")

    CACHE_EXPERT_B_OOF = os.path.join(WORKING_DIR, "expert_b_oof.npy")
    CACHE_EXPERT_B_TEST = os.path.join(WORKING_DIR, "expert_b_test.npy")

    CACHE_META_FEATURES_TRAIN = os.path.join(WORKING_DIR, "meta_features_train.parquet")
    CACHE_META_FEATURES_TEST = os.path.join(WORKING_DIR, "meta_features_test.parquet")

    # Model checkpoints directory
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
