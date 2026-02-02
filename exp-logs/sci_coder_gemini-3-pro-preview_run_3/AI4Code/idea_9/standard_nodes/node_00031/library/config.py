import os
import torch


class Config:
    """
    Configuration for the Multi-Scale Structural Alignment Regressor (MSSAR) solution.
    """

    # =========================================================================
    # Directories and Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_9"
    SUBMISSION_DIR = "./submission"

    # Metadata Paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Output Paths
    MODEL_OUTPUT_PATH = os.path.join(WORKING_DIR, "fine_tuned_mpnet")
    LGBM_MODEL_PATH = os.path.join(WORKING_DIR, "lgbm_model.txt")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Cache Paths (for intermediate features)
    TRAIN_FEATURES_PATH = os.path.join(WORKING_DIR, "train_features.parquet")
    VAL_FEATURES_PATH = os.path.join(WORKING_DIR, "val_features.parquet")
    TEST_FEATURES_PATH = os.path.join(WORKING_DIR, "test_features.parquet")

    # =========================================================================
    # Data Parameters
    # =========================================================================
    # Number of notebooks to use for contrastive fine-tuning (Idea 9: 40k)
    NUM_FINE_TUNE_NOTEBOOKS = 40000

    # Max token length for tokenizer (MPNet limit is 512, but 128 is efficient for code/md snippets)
    MAX_LENGTH = 128

    # Debugging
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 1000  # Number of notebooks to process if DEBUG is True

    # =========================================================================
    # Model Architecture Parameters
    # =========================================================================
    MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

    # Multi-Scale Smoothing Kernels for feature extraction
    # k=1: Fine scale (raw)
    # k=3: Medium scale (local blocks)
    # k=5: Coarse scale (large chunks)
    SMOOTHING_KERNELS = [1, 3, 5]

    # =========================================================================
    # Training Parameters (Stage 1: Contrastive Fine-Tuning)
    # =========================================================================
    BATCH_SIZE = 48  # Tuned for A100 40GB
    EPOCHS = 1
    LEARNING_RATE = 2e-5
    WARMUP_STEPS = 1000
    WEIGHT_DECAY = 0.01

    # =========================================================================
    # Training Parameters (Stage 2: LightGBM Regressor)
    # =========================================================================
    LGBM_PARAMS = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbosity": -1,
        "n_jobs": -1,
        "seed": 42,
        "force_col_wise": True,
    }
    LGBM_NUM_BOOST_ROUND = 2000
    LGBM_EARLY_STOPPING_ROUNDS = 100

    # =========================================================================
    # Compute and Environment
    # =========================================================================
    SEED = 42
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def setup(cls):
        """
        Creates necessary working directories.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        # Set deterministic behavior where possible
        os.environ["PYTHONHASHSEED"] = str(cls.SEED)
        torch.manual_seed(cls.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cls.SEED)
