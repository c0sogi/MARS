import os
import random
import numpy as np
import torch


class Config:
    """
    Global configuration for the Natural Questions pipeline.
    This class centralizes all file paths, hyperparameters, and environment settings.
    """

    def __init__(self, debug: bool = False, load_cached_data: bool = True):
        """
        Initialize configuration.

        Args:
            debug (bool): If True, uses small subsets of data for rapid prototyping.
            load_cached_data (bool): If True, attempts to load processed features from disk.
        """
        # ---------------------------------------------------------------------
        # 1. Directory and File Paths
        # ---------------------------------------------------------------------
        self.ROOT_DIR = "."
        self.INPUT_DIR = os.path.join(self.ROOT_DIR, "input")
        self.METADATA_DIR = os.path.join(self.ROOT_DIR, "metadata")
        self.WORKING_DIR = os.path.join(self.ROOT_DIR, "working")
        self.CACHE_DIR = os.path.join(self.WORKING_DIR, "idea_1")
        self.SUBMISSION_DIR = os.path.join(self.ROOT_DIR, "submission")

        # Ensure mutable directories exist
        os.makedirs(self.CACHE_DIR, exist_ok=True)
        os.makedirs(self.SUBMISSION_DIR, exist_ok=True)

        # Raw Data Files
        self.TRAIN_DATA_PATH = os.path.join(self.INPUT_DIR, "simplified-nq-train.jsonl")
        self.TEST_DATA_PATH = os.path.join(self.INPUT_DIR, "simplified-nq-test.jsonl")
        self.SAMPLE_SUBMISSION_PATH = os.path.join(
            self.INPUT_DIR, "sample_submission.csv"
        )

        # Metadata Files (Parquet)
        self.TRAIN_META_PATH = os.path.join(self.METADATA_DIR, "train.parquet")
        self.VAL_META_PATH = os.path.join(self.METADATA_DIR, "val.parquet")
        self.TEST_META_PATH = os.path.join(self.METADATA_DIR, "test.parquet")

        # Output Files
        self.SUBMISSION_FILE = os.path.join(self.SUBMISSION_DIR, "submission.csv")

        # ---------------------------------------------------------------------
        # 2. General Settings
        # ---------------------------------------------------------------------
        self.DEBUG = debug
        self.LOAD_CACHED_DATA = load_cached_data
        self.SEED = 42
        self.NUM_WORKERS = 12  # Based on available vCPUs

        # ---------------------------------------------------------------------
        # 3. Data Preprocessing Hyperparameters
        # ---------------------------------------------------------------------
        # If debug is True, limit the number of samples to process
        self.TRAIN_SAMPLE_SIZE = 2000 if self.DEBUG else None
        self.VAL_SAMPLE_SIZE = 500 if self.DEBUG else None

        # Negative Subsampling: Ratio of negative candidates to keep per positive candidate
        # This helps balance the dataset for the ranking task.
        self.NEGATIVE_RATIO = 10

        # Context window for feature engineering (previous/next candidates)
        self.CONTEXT_WINDOW_SIZE = 2

        # ---------------------------------------------------------------------
        # 4. Feature Engineering Configuration
        # ---------------------------------------------------------------------
        self.USE_TFIDF = True
        self.USE_BM25 = True
        self.USE_JACCARD = True
        self.USE_HTML_TAGS = True

        # Max features for TF-IDF vectorizer
        self.MAX_VOCAB_SIZE = 10000

        # ---------------------------------------------------------------------
        # 5. Model Hyperparameters (LightGBM)
        # ---------------------------------------------------------------------
        self.LGBM_PARAMS = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": -1,
            "min_data_in_leaf": 20,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "lambda_l1": 0.1,
            "lambda_l2": 0.1,
            "n_jobs": self.NUM_WORKERS,
            "seed": self.SEED,
            "verbose": -1,  # Silent execution
        }

        # Training Loop Settings
        self.NUM_BOOST_ROUND = 100 if self.DEBUG else 2000
        self.EARLY_STOPPING_ROUNDS = 50
        self.VERBOSE_EVAL = 50  # Print metrics every N rounds

        # ---------------------------------------------------------------------
        # 6. Inference Thresholds
        # ---------------------------------------------------------------------
        # Probability threshold to decide if a Long Answer exists
        # If max probability < threshold, predict NULL
        self.LONG_ANSWER_THRESHOLD = 0.4

        # Short Answer Extraction Heuristics
        self.SHORT_ANSWER_MAX_TOKENS = 30

        # Initialize environment seeds
        self._set_seed()

    def _set_seed(self):
        """
        Sets random seeds for reproducibility across Python, Numpy, and Torch.
        """
        random.seed(self.SEED)
        np.random.seed(self.SEED)
        torch.manual_seed(self.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.SEED)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def get_cache_path(self, filename: str) -> str:
        """
        Helper to get full path for a cached file.

        Args:
            filename (str): Name of the file (e.g., 'train_features.parquet')

        Returns:
            str: Full path to the cached file.
        """
        return os.path.join(self.CACHE_DIR, filename)
