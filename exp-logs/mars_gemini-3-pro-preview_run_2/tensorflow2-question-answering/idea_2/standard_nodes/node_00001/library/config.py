import os


class PathConfig:
    # Root directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_2"
    SUBMISSION_DIR = "./submission"

    # Input Files
    TRAIN_JSONL = os.path.join(INPUT_DIR, "simplified-nq-train.jsonl")
    TEST_JSONL = os.path.join(INPUT_DIR, "simplified-nq-test.jsonl")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Metadata Files
    TRAIN_META = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_META = os.path.join(METADATA_DIR, "validation_metadata.csv")
    TEST_META = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Cache Files (Parquet/NPY for intermediate data)
    IDF_CACHE = os.path.join(WORKING_DIR, "idf_stats.npy")
    TRAIN_FEATURES_CACHE = os.path.join(WORKING_DIR, "train_features.parquet")
    VAL_FEATURES_CACHE = os.path.join(WORKING_DIR, "val_features.parquet")
    TEST_FEATURES_CACHE = os.path.join(WORKING_DIR, "test_features.parquet")

    # Model File
    MODEL_FILE = os.path.join(WORKING_DIR, "lgbm_ranker.txt")

    # Output
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    @classmethod
    def ensure_directories(cls):
        """Creates necessary working and submission directories."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


class FeatureConfig:
    # Text Preprocessing
    USE_STEMMING = True
    REMOVE_STOPWORDS = True

    # BM25 Hyperparameters
    BM25_K1 = 1.2
    BM25_B = 0.75

    # Feature Generation
    # If True, computes features for all candidates.
    # If False, might limit to top N candidates by simple overlap to save memory (optional optimization)
    PROCESS_ALL_CANDIDATES = True


class ModelConfig:
    # Reproducibility
    SEED = 42

    # Training Data Sampling
    # Ratio of negative examples (incorrect candidates) to positive examples (correct candidates)
    # to keep during training. Helps with class imbalance.
    NEG_SAMPLING_RATIO = 5

    # LightGBM Hyperparameters
    LGBM_PARAMS = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 0.8,
        "subsample_freq": 5,
        "colsample_bytree": 0.9,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "n_jobs": 12,  # Use available vCPUs
        "verbose": -1,
        "seed": SEED,
    }

    # Training Loop
    NUM_BOOST_ROUND = 2000
    EARLY_STOPPING_ROUNDS = 50
    VERBOSE_EVAL = 50

    # Inference Thresholds
    # Minimum probability score required to predict a long answer.
    # If max probability < threshold, predict BLANK.
    LONG_CONFIDENCE_THRESHOLD = 0.45

    # Minimum Jaccard similarity required to select a sentence as a short answer.
    # If max similarity < threshold, predict BLANK for short answer.
    SHORT_CONFIDENCE_THRESHOLD = 0.35


# Ensure directories exist upon module import
PathConfig.ensure_directories()
