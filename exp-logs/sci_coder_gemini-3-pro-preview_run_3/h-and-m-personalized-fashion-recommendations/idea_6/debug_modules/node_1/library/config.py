import os
import torch
from pathlib import Path


class Config:
    """
    Global configuration for the Multi-Temporal Cascade System with Dynamic Ensemble Ranking.
    Centralizes all hyperparameters, file paths, and execution settings.
    """

    # =========================================================================
    # Global Execution Settings
    # =========================================================================
    SEED = 42
    DEBUG = False  # Set to True to run on a smaller subset of data for debugging
    DEBUG_SAMPLE_SIZE = 5000  # Number of customers/articles to sample in debug mode
    NUM_WORKERS = 12  # Number of CPU cores available
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # =========================================================================
    # Directory Structure
    # =========================================================================
    INPUT_DIR = Path("./input")
    METADATA_DIR = Path("./metadata")
    WORKING_DIR = Path("./working/idea_6")
    SUBMISSION_DIR = Path("./submission")

    # Ensure output directories exist
    WORKING_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Data Paths
    # =========================================================================
    # Raw Input Files (Read-Only)
    ARTICLES_CSV = INPUT_DIR / "articles.csv"
    CUSTOMERS_CSV = INPUT_DIR / "customers.csv"
    TRANSACTIONS_CSV = INPUT_DIR / "transactions_train.csv"
    SAMPLE_SUBMISSION_CSV = INPUT_DIR / "sample_submission.csv"
    IMAGES_DIR = INPUT_DIR / "images"

    # Metadata Files (Generated Splits)
    TRAIN_METADATA = METADATA_DIR / "train_metadata.parquet"
    VAL_METADATA = METADATA_DIR / "val_metadata.parquet"
    TEST_METADATA = METADATA_DIR / "test_metadata.parquet"

    # =========================================================================
    # Caching Paths (Intermediate Artifacts)
    # =========================================================================
    # Preprocessing Mappings & Embeddings
    CACHE_ARTICLE_MAP = WORKING_DIR / "article_map.npy"  # Map article_id -> int index
    CACHE_CUSTOMER_MAP = (
        WORKING_DIR / "customer_map.npy"
    )  # Map customer_id -> int index
    CACHE_IMAGE_EMBEDDINGS = WORKING_DIR / "image_embeddings.npy"

    # Sparse Graph Structures (Stage 1)
    CACHE_GRAPH_SHORT = WORKING_DIR / "graph_short_term.npz"  # Short-term transitions
    CACHE_GRAPH_LONG = WORKING_DIR / "graph_long_term.npz"  # Long-term transitions
    CACHE_GRAPH_VISUAL = WORKING_DIR / "graph_visual.npz"  # Visual similarity graph
    CACHE_USER_HISTORY = WORKING_DIR / "user_history.npz"  # User purchase history

    # Ranker Datasets (Stage 2)
    # These store the features and labels for the LightGBM model
    CACHE_RANKER_TRAIN = WORKING_DIR / "ranker_train.parquet"
    CACHE_RANKER_VAL = WORKING_DIR / "ranker_val.parquet"
    CACHE_RANKER_TEST = WORKING_DIR / "ranker_test.parquet"

    # Model Checkpoints
    MODEL_LGBM_FILE = WORKING_DIR / "lgbm_ranker.txt"

    # Final Output
    SUBMISSION_PATH = SUBMISSION_DIR / "submission.csv"

    # =========================================================================
    # Stage 1: Multi-Temporal Retrieval Hyperparameters
    # =========================================================================
    # Temporal Window Definitions (in Days)
    # Short-term: Captures immediate trends (Last 4 weeks)
    SHORT_TERM_WINDOW = 28
    # Long-term: Captures stable style preferences (Weeks 5-20)
    LONG_TERM_WINDOW = 112

    # Visual Graph Construction
    VISUAL_KNN_K = 20  # Number of neighbors for visual similarity
    IMAGE_MODEL = "resnet18"  # Pre-trained architecture for embeddings
    IMAGE_SIZE = 224
    IMAGE_BATCH_SIZE = 128

    # Candidate Generation
    RETRIEVAL_TOP_K = 100  # Number of items to retrieve per graph (Short/Long/Visual)
    FINAL_CANDIDATE_K = 100  # Target number of unique candidates per user for Ranking

    # User History Aggregation
    HISTORY_DECAY_FACTOR = 0.98  # Decay weight for older user interactions

    # =========================================================================
    # Stage 2: Dynamic Ensemble Ranking Hyperparameters
    # =========================================================================
    # Sliding Window Strategy for Training
    # We generate multiple training samples per user by sliding back in time
    NUM_SLIDING_WINDOWS = 3
    SLIDING_WINDOW_SIZE = 7  # Size of the target window (prediction horizon)
    SLIDING_WINDOW_STEP = 7  # Step size between windows

    # LightGBM Model Parameters
    LGBM_PARAMS = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "eval_at": [12],
        "boosting_type": "gbdt",
        "n_estimators": 1500,
        "learning_rate": 0.05,
        "num_leaves": 64,
        "max_depth": -1,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "random_state": SEED,
        "n_jobs": NUM_WORKERS,
        "verbose": -1,
        "force_col_wise": True,
    }

    EARLY_STOPPING_ROUNDS = 50

    # =========================================================================
    # Inference & Submission
    # =========================================================================
    TOP_K_PREDICTION = 12  # Number of items to predict per user

    # Fallback prediction (Popular items) if no history/candidates found
    # (Pre-computed popular items from training data)
    FALLBACK_PREDICTION = "0706016001 0706016002 0372860001 0610776002 0759871002 0448509014 0372860002 0579541001 0706016003 0573085028 0751471001 0673677002"

    @classmethod
    def get_lgbm_params(cls):
        """Helper to get a copy of LGBM params to prevent mutation."""
        return cls.LGBM_PARAMS.copy()
