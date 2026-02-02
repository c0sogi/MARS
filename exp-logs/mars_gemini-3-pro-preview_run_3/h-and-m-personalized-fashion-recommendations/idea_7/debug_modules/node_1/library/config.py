import os
from pathlib import Path


class Config:
    """
    Global configuration for the Dual-Graph Vectorized Cascade system (Idea 7).
    Defines paths, hyperparameters, and model configurations.
    """

    # ==========================================
    # Directories
    # ==========================================
    INPUT_DIR = Path("./input")
    METADATA_DIR = Path("./metadata")
    WORKING_DIR = Path("./working/idea_7")
    SUBMISSION_DIR = Path("./submission")

    # ==========================================
    # Input Data Paths (from Metadata Generation)
    # ==========================================
    TRAIN_DATA_PATH = METADATA_DIR / "train_metadata.parquet"
    VAL_DATA_PATH = METADATA_DIR / "val_metadata.parquet"
    TEST_DATA_PATH = METADATA_DIR / "test_metadata.parquet"

    ARTICLES_PATH = INPUT_DIR / "articles.csv"
    CUSTOMERS_PATH = INPUT_DIR / "customers.csv"
    SAMPLE_SUBMISSION_PATH = INPUT_DIR / "sample_submission.csv"

    # ==========================================
    # Cache Paths (Intermediate Artifacts)
    # ==========================================
    # Mappings
    CACHE_ARTICLE_ID_MAP = WORKING_DIR / "article_id_map.npy"
    CACHE_CUSTOMER_ID_MAP = WORKING_DIR / "customer_id_map.npy"

    # Embeddings & Graphs
    CACHE_ARTICLE_EMBEDDINGS = WORKING_DIR / "article_embeddings.npy"
    CACHE_VISUAL_GRAPH = WORKING_DIR / "visual_graph.npz"
    CACHE_SEQUENTIAL_GRAPH = WORKING_DIR / "sequential_graph.npz"

    # Processed Features & Datasets
    CACHE_USER_HISTORY = WORKING_DIR / "user_history.npz"
    CACHE_RANKER_TRAIN = WORKING_DIR / "ranker_train.parquet"
    CACHE_RANKER_VAL = WORKING_DIR / "ranker_val.parquet"

    # Final Output
    SUBMISSION_PATH = SUBMISSION_DIR / "submission.csv"

    # ==========================================
    # Global Settings
    # ==========================================
    SEED = 42
    NUM_WORKERS = 12  # Matches available vCPUs

    # ==========================================
    # Stage 1: Retrieval Hyperparameters
    # ==========================================
    RETRIEVAL_TOP_K = 100  # Number of candidates to retrieve per user
    RETRIEVAL_HISTORY_WEEKS = 10  # Weeks of history to use for Sequential Graph
    RETRIEVAL_VISUAL_WEIGHT = 0.5  # Weight (lambda) for Visual Graph scores
    RETRIEVAL_REPURCHASE_WEIGHT = 1.5  # Weight (alpha) for User History (Repurchase)

    # Visual Graph Construction
    VISUAL_KNN_K = 20  # Number of neighbors in Visual Graph
    IMAGE_SIZE = (224, 224)
    IMAGE_MODEL_NAME = "resnet18"
    IMAGE_BATCH_SIZE = 128

    # ==========================================
    # Stage 2: Ranking Hyperparameters
    # ==========================================
    # LightGBM Configuration
    LGBM_PARAMS = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "eval_at": 12,
        "boosting_type": "gbdt",
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "verbose": -1,
        "random_state": SEED,
        "n_jobs": NUM_WORKERS,
        "early_stopping_rounds": 50,
    }

    # Sliding Window Strategy for Ranker Training
    # We train on weeks [N-9...N-1] and target week N
    RANKER_WINDOW_WEEKS = 10

    @staticmethod
    def setup():
        """
        Initialize the working directories.
        Should be called at the start of the pipeline.
        """
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
