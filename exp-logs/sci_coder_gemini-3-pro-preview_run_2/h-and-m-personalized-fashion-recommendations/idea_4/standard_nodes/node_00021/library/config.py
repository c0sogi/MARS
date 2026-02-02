import os
import hashlib
import json
from pathlib import Path


class Config:
    """
    Central configuration for the Hybrid Multi-Source Retrieval with Latent Behavioral Embeddings pipeline.
    Handles paths, hyperparameters, and configuration-aware caching logic.
    """

    # -------------------------------------------------------------------------
    # Paths
    # -------------------------------------------------------------------------
    INPUT_DIR = Path("./input")
    METADATA_DIR = Path("./metadata")
    WORKING_DIR = Path("./working/idea_4")
    SUBMISSION_DIR = Path("./submission")

    # -------------------------------------------------------------------------
    # Global Settings
    # -------------------------------------------------------------------------
    SEED = 42

    # -------------------------------------------------------------------------
    # Data Splitting & Time Windows
    # -------------------------------------------------------------------------
    # Validation set: Last 7 days of the training data
    VAL_SIZE_DAYS = 7

    # Source A: Linear-Decay Co-occurrence
    # Restricted to recent history to minimize noise from old trends
    COOC_WINDOW_WEEKS = 4

    # Source B: Latent Behavioral Embeddings (Item2Vec)
    # Longer window to capture robust semantic relationships
    EMBED_WINDOW_WEEKS = 10

    # Source D: Recent Popularity
    # Fallback to top items from the most recent week
    POPULARITY_WINDOW_DAYS = 7

    # -------------------------------------------------------------------------
    # Retrieval Hyperparameters
    # -------------------------------------------------------------------------
    # Number of candidates to retrieve per source per customer
    TOP_K_COOC = 60
    TOP_K_EMBED = 60
    TOP_K_REPURCHASE = 60
    TOP_K_POPULARITY = 60

    # -------------------------------------------------------------------------
    # Embedding Model (Word2Vec/Item2Vec) Hyperparameters
    # -------------------------------------------------------------------------
    EMBED_DIM = 128
    W2V_WINDOW = 5
    W2V_MIN_COUNT = 2
    W2V_EPOCHS = 5
    W2V_SG = 1  # Skip-gram
    W2V_HS = 0  # Negative Sampling
    W2V_NEGATIVE = 5  # Number of negative samples

    # -------------------------------------------------------------------------
    # Ranking Model (LightGBM) Hyperparameters
    # -------------------------------------------------------------------------
    LGBM_EARLY_STOPPING_ROUNDS = 50
    LGBM_NUM_BOOST_ROUND = 2000

    LGBM_PARAMS = {
        "objective": "lambdarank",
        "metric": "map",
        "eval_at": 12,
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 100,
        "max_depth": -1,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.7,
        "bagging_freq": 1,
        "verbose": -1,
        "random_state": SEED,
        "n_jobs": 12,  # Utilize available vCPUs
    }

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------
    @classmethod
    def setup(cls):
        """Creates necessary working and submission directories."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

    @classmethod
    def get_cache_path(cls, file_name, params_dict=None):
        """
        Generates a configuration-aware cache path.

        This method ensures that if hyperparameters change (e.g., window size),
        the filename changes, preventing the loading of stale data.

        Args:
            file_name (str): The base filename (e.g., 'candidates_train.parquet').
            params_dict (dict, optional): Dictionary of parameters affecting this file.
                                          Used to generate a hash suffix.

        Returns:
            Path: The full path to the cached file in WORKING_DIR.
        """
        if params_dict is None:
            params_dict = {}

        # Create a deterministic string representation of the parameters
        # sort_keys=True ensures {a:1, b:2} hashes strictly same as {b:2, a:1}
        # default=str handles non-serializable types like Path objects
        param_str = json.dumps(params_dict, sort_keys=True, default=str)

        # Generate MD5 hash (first 10 chars are sufficient for collision avoidance)
        param_hash = hashlib.md5(param_str.encode("utf-8")).hexdigest()[:10]

        # Split extension to insert hash before it
        name, ext = os.path.splitext(file_name)

        # Construct new filename with hash
        new_filename = f"{name}_{param_hash}{ext}"

        return cls.WORKING_DIR / new_filename


# Ensure directories exist when module is imported
Config.setup()
