import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
from library.config import Config
from library.utils import setup_logger, timer


class DualBranchPreprocessor:
    """
    Preprocessor for the Dual-Branch Stacking Ensemble.
    Branch A (Linear): Uses full SentenceTransformer embeddings + Scaled Metadata.
    Branch B (Tree): Uses PCA-reduced embeddings + Scaled Metadata.
    """

    def __init__(self):
        self.logger = setup_logger("preprocessor")
        self.scaler = RobustScaler()
        self.pca = PCA(n_components=Config.N_PCA_COMPONENTS, random_state=Config.SEED)
        self.text_model = None  # Lazy load to avoid overhead if loading from cache

    def _load_text_model(self):
        if self.text_model is None:
            self.logger.info(f"Loading SentenceTransformer: {Config.TEXT_MODEL_NAME}")
            self.text_model = SentenceTransformer(Config.TEXT_MODEL_NAME)

    def _get_text_embeddings(self, df):
        self._load_text_model()
        # Combine title and edit-aware text
        # Fill NaNs with empty string
        titles = df["request_title"].fillna("").astype(str)
        texts = df["request_text_edit_aware"].fillna("").astype(str)

        # Concatenate with a space separator
        combined_text = (titles + " " + texts).tolist()

        # Encode to dense vectors
        embeddings = self.text_model.encode(
            combined_text, batch_size=32, show_progress_bar=False, convert_to_numpy=True
        )
        return embeddings

    def _get_numerical_features(self, df):
        # 1. Base Numerical Features
        # Extract numerical columns defined in Config
        X_base = df[Config.NUMERICAL_COLS].fillna(0).values

        # 2. Derived Features
        # Text Lengths (Proxy for effort/desperation)
        titles = df["request_title"].fillna("").astype(str)
        texts = df["request_text_edit_aware"].fillna("").astype(str)

        len_title = titles.apply(len).values.reshape(-1, 1)
        len_text = texts.apply(len).values.reshape(-1, 1)

        # Temporal Features (Cyclic Encoding)
        # unix_timestamp_of_request is in seconds. Fill NaNs with 0 (1970) to prevent errors.
        timestamps = pd.to_datetime(df["unix_timestamp_of_request"].fillna(0), unit="s")

        # Hour of Day (0-23)
        hour = timestamps.dt.hour
        hour_sin = np.sin(2 * np.pi * hour / 24).values.reshape(-1, 1)
        hour_cos = np.cos(2 * np.pi * hour / 24).values.reshape(-1, 1)

        # Day of Week (0-6)
        day = timestamps.dt.dayofweek
        day_sin = np.sin(2 * np.pi * day / 7).values.reshape(-1, 1)
        day_cos = np.cos(2 * np.pi * day / 7).values.reshape(-1, 1)

        # Month (1-12) - Capture seasonality
        month = timestamps.dt.month
        month_sin = np.sin(2 * np.pi * (month - 1) / 12).values.reshape(-1, 1)
        month_cos = np.cos(2 * np.pi * (month - 1) / 12).values.reshape(-1, 1)

        # 3. Concatenate All
        X_combined = np.hstack(
            [
                X_base,
                len_title,
                len_text,
                hour_sin,
                hour_cos,
                day_sin,
                day_cos,
                month_sin,
                month_cos,
            ]
        )

        return X_combined

    def fit(self, df_train):
        """
        Fits the PCA and Scaler on the training data.
        """
        with timer("Fit Preprocessor", self.logger):
            # 1. Text Embeddings
            self.logger.info("Generating training embeddings for PCA fit...")
            train_embeddings = self._get_text_embeddings(df_train)

            # 2. Fit PCA
            self.logger.info(
                f"Fitting PCA with n_components={Config.N_PCA_COMPONENTS}..."
            )
            self.pca.fit(train_embeddings)

            # 3. Numericals
            self.logger.info("Fitting RobustScaler on numerical features...")
            X_num = self._get_numerical_features(df_train)
            self.scaler.fit(X_num)

    def transform(self, df):
        """
        Transforms the dataframe into features for both branches.
        Returns a dictionary with 'linear' and 'tree' feature arrays.
        """
        # 1. Text
        embeddings = self._get_text_embeddings(df)

        # 2. PCA Transform (for Tree Branch)
        embeddings_pca = self.pca.transform(embeddings)

        # 3. Numericals
        X_num = self._get_numerical_features(df)
        X_num_scaled = self.scaler.transform(X_num)

        # 4. Construct Feature Sets
        # Linear Branch: Raw Embeddings (384) + Scaled Metadata
        X_linear = np.hstack([embeddings, X_num_scaled])

        # Tree Branch: PCA Embeddings (32) + Scaled Metadata
        X_tree = np.hstack([embeddings_pca, X_num_scaled])

        return {
            "linear": X_linear.astype(np.float32),
            "tree": X_tree.astype(np.float32),
        }


def process_data(df_train, df_val, df_test, load_cached_data=True):
    """
    Main function to process data for the Dual-Branch model.
    Handles caching of numpy arrays to avoid re-computation.

    Args:
        df_train, df_val, df_test: Pandas DataFrames containing the data.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        dict: Dictionary containing processed numpy arrays for all splits.
    """
    logger = setup_logger("feature_engineering")

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Define cache file paths
    # We use .npy for efficient storage of dense arrays
    cache_files = {
        "train_linear": os.path.join(Config.WORKING_DIR, "X_train_linear.npy"),
        "train_tree": os.path.join(Config.WORKING_DIR, "X_train_tree.npy"),
        "y_train": os.path.join(Config.WORKING_DIR, "y_train.npy"),
        "val_linear": os.path.join(Config.WORKING_DIR, "X_val_linear.npy"),
        "val_tree": os.path.join(Config.WORKING_DIR, "X_val_tree.npy"),
        "y_val": os.path.join(Config.WORKING_DIR, "y_val.npy"),
        "test_linear": os.path.join(Config.WORKING_DIR, "X_test_linear.npy"),
        "test_tree": os.path.join(Config.WORKING_DIR, "X_test_tree.npy"),
    }

    # Check if all cache files exist
    cache_exists = all(os.path.exists(path) for path in cache_files.values())

    if load_cached_data and cache_exists:
        with timer("Load Features from Cache", logger):
            data = {}
            for key, path in cache_files.items():
                data[key] = np.load(path)
            logger.info("Successfully loaded features from cache.")
            return data

    # If not cached or reload forced, process from scratch
    with timer("Feature Engineering Pipeline", logger):
        preprocessor = DualBranchPreprocessor()

        # Fit on Train
        preprocessor.fit(df_train)

        # Transform all sets
        data_map = {"train": df_train, "val": df_val, "test": df_test}

        output_data = {}

        for name, df in data_map.items():
            logger.info(f"Transforming {name} set...")
            features = preprocessor.transform(df)

            # Store features in memory
            output_data[f"{name}_linear"] = features["linear"]
            output_data[f"{name}_tree"] = features["tree"]

            # Save features to cache
            np.save(cache_files[f"{name}_linear"], features["linear"])
            np.save(cache_files[f"{name}_tree"], features["tree"])

            # Handle Targets (only for train and val)
            if "requester_received_pizza" in df.columns:
                y = df["requester_received_pizza"].values.astype(int)
                output_data[f"y_{name}"] = y
                np.save(cache_files[f"y_{name}"], y)

        logger.info("Feature engineering complete. Cache saved.")
        return output_data
