import os
import numpy as np
import pandas as pd
import scipy.sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from library.config import Config
from library.data_loader import load_dataset
from library.utils import set_seed


class PizzaPreprocessor:
    """
    Handles feature engineering: Text vectorization and numerical feature extraction.
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(**Config.VECTORIZER_PARAMS)
        self.scaler = StandardScaler()
        self.num_cols = Config.NUMERICAL_COLS

    def fit(self, train_df: pd.DataFrame):
        """
        Fits the TF-IDF vectorizer and numerical scaler.
        """
        # Fit vectorizer on the combined text column
        self.vectorizer.fit(train_df["combined_text"])

        # Fit scaler on numerical columns
        self.scaler.fit(train_df[self.num_cols].values.astype(np.float32))
        return self

    def transform(self, df: pd.DataFrame):
        """
        Transforms the dataframe into a sparse feature matrix.
        """
        # 1. Text Features (Sparse)
        text_features = self.vectorizer.transform(df["combined_text"])

        # 2. Numerical Features (Dense & Scaled)
        num_features = self.scaler.transform(
            df[self.num_cols].values.astype(np.float32)
        )

        # 3. Combine
        X = scipy.sparse.hstack([text_features, num_features])
        return X


def load_processed_data(
    load_cached_data: bool = True, debug_size: int = Config.DEBUG_SAMPLE_SIZE
):
    """
    Loads processed sparse matrices and targets.
    Uses caching to avoid re-computing TF-IDF and stacking.
    """
    set_seed(Config.RANDOM_SEED)

    # Define paths for cached files
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    paths = {
        "X_train": os.path.join(cache_dir, "X_train.npz"),
        "y_train": os.path.join(cache_dir, "y_train.npy"),
        "X_val": os.path.join(cache_dir, "X_val.npz"),
        "y_val": os.path.join(cache_dir, "y_val.npy"),
        "X_test": os.path.join(cache_dir, "X_test.npz"),
        "test_ids": os.path.join(cache_dir, "test_ids.npy"),
    }

    # Check if all files exist
    all_files_exist = all(os.path.exists(p) for p in paths.values())

    if load_cached_data and all_files_exist:
        print(f"Loading processed data from {cache_dir}...")
        X_train = scipy.sparse.load_npz(paths["X_train"])
        y_train = np.load(paths["y_train"])
        X_val = scipy.sparse.load_npz(paths["X_val"])
        y_val = np.load(paths["y_val"])
        X_test = scipy.sparse.load_npz(paths["X_test"])
        test_ids = np.load(paths["test_ids"], allow_pickle=True)  # IDs are strings
        return X_train, y_train, X_val, y_val, X_test, test_ids

    # If not cached or reload forced, process from scratch
    print("Processing data from scratch...")

    # Load DataFrames (data_loader handles raw loading and basic cleaning)
    train_df, val_df, test_df = load_dataset(
        load_cached_data=load_cached_data, debug_size=debug_size
    )

    # Initialize and Fit Preprocessor
    preprocessor = PizzaPreprocessor()
    preprocessor.fit(train_df)

    # Transform
    print("Transforming features...")
    X_train = preprocessor.transform(train_df)
    X_val = preprocessor.transform(val_df)
    X_test = preprocessor.transform(test_df)

    # Extract Targets and IDs
    y_train = train_df[Config.TARGET_COL].values.astype(int)
    y_val = val_df[Config.TARGET_COL].values.astype(int)
    test_ids = test_df[Config.ID_COL].values

    # Save to cache
    print(f"Saving processed data to {cache_dir}...")
    scipy.sparse.save_npz(paths["X_train"], X_train)
    np.save(paths["y_train"], y_train)
    scipy.sparse.save_npz(paths["X_val"], X_val)
    np.save(paths["y_val"], y_val)
    scipy.sparse.save_npz(paths["X_test"], X_test)
    np.save(paths["test_ids"], test_ids)

    return X_train, y_train, X_val, y_val, X_test, test_ids
