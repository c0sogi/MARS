import os
import pickle
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from library.config import NUMERICAL_COLS, CATEGORICAL_COLS, WORKING_DIR


class TabularTransformer:
    """
    A wrapper around sklearn's ColumnTransformer to handle
    preprocessing of tabular metadata for the skin lesion task.
    """

    def __init__(self, numerical_cols=None, categorical_cols=None):
        self.numerical_cols = (
            numerical_cols if numerical_cols is not None else NUMERICAL_COLS
        )
        self.categorical_cols = (
            categorical_cols if categorical_cols is not None else CATEGORICAL_COLS
        )

        # Pipeline for numerical features: Impute Mean -> Standard Scale
        self.num_pipeline = Pipeline(
            [("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())]
        )

        # Pipeline for categorical features: Impute 'missing' -> One-Hot Encode
        self.cat_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )

        # Combine into ColumnTransformer
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", self.num_pipeline, self.numerical_cols),
                ("cat", self.cat_pipeline, self.categorical_cols),
            ],
            verbose_feature_names_out=False,
        )

    def fit(self, df):
        """Fits the preprocessor on the provided DataFrame."""
        self.preprocessor.fit(df)
        return self

    def transform(self, df):
        """Transforms the DataFrame into a numpy array."""
        return self.preprocessor.transform(df)

    def save(self, path):
        """Saves the fitted sklearn preprocessor object to disk using pickle."""
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.preprocessor, f)
        print(f"TabularTransformer saved to {path}")

    def load(self, path):
        """Loads the fitted sklearn preprocessor object from disk."""
        with open(path, "rb") as f:
            self.preprocessor = pickle.load(f)
        return self


def preprocess_metadata(
    train_csv, val_csv, test_csv, cache_dir=WORKING_DIR, load_cached_data=True
):
    """
    Orchestrates the loading, fitting, transforming, and caching of tabular metadata.

    Args:
        train_csv (str): Path to training metadata CSV.
        val_csv (str): Path to validation metadata CSV.
        test_csv (str): Path to test metadata CSV.
        cache_dir (str): Directory to store cached .npy files and the transformer.
        load_cached_data (bool): If True, attempts to load from cache first.

    Returns:
        tuple: (X_train, X_val, X_test) as numpy arrays.
    """
    # Define cache file paths
    train_cache_path = os.path.join(cache_dir, "train_tabular.npy")
    val_cache_path = os.path.join(cache_dir, "val_tabular.npy")
    test_cache_path = os.path.join(cache_dir, "test_tabular.npy")
    transformer_path = os.path.join(cache_dir, "tabular_transformer.pkl")

    # 1. Try to load from cache
    if load_cached_data:
        if (
            os.path.exists(train_cache_path)
            and os.path.exists(val_cache_path)
            and os.path.exists(test_cache_path)
        ):
            print(f"Loading cached tabular features from {cache_dir}...")
            X_train = np.load(train_cache_path)
            X_val = np.load(val_cache_path)
            X_test = np.load(test_cache_path)
            return X_train, X_val, X_test
        else:
            print("Cached tabular data not found. Processing from scratch...")
    else:
        print("Ignoring cache. Processing metadata from scratch...")

    # 2. Load Raw Metadata
    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)
    df_test = pd.read_csv(test_csv)

    # 3. Initialize and Fit Transformer
    transformer = TabularTransformer()
    print("Fitting TabularTransformer on training data...")
    transformer.fit(df_train)

    # 4. Transform Data
    print("Transforming metadata...")
    X_train = transformer.transform(df_train)
    X_val = transformer.transform(df_val)
    X_test = transformer.transform(df_test)

    # 5. Save to Cache
    os.makedirs(cache_dir, exist_ok=True)
    np.save(train_cache_path, X_train)
    np.save(val_cache_path, X_val)
    np.save(test_cache_path, X_test)
    transformer.save(transformer_path)

    print(f"Tabular features cached to {cache_dir}")

    return X_train, X_val, X_test
