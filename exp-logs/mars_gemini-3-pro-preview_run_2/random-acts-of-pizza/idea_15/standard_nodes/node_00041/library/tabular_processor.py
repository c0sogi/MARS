import os
import numpy as np
import pandas as pd
from library.config import Config


class TabularProcessor:
    """
    Handles the extraction, cleaning, and preparation of numerical tabular features.
    Ensures data is in the correct format (numpy array of float32) for the machine learning pipeline.
    """

    def __init__(self):
        """
        Initialize the processor with feature definitions from Config.
        """
        self.numeric_cols = Config.NUMERIC_COLS

    def process(
        self, df: pd.DataFrame, cache_path: str = None, load_cached_data: bool = True
    ) -> np.ndarray:
        """
        Extracts numerical features from the DataFrame, handles missing values,
        and returns a numpy array. Supports caching to .npy files.

        Args:
            df (pd.DataFrame): Input DataFrame containing the raw features.
            cache_path (str, optional): Path to save/load the processed numpy array.
            load_cached_data (bool): Whether to attempt loading from cache.

        Returns:
            np.ndarray: Processed numerical features matrix of shape (n_samples, n_features).
        """
        # 1. Caching Mechanism
        if cache_path and load_cached_data and os.path.exists(cache_path):
            print(f"Loading tabular features from cache: {cache_path}")
            try:
                data = np.load(cache_path)
                # Verify shape consistency
                if data.shape[0] == len(df):
                    return data
                else:
                    print(
                        f"Cached tabular data shape {data.shape} does not match DataFrame length {len(df)}. Recomputing..."
                    )
            except Exception as e:
                print(f"Error loading cache: {e}. Recomputing...")

        print(f"Processing tabular features for {len(df)} samples...")

        # 2. Validation
        missing_cols = [col for col in self.numeric_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"The following required numerical columns are missing from the DataFrame: {missing_cols}"
            )

        # 3. Extraction
        # Select columns and convert to float32 for memory efficiency and compatibility
        X_tab = df[self.numeric_cols].values.astype(np.float32)

        # 4. Cleaning / Imputation
        # While DataLoader handles basic filling, we ensure no NaNs or Infs remain.
        # We use 0.0 as the fill value, which is appropriate for counts and normalized timestamps.
        if np.isnan(X_tab).any() or np.isinf(X_tab).any():
            print("NaNs or Infinite values detected in tabular data. Cleaning...")
            X_tab = np.nan_to_num(X_tab, nan=0.0, posinf=0.0, neginf=0.0)

        # 5. Save to Cache
        if cache_path:
            print(f"Saving tabular features to {cache_path}...")
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            np.save(cache_path, X_tab)

        return X_tab
