import numpy as np
import pandas as pd
from library.config import Config
from library.embeddings import extract_features


def prepare_features(split="train", sample_size=None, load_cached_data=True):
    """
    Orchestrates the generation of the final feature matrix and target vectors.

    This function calls the embedding extraction pipeline to get raw features,
    performs necessary preprocessing (like One-Hot Encoding for spacegroups),
    and transforms the targets for training.

    Args:
        split (str): Dataset split to process ('train', 'val', 'test').
        sample_size (int, optional): Number of samples to process (for debugging).
        load_cached_data (bool): Whether to load intermediate feature files from cache.

    Returns:
        X (pd.DataFrame): The feature matrix ready for the model.
        y (pd.DataFrame or None): The log-transformed target variables (None for test split).
        ids (pd.Series): The sequence of IDs corresponding to the rows in X.
    """
    # 1. Load or Compute Combined Features
    # This pulls GNN embeddings, physical descriptors, and tabular metadata
    print(f"Preparing features for split: {split}")
    df = extract_features(
        split=split, sample_size=sample_size, load_cached_data=load_cached_data
    )

    # 2. Extract IDs
    if "id" in df.columns:
        ids = df["id"]
        # Drop ID from features, it's not a predictor
        X = df.drop(columns=["id"])
    else:
        # Fallback if ID is not a column (unlikely given metadata structure)
        ids = df.index.to_series()
        X = df.copy()

    # 3. Handle Targets
    y = None
    target_cols = Config.TARGET_COLS

    # Check if target columns exist in the dataframe
    # They should exist for 'train' and 'val', but not 'test'
    if set(target_cols).issubset(X.columns):
        y_raw = X[target_cols]

        # Apply Log Transformation if configured (for RMSLE optimization)
        if Config.LOG_TRANSFORM_TARGETS:
            # Clip to 0 to avoid log of negative numbers (though energy >= 0 usually)
            y_raw = y_raw.clip(lower=0)
            y = np.log1p(y_raw)
        else:
            y = y_raw

        # Drop targets from the feature matrix X
        X = X.drop(columns=target_cols)

    # 4. Feature Engineering: One-Hot Encoding for Spacegroup
    # The spacegroup is a categorical variable (1-230).
    # We convert it to string to ensure get_dummies treats it as categorical.
    if "spacegroup" in X.columns:
        X["spacegroup"] = X["spacegroup"].astype(str)
        X = pd.get_dummies(X, columns=["spacegroup"], prefix="sg")

    # 5. Cleanup
    # Remove file_path if it exists (it's metadata, not a feature)
    if "file_path" in X.columns:
        X = X.drop(columns=["file_path"])

    print(f"Feature preparation complete. X shape: {X.shape}")
    return X, y, ids
