import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from library import config, features


def load_and_preprocess(split_name, load_cached_data=True):
    """
    Loads and preprocesses the dataset for a given split (train, val, test).
    Delegates to library.features to handle feature engineering and caching.

    Args:
        split_name (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to load from cache if available.

    Returns:
        pd.DataFrame: The processed dataframe.
    """
    return features.get_processed_data(split_name, load_cached_data=load_cached_data)


def get_stratified_folds(y, n_folds=config.N_FOLDS):
    """
    Generates stratified K-Fold indices.

    Args:
        y (array-like): Target variable.
        n_folds (int): Number of folds.

    Returns:
        generator: A generator yielding (train_index, test_index).
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=config.SEED)
    # X is ignored by StratifiedKFold.split, providing zeros as placeholder
    return skf.split(np.zeros(len(y)), y)


def get_X_y(df):
    """
    Separates features (X) and target (y) from the dataframe.
    Drops the ID column and the Target column from X.

    Args:
        df (pd.DataFrame): The dataframe.

    Returns:
        tuple: (X, y) where X is a DataFrame and y is a Series (or None if target missing).
    """
    # Drop ID column if present
    X = df.drop(columns=[config.ID_COL], errors="ignore")

    y = None
    if config.TARGET_COL in df.columns:
        y = df[config.TARGET_COL]
        X = X.drop(columns=[config.TARGET_COL], errors="ignore")

    return X, y


def create_augmented_dataset(train_df, test_df, test_probs, threshold=None):
    """
    Augments the training dataframe with high-confidence pseudo-labels from the test set.

    Args:
        train_df (pd.DataFrame): The original training data.
        test_df (pd.DataFrame): The test data (features).
        test_probs (np.ndarray): Predicted probabilities for the test data (shape: [n_test, n_classes]).
        threshold (float): Confidence threshold for pseudo-labeling.
                           Defaults to config.PSEUDO_LABEL_THRESHOLD.

    Returns:
        pd.DataFrame: The augmented dataframe containing original train data + pseudo-labeled test data.
    """
    if threshold is None:
        threshold = config.PSEUDO_LABEL_THRESHOLD

    # Calculate max probability and predicted class for each test sample
    max_probs = np.max(test_probs, axis=1)
    preds = np.argmax(test_probs, axis=1)

    # Identify samples exceeding the confidence threshold
    high_conf_mask = max_probs > threshold
    high_conf_indices = np.where(high_conf_mask)[0]

    if len(high_conf_indices) == 0:
        print("No test samples met the pseudo-label confidence threshold.")
        return train_df.copy()

    print(
        f"Augmenting training data with {len(high_conf_indices)} pseudo-labeled samples (Threshold: {threshold})."
    )

    # Create the pseudo-labeled subset
    pseudo_subset = test_df.iloc[high_conf_indices].copy()
    pseudo_subset[config.TARGET_COL] = preds[high_conf_indices]

    # Ensure the target column type matches the training data
    if config.TARGET_COL in train_df.columns:
        target_dtype = train_df[config.TARGET_COL].dtype
        pseudo_subset[config.TARGET_COL] = pseudo_subset[config.TARGET_COL].astype(
            target_dtype
        )

    # Concatenate original training data with pseudo-labeled data
    augmented_df = pd.concat([train_df, pseudo_subset], axis=0, ignore_index=True)

    return augmented_df
