import os
import random
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import PowerTransformer, LabelEncoder
from sklearn.metrics import log_loss


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def clip_probabilities(preds):
    """
    Clips probabilities to the range [1e-15, 1-1e-15] to avoid log loss extremes
    as specified in the metric description.

    Args:
        preds (np.ndarray): Array of predicted probabilities.

    Returns:
        np.ndarray: Clipped probabilities.
    """
    return np.clip(preds, 1e-15, 1 - 1e-15)


def load_metadata(split):
    """
    Loads the metadata CSV file for a given split.

    Args:
        split (str): One of 'train', 'val', or 'test'.

    Returns:
        pd.DataFrame: The loaded metadata.
    """
    path = os.path.join("./metadata", f"{split}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found: {path}")
    return pd.read_csv(path)


def get_feature_columns(df):
    """
    Extracts the 192 feature columns (margin, shape, texture) from the dataframe.

    Args:
        df (pd.DataFrame): Dataframe containing feature columns.

    Returns:
        list: List of column names.
    """
    feature_cols = [
        c
        for c in df.columns
        if c.startswith("margin") or c.startswith("shape") or c.startswith("texture")
    ]
    return feature_cols


def preprocess_data(load_cached_data=True, cache_dir="./working/idea_20"):
    """
    Loads data, applies Global Gaussianization (PowerTransformer), and handles caching.
    Fits transformation on Train, applies to Val and Test.

    Args:
        load_cached_data (bool): If True, attempts to load from cache.
        cache_dir (str): Directory to store cached .npy files.

    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, test_ids, classes)
    """
    os.makedirs(cache_dir, exist_ok=True)

    cache_files = {
        "X_train": os.path.join(cache_dir, "X_train.npy"),
        "y_train": os.path.join(cache_dir, "y_train.npy"),
        "X_val": os.path.join(cache_dir, "X_val.npy"),
        "y_val": os.path.join(cache_dir, "y_val.npy"),
        "X_test": os.path.join(cache_dir, "X_test.npy"),
        "test_ids": os.path.join(cache_dir, "test_ids.npy"),
        "classes": os.path.join(cache_dir, "classes.npy"),
    }

    # Check if all cache files exist
    cache_exists = all(os.path.exists(f) for f in cache_files.values())

    if load_cached_data and cache_exists:
        print(f"Loading cached data from {cache_dir}...")
        X_train = np.load(cache_files["X_train"])
        y_train = np.load(cache_files["y_train"])
        X_val = np.load(cache_files["X_val"])
        y_val = np.load(cache_files["y_val"])
        X_test = np.load(cache_files["X_test"])
        test_ids = np.load(cache_files["test_ids"])
        classes = np.load(cache_files["classes"], allow_pickle=True)
        return X_train, y_train, X_val, y_val, X_test, test_ids, classes

    print("Processing data from scratch (Global Gaussianization)...")

    # Load Metadata
    df_train = load_metadata("train")
    df_val = load_metadata("val")
    df_test = load_metadata("test")

    # Extract Features
    feat_cols = get_feature_columns(df_train)
    X_train_raw = df_train[feat_cols].values
    X_val_raw = df_val[feat_cols].values
    X_test_raw = df_test[feat_cols].values

    test_ids = df_test["id"].values

    # Encode Targets
    le = LabelEncoder()
    y_train = le.fit_transform(df_train["species"])
    y_val = le.transform(df_val["species"])
    classes = le.classes_

    # Gaussianization (PowerTransformer)
    # Fit on Train, Transform Train/Val/Test to prevent leakage
    pt = PowerTransformer(method="yeo-johnson", standardize=True)
    X_train = pt.fit_transform(X_train_raw)
    X_val = pt.transform(X_val_raw)
    X_test = pt.transform(X_test_raw)

    # Save to Cache
    print(f"Saving processed data to {cache_dir}...")
    np.save(cache_files["X_train"], X_train)
    np.save(cache_files["y_train"], y_train)
    np.save(cache_files["X_val"], X_val)
    np.save(cache_files["y_val"], y_val)
    np.save(cache_files["X_test"], X_test)
    np.save(cache_files["test_ids"], test_ids)
    np.save(cache_files["classes"], classes)

    return X_train, y_train, X_val, y_val, X_test, test_ids, classes


def calculate_metric(y_true, y_pred_probs, classes=None):
    """
    Calculates Multi-class Log Loss with clipping and normalization.

    Args:
        y_true (np.ndarray): True class indices or labels.
        y_pred_probs (np.ndarray): Predicted probabilities (n_samples, n_classes).
        classes (list, optional): List of class labels.

    Returns:
        float: The log loss value.
    """
    # Clip probabilities
    y_pred_probs = clip_probabilities(y_pred_probs)

    # Normalize rows to sum to 1
    row_sums = y_pred_probs.sum(axis=1, keepdims=True)
    y_pred_probs = y_pred_probs / row_sums

    return log_loss(y_true, y_pred_probs, labels=classes)


def save_submission(
    test_ids, classes, probs, output_path="./submission/submission.csv"
):
    """
    Saves the submission file in the correct format.

    Args:
        test_ids (np.ndarray): Array of test image IDs.
        classes (np.ndarray): Array of class names (column headers).
        probs (np.ndarray): Predicted probabilities (n_samples, n_classes).
        output_path (str): Path to save the CSV file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Ensure probs are clipped and valid
    probs = clip_probabilities(probs)

    # Create DataFrame
    df_sub = pd.DataFrame(probs, columns=classes)
    df_sub.insert(0, "id", test_ids)

    # Save
    df_sub.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
