import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Constants
METADATA_DIR = "./metadata"
CACHE_DIR = "./working/idea_18"
INPUT_DIR = "./input"
RANDOM_SEED = 42

# Set random seed for reproducibility
np.random.seed(RANDOM_SEED)


def preprocess_labels(y_train, y_val):
    """
    Encodes species labels using LabelEncoder.
    Fits on the union of train and val labels to ensure all classes are covered.
    """
    le = LabelEncoder()
    # Fit on all unique species found in train and val
    all_species = np.unique(np.concatenate([y_train, y_val]))
    le.fit(all_species)

    y_train_enc = le.transform(y_train)
    y_val_enc = le.transform(y_val)

    return y_train_enc, y_val_enc, le.classes_


def load_dataset(load_cached_data=True):
    """
    Loads the dataset, using caching to speed up subsequent runs.
    Returns:
        X_train, y_train, X_val, y_val, X_test, test_ids, classes
    """
    # Ensure cache directory exists
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Define cache file paths
    cache_paths = {
        "X_train": os.path.join(CACHE_DIR, "X_train.npy"),
        "y_train": os.path.join(CACHE_DIR, "y_train.npy"),
        "X_val": os.path.join(CACHE_DIR, "X_val.npy"),
        "y_val": os.path.join(CACHE_DIR, "y_val.npy"),
        "X_test": os.path.join(CACHE_DIR, "X_test.npy"),
        "test_ids": os.path.join(CACHE_DIR, "test_ids.npy"),
        "classes": os.path.join(CACHE_DIR, "classes.npy"),
    }

    # Check if cache exists
    cache_exists = all(os.path.exists(p) for p in cache_paths.values())

    if load_cached_data and cache_exists:
        X_train = np.load(cache_paths["X_train"])
        y_train = np.load(cache_paths["y_train"])
        X_val = np.load(cache_paths["X_val"])
        y_val = np.load(cache_paths["y_val"])
        X_test = np.load(cache_paths["X_test"])
        test_ids = np.load(cache_paths["test_ids"])
        classes = np.load(cache_paths["classes"], allow_pickle=True)
        return X_train, y_train, X_val, y_val, X_test, test_ids, classes

    # Load from metadata CSVs
    train_df = pd.read_csv(os.path.join(METADATA_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(METADATA_DIR, "val.csv"))
    test_df = pd.read_csv(os.path.join(METADATA_DIR, "test.csv"))

    # Identify feature columns (margin, shape, texture)
    feature_cols = [
        c for c in train_df.columns if c.startswith(("margin", "shape", "texture"))
    ]

    # Extract features
    X_train = train_df[feature_cols].values.astype(np.float32)
    X_val = val_df[feature_cols].values.astype(np.float32)
    X_test = test_df[feature_cols].values.astype(np.float32)

    # Extract IDs
    test_ids = test_df["id"].values

    # Process Labels
    y_train_raw = train_df["species"].values
    y_val_raw = val_df["species"].values
    y_train, y_val, classes = preprocess_labels(y_train_raw, y_val_raw)

    # Save to cache
    np.save(cache_paths["X_train"], X_train)
    np.save(cache_paths["y_train"], y_train)
    np.save(cache_paths["X_val"], X_val)
    np.save(cache_paths["y_val"], y_val)
    np.save(cache_paths["X_test"], X_test)
    np.save(cache_paths["test_ids"], test_ids)
    np.save(cache_paths["classes"], classes)

    return X_train, y_train, X_val, y_val, X_test, test_ids, classes


def get_stratified_split(load_cached_data=True):
    """
    Returns the stratified train/val split.
    Since the metadata files are already split stratifiedly, this function
    wraps load_dataset to return the relevant arrays.
    """
    X_train, y_train, X_val, y_val, _, _, _ = load_dataset(
        load_cached_data=load_cached_data
    )
    return X_train, y_train, X_val, y_val


def save_submission(
    predictions, test_ids, classes, output_path="./submission/submission.csv"
):
    """
    Formats and saves the submission file.
    Args:
        predictions: numpy array of shape (n_samples, n_classes) with probabilities.
        test_ids: numpy array of shape (n_samples,) with image IDs.
        classes: list or array of class names corresponding to prediction columns.
        output_path: path to save the csv.
    """
    # Clip probabilities to avoid log loss extremes as per metric description
    # max(min(p, 1-10^-15), 10^-15)
    epsilon = 1e-15
    predictions = np.clip(predictions, epsilon, 1 - epsilon)

    # Create DataFrame
    df_submission = pd.DataFrame(predictions, columns=classes)
    df_submission.insert(0, "id", test_ids)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to CSV
    df_submission.to_csv(output_path, index=False)
