import os
import random
import numpy as np
import pandas as pd
import torch
from library.config import (
    TRAIN_DATA_PATH,
    VAL_DATA_PATH,
    TEST_DATA_PATH,
    WORKING_DIR,
    SAMPLE_SUBMISSION_PATH,
    RANDOM_SEED,
)


def set_seed(seed=RANDOM_SEED):
    """
    Sets the random seed for reproducibility across various libraries.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def load_dataset(split="train", load_cached=True, cache_dir=WORKING_DIR):
    """
    Loads the dataset for a specific split (train, val, test).
    Implements caching to .npy files to speed up subsequent loads.

    Args:
        split (str): One of 'train', 'val', 'test'.
        load_cached (bool): Whether to try loading from cache.
        cache_dir (str): Directory to store/load cached files.

    Returns:
        tuple: (X, y, ids)
            X (np.ndarray): Feature matrix (n_samples, 192).
            y (np.ndarray or None): Target labels (species names) or None for test.
            ids (np.ndarray): Image IDs.
    """
    os.makedirs(cache_dir, exist_ok=True)

    # Define cache paths
    x_path = os.path.join(cache_dir, f"X_{split}.npy")
    y_path = os.path.join(cache_dir, f"y_{split}.npy")
    ids_path = os.path.join(cache_dir, f"ids_{split}.npy")

    # Try loading from cache
    if load_cached:
        # Check if files exist
        files_exist = os.path.exists(x_path) and os.path.exists(ids_path)
        if split != "test":
            files_exist = files_exist and os.path.exists(y_path)

        if files_exist:
            print(f"Loading {split} data from cache...")
            X = np.load(x_path)
            ids = np.load(ids_path)
            if split != "test":
                y = np.load(y_path, allow_pickle=True)
            else:
                y = None
            return X, y, ids

    # Load from CSV if cache miss or forced reload
    print(f"Processing {split} data from CSV...")
    if split == "train":
        data_path = TRAIN_DATA_PATH
    elif split == "val":
        data_path = VAL_DATA_PATH
    elif split == "test":
        data_path = TEST_DATA_PATH
    else:
        raise ValueError(f"Invalid split: {split}")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path)

    # Extract IDs
    ids = df["id"].values

    # Extract Features
    # We exclude 'id', 'species', and 'image_path' to get the 192 features
    exclude_cols = ["id", "species", "image_path"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    # Ensure we are getting the expected number of features (64*3 = 192)
    # The columns in metadata are already sorted/grouped, so direct extraction preserves order.
    X = df[feature_cols].values.astype(np.float32)

    # Extract Targets
    if "species" in df.columns:
        y = df["species"].values
    else:
        y = None

    # Save to cache
    np.save(x_path, X)
    np.save(ids_path, ids)
    if y is not None:
        np.save(y_path, y)

    return X, y, ids


def save_submission(ids, probabilities, class_names, output_path):
    """
    Formats and saves the submission file.

    Args:
        ids (np.ndarray or list): Image IDs.
        probabilities (np.ndarray): Predicted probabilities (n_samples, n_classes).
        class_names (list): List of class names corresponding to probability columns.
        output_path (str): Path to save the CSV.
    """
    # Create initial dataframe with predicted probabilities
    submission_df = pd.DataFrame(probabilities, columns=class_names)
    submission_df["id"] = ids

    # Load sample submission to get correct column order
    if os.path.exists(SAMPLE_SUBMISSION_PATH):
        sample_df = pd.read_csv(SAMPLE_SUBMISSION_PATH)
        # Get all columns from sample submission
        target_cols = sample_df.columns.tolist()

        # Ensure 'id' is in the list
        if "id" not in target_cols:
            target_cols.insert(0, "id")

        # Reorder columns and fill missing with 0.0 (in case a class was missing in training)
        submission_df = submission_df.reindex(columns=target_cols, fill_value=0.0)
    else:
        print("Warning: Sample submission not found. Saving with provided class order.")
        # Ensure id is first
        cols = ["id"] + [c for c in submission_df.columns if c != "id"]
        submission_df = submission_df[cols]

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
