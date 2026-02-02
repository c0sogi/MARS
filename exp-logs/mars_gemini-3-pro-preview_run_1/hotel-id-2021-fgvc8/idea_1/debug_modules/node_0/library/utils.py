import os
import numpy as np
import pandas as pd
from library.config import Config, seed_everything


class HotelIdLabelEncoder:
    """
    Encodes sparse hotel_ids to contiguous integers (0 to N-1).
    Wraps numpy operations and supports saving/loading to .npy files.
    """

    def __init__(self):
        self.classes_ = np.array([])
        self.class_to_idx = {}

    def fit(self, y):
        """
        Fit label encoder to the array of hotel IDs.

        Args:
            y (array-like): Array of hotel IDs.
        """
        self.classes_ = np.unique(y)
        self.classes_.sort()
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes_)}
        return self

    def transform(self, y):
        """
        Transform hotel IDs to indices.

        Args:
            y (array-like): Array of hotel IDs.

        Returns:
            np.ndarray: Array of indices.
        """
        return np.array([self.class_to_idx.get(x, -1) for x in y], dtype=np.int64)

    def inverse_transform(self, y):
        """
        Transform indices back to hotel IDs.

        Args:
            y (array-like): Array of indices.

        Returns:
            np.ndarray: Array of hotel IDs.
        """
        return self.classes_[y]

    def save(self, path):
        """
        Save the encoder classes to a .npy file.

        Args:
            path (str): File path to save the .npy file.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, self.classes_)

    def load(self, path):
        """
        Load the encoder classes from a .npy file.

        Args:
            path (str): File path to load the .npy file from.
        """
        self.classes_ = np.load(path)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes_)}
        return self


def get_label_encoder(metadata_df=None, encoder_path=None, load_cached_data=True):
    """
    Retrieves a HotelIdLabelEncoder, handling caching logic.

    Args:
        metadata_df (pd.DataFrame, optional): DataFrame containing 'hotel_id' column for fitting.
                                              Required if cache is missing or ignored.
        encoder_path (str, optional): Path to save/load the encoder (.npy file).
                                      Defaults to 'label_encoder.npy' in Config.WORKING_DIR.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        HotelIdLabelEncoder: The fitted encoder.
    """
    if encoder_path is None:
        encoder_path = os.path.join(Config.WORKING_DIR, "label_encoder.npy")

    encoder = HotelIdLabelEncoder()

    # 1. IF load_cached_data is True: Try to load the file.
    if load_cached_data and os.path.exists(encoder_path):
        try:
            encoder.load(encoder_path)
            return encoder
        except Exception:
            # If load fails, proceed to fit
            pass

    # 2. IF loading fails OR load_cached_data is False:
    # Compute/process from scratch
    if metadata_df is None:
        raise ValueError(
            "metadata_df is required to fit the encoder if cache is missing or ignored."
        )

    encoder.fit(metadata_df["hotel_id"].values)

    # Save the result to the cache directory
    encoder.save(encoder_path)

    return encoder


def calculate_map5(predictions, targets):
    """
    Calculates Mean Average Precision @ 5.

    Args:
        predictions (list or np.ndarray): Shape (N, 5). Top 5 predicted labels/indices.
        targets (list or np.ndarray): Shape (N,). Ground truth labels/indices.

    Returns:
        float: The MAP@5 score.
    """
    # Ensure inputs are numpy arrays
    predictions = np.array(predictions)
    targets = np.array(targets)

    n = len(targets)
    if n == 0:
        return 0.0

    score_sum = 0.0

    for i in range(n):
        pred = predictions[i]
        target = targets[i]

        ap = 0.0
        # Check top 5 predictions
        for rank, p in enumerate(pred[:5]):
            if p == target:
                # Rank is 0-indexed in enumerate, formula uses 1-based rank
                # AP = 1/rank because there is only 1 relevant item
                ap = 1.0 / (rank + 1)
                break

        score_sum += ap

    return score_sum / n
