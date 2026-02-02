import os
import json
import random
import numpy as np
import pandas as pd
import torch
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class LabelEncoder:
    """
    Handles encoding of raw hotel_ids to class indices and decoding back to hotel_ids.
    Supports saving and loading state via JSON.
    """

    def __init__(self):
        self.id_to_class = {}
        self.class_to_id = {}

    def fit(self, hotel_ids):
        """
        Fits the encoder to a list of unique hotel_ids.
        """
        unique_ids = sorted(list(set(hotel_ids)))
        self.id_to_class = {int(hid): i for i, hid in enumerate(unique_ids)}
        self.class_to_id = {i: int(hid) for i, hid in enumerate(unique_ids)}

    def transform(self, hotel_ids):
        """
        Converts raw hotel_ids to class indices.
        """
        # Handle scalar
        if np.isscalar(hotel_ids):
            return self.id_to_class.get(int(hotel_ids), -1)
        # Handle iterable
        return np.array([self.id_to_class.get(int(hid), -1) for hid in hotel_ids])

    def inverse_transform(self, class_indices):
        """
        Converts class indices back to raw hotel_ids.
        """
        # Handle scalar
        if np.isscalar(class_indices):
            return self.class_to_id.get(int(class_indices), -1)
        # Handle iterable
        return np.array([self.class_to_id.get(int(idx), -1) for idx in class_indices])

    def to_dict(self):
        """
        Returns the internal mappings as a dictionary for serialization.
        JSON requires keys to be strings.
        """
        return {
            "id_to_class": {str(k): v for k, v in self.id_to_class.items()},
            "class_to_id": {str(k): v for k, v in self.class_to_id.items()},
        }

    def from_dict(self, data):
        """
        Restores internal mappings from a dictionary.
        Converts keys back to integers.
        """
        self.id_to_class = {int(k): int(v) for k, v in data["id_to_class"].items()}
        self.class_to_id = {int(k): int(v) for k, v in data["class_to_id"].items()}


def get_label_encoder(metadata_path=Config.TRAIN_CSV, load_cached_data=True):
    """
    Factory function to get a fitted LabelEncoder.
    Implements caching logic to avoid re-processing metadata.

    Args:
        metadata_path (str): Path to the training metadata CSV.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        LabelEncoder: A fitted LabelEncoder instance.
    """
    cache_path = os.path.join(Config.CACHE_DIR, "label_encoder.json")
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    encoder = LabelEncoder()

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            with open(cache_path, "r") as f:
                data = json.load(f)
            encoder.from_dict(data)
            # Simple validation to ensure it loaded correctly
            if len(encoder.id_to_class) > 0:
                return encoder
        except Exception as e:
            # If load fails, proceed to compute from scratch
            pass

    # 2. Compute from scratch
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

    df = pd.read_csv(metadata_path)
    if "hotel_id" not in df.columns:
        raise ValueError(f"Column 'hotel_id' not found in {metadata_path}")

    unique_hotels = df["hotel_id"].unique()
    encoder.fit(unique_hotels)

    # 3. Save to cache
    try:
        with open(cache_path, "w") as f:
            json.dump(encoder.to_dict(), f)
    except Exception as e:
        # Non-critical failure if cache cannot be written
        pass

    return encoder


def calculate_map5(predictions, targets):
    """
    Calculates Mean Average Precision @ 5 (MAP@5).

    Args:
        predictions: List of lists or 2D array (N, 5) containing predicted IDs/indices.
                     The items should be ordered by relevance (most relevant first).
        targets: List or 1D array (N,) containing the ground truth ID/index.

    Returns:
        float: The MAP@5 score.
    """
    if len(targets) == 0:
        return 0.0

    score_sum = 0.0
    n = len(targets)

    for i in range(n):
        preds = predictions[i]
        target = targets[i]

        # Check if target is in the top 5 predictions
        # Note: We assume preds contains at most 5 items, but logic holds if more.
        # We only care about the first 5.

        # Convert to list if numpy array for easy 'index' lookup
        if isinstance(preds, np.ndarray):
            preds = preds.tolist()

        # Truncate to top 5 just in case
        top_5 = preds[:5]

        if target in top_5:
            # Rank is 1-based index
            rank = top_5.index(target) + 1
            score_sum += 1.0 / rank

    return score_sum / n
