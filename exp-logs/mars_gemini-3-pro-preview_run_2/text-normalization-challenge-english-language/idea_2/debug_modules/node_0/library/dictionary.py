import os
import json
import pandas as pd
from collections import defaultdict

from library.config import Config
from library.utils import load_dataset


class NormalizationDictionary:
    """
    Manages the memory-augmented normalization dictionary.
    Maps (raw_token, predicted_class) -> most_frequent_normalized_text.
    """

    def __init__(self):
        self.mapping = {}
        self.dict_path = Config.NORM_DICT_PATH
        self.train_path = Config.TRAIN_DATA_PATH

    def build(self, load_cached_data=True, row_limit=None):
        """
        Builds the normalization dictionary from training data.

        Args:
            load_cached_data (bool): If True, attempts to load from JSON first.
            row_limit (int, optional): Limit rows for debugging.
        """
        # 1. Try loading from cache
        if load_cached_data and os.path.exists(self.dict_path):
            print(f"Loading normalization dictionary from {self.dict_path}...")
            try:
                self.load()
                return
            except Exception as e:
                print(f"Failed to load cache: {e}. Rebuilding...")

        # 2. Build from scratch
        print("Building normalization dictionary from scratch...")

        # Load data
        # We only need specific columns for the dictionary
        # Using specific dtypes to optimize memory
        cols = {"class": "category", "before": "object", "after": "object"}

        # Load dataset
        df = load_dataset(self.train_path, dtype=cols)

        # Apply limits
        limit = row_limit if row_limit is not None else Config.DEBUG_ROW_LIMIT
        if limit:
            print(f"Limiting dictionary build to {limit} rows.")
            df = df.head(limit)

        print("Aggregating token statistics...")

        # Ensure data is string type for grouping
        df["before"] = df["before"].astype(str)
        df["after"] = df["after"].astype(str)

        # We want the most frequent 'after' for each ('class', 'before') pair.
        # Group by [class, before, after] and count occurrences.
        # observed=True is required for categorical groupby in newer pandas versions
        counts = (
            df.groupby(["class", "before", "after"], observed=True)
            .size()
            .reset_index(name="count")
        )

        # Sort by count descending so the most frequent is first
        counts.sort_values("count", ascending=False, inplace=True)

        # Drop duplicates to keep only the most frequent 'after' for each (class, before)
        best_mappings = counts.drop_duplicates(subset=["class", "before"], keep="first")

        print(f"Extracted {len(best_mappings)} unique (class, token) pairs.")

        # Convert to nested dictionary structure: dict[class][before] = after
        print("Constructing dictionary...")
        temp_mapping = defaultdict(dict)

        # Iterate and fill
        for cls, before, after in zip(
            best_mappings["class"], best_mappings["before"], best_mappings["after"]
        ):
            temp_mapping[cls][before] = after

        # Convert defaultdict to regular dict for JSON serialization
        self.mapping = {k: dict(v) for k, v in temp_mapping.items()}

        # Save to cache
        self.save()
        print("Dictionary build complete.")

    def save(self):
        """Saves the dictionary to JSON."""
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.dict_path), exist_ok=True)

        with open(self.dict_path, "w", encoding="utf-8") as f:
            json.dump(self.mapping, f, ensure_ascii=False, indent=2)
        print(f"Saved normalization dictionary to {self.dict_path}")

    def load(self):
        """Loads the dictionary from JSON."""
        if not os.path.exists(self.dict_path):
            raise FileNotFoundError(f"Dictionary file not found at {self.dict_path}")

        with open(self.dict_path, "r", encoding="utf-8") as f:
            self.mapping = json.load(f)

    def get_normalization(self, raw_token, pred_class):
        """
        Retrieves the normalized text based on the raw token and predicted class.
        Implements the fallback logic (Copy) if the pair is not found.

        Args:
            raw_token (str): The input token.
            pred_class (str): The predicted semiotic class.

        Returns:
            str: The normalized text.
        """
        # Ensure inputs are strings
        raw_token = str(raw_token)
        pred_class = str(pred_class)

        # 1. Lookup Class
        if pred_class in self.mapping:
            class_dict = self.mapping[pred_class]
            # 2. Lookup Token
            if raw_token in class_dict:
                return class_dict[raw_token]

        # 3. Fallback: Return original token (Copy mechanism)
        return raw_token
