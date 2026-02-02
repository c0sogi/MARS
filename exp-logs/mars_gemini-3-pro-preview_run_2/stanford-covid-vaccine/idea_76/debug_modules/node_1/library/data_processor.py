import os
import ast
import numpy as np
import pandas as pd
from library.config import Config


class DataProcessor:
    """
    Handles data loading, feature engineering, and caching for the AHC-HIDN model.
    """

    def __init__(self):
        self.seq_map = {"A": 0, "G": 1, "C": 2, "U": 3}
        self.struct_map = {"(": 0, ")": 1, ".": 2}
        self.loop_map = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}

        # Reverse map for checking
        self.seq_chars = ["A", "G", "C", "U"]

    def parse_list_column(self, x):
        """Parses a stringified list into a numpy array."""
        try:
            return np.array(ast.literal_eval(x), dtype=np.float32)
        except Exception:
            return np.array([], dtype=np.float32)

    def get_pair_map(self, structure):
        """
        Parses dot-bracket structure to find base pairs.
        Returns an array of indices where arr[i] = j means i pairs with j.
        If unpaired, arr[i] = -1.
        """
        pair_map = np.full(len(structure), -1, dtype=int)
        stack = []

        for i, char in enumerate(structure):
            if char == "(":
                stack.append(i)
            elif char == ")":
                if stack:
                    start_index = stack.pop()
                    pair_map[start_index] = i
                    pair_map[i] = start_index

        return pair_map

    def one_hot_encode_sequence(self, sequence, mapping, length):
        """One-hot encodes a sequence string based on a mapping."""
        arr = np.zeros((length, len(mapping)), dtype=np.float32)
        for i, char in enumerate(sequence):
            if i >= length:
                break
            if char in mapping:
                arr[i, mapping[char]] = 1.0
        return arr

    def generate_features(self, df):
        """
        Generates input features and partner indices for a dataframe.

        Features:
        1. Sequence One-Hot (4)
        2. Structure One-Hot (3)
        3. Loop Type One-Hot (7)
        4. Partner Identity One-Hot (4)

        Total Channels: 18
        """
        n_samples = len(df)
        seq_len = Config.SEQ_LENGTH

        # Initialize arrays
        # Channels: 4 (Seq) + 3 (Struct) + 7 (Loop) + 4 (PartnerID) = 18
        inputs = np.zeros((n_samples, seq_len, 18), dtype=np.float32)
        partner_indices = np.zeros((n_samples, seq_len), dtype=np.int32)

        for idx, row in df.iterrows():
            # 1. Base Features
            seq_oh = self.one_hot_encode_sequence(
                row["sequence"], self.seq_map, seq_len
            )
            struct_oh = self.one_hot_encode_sequence(
                row["structure"], self.struct_map, seq_len
            )
            loop_oh = self.one_hot_encode_sequence(
                row["predicted_loop_type"], self.loop_map, seq_len
            )

            # 2. Partner Mapping
            p_map = self.get_pair_map(row["structure"])
            partner_indices[idx] = p_map

            # 3. Partner Identity
            # If i is paired with j, partner_id[i] = one_hot(sequence[j])
            # If unpaired, partner_id[i] = 0
            partner_id_oh = np.zeros((seq_len, 4), dtype=np.float32)

            # Vectorized partner identity assignment
            # Get indices where pairs exist
            paired_mask = p_map != -1
            if np.any(paired_mask):
                # Get the indices of the partners
                partners = p_map[paired_mask]
                # Assign the sequence one-hot of the partner to the current position
                partner_id_oh[paired_mask] = seq_oh[partners]

            # Concatenate all features
            # [Seq(4), Struct(3), Loop(7), PartnerID(4)]
            inputs[idx] = np.concatenate(
                [seq_oh, struct_oh, loop_oh, partner_id_oh], axis=1
            )

        return inputs, partner_indices

    def generate_targets(self, df):
        """
        Generates target arrays with Boundary Anchoring.
        Unscored positions (68-107) are filled with 0.0.
        """
        n_samples = len(df)
        seq_len = Config.SEQ_LENGTH
        seq_scored = Config.SEQ_SCORED
        n_targets = len(Config.TARGET_COLS)

        # Initialize with zeros (this handles the Boundary Anchoring for tail implicitly)
        targets = np.zeros((n_samples, seq_len, n_targets), dtype=np.float32)

        # Columns to process
        cols = (
            Config.TARGET_COLS
        )  # ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

        for i, col in enumerate(cols):
            # Parse the column (it's a stringified list in the CSV)
            # We assume the CSVs in metadata have these columns populated for train/val
            # Note: The metadata CSVs store lists as strings.

            # Extract all rows for this column, parse them
            # This can be slow row-by-row, but dataset is small (2k).
            parsed_col = df[col].apply(self.parse_list_column).values

            for idx, val_array in enumerate(parsed_col):
                # val_array should be length 68
                length = min(len(val_array), seq_scored)
                targets[idx, :length, i] = val_array[:length]

        return targets

    def process_data(self, mode="train", load_cached_data=True):
        """
        Main function to load, process, and cache data.

        Args:
            mode (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to try loading from cache.

        Returns:
            dict: Containing 'inputs', 'partner_indices', 'targets' (if not test), 'ids'.
        """
        # Determine cache path
        # We use a single cache file for all splits or separate?
        # The prompt suggests a specific cache key `train_data_ahc_hidn_v1.npz`.
        # To avoid conflicts, we'll append the mode to the filename or use a dict structure.
        # Let's use separate files for clarity: train_data_..., val_data_..., test_data_...

        base_name = Config.CACHE_NAME.replace("train_data", f"{mode}_data")
        cache_path = os.path.join(Config.WORKING_DIR, base_name)

        # 1. Try Load Cache
        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading cached {mode} data from {cache_path}...")
            try:
                data = np.load(cache_path, allow_pickle=True)
                result = {
                    "inputs": data["inputs"],
                    "partner_indices": data["partner_indices"],
                    "ids": data["ids"],
                }
                if "targets" in data:
                    result["targets"] = data["targets"]
                return result
            except Exception as e:
                print(f"Failed to load cache: {e}. Reprocessing...")

        # 2. Load Metadata
        print(f"Processing {mode} data from metadata...")
        csv_path = os.path.join(Config.METADATA_DIR, f"{mode}.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Metadata file not found: {csv_path}")

        df = pd.read_csv(csv_path)

        # Debugging subset
        if Config.SUBSET_SIZE is not None and mode == "train":
            print(f"DEBUG: Using subset of {Config.SUBSET_SIZE} samples.")
            df = df.head(Config.SUBSET_SIZE)

        # 3. Generate Features
        inputs, partner_indices = self.generate_features(df)
        ids = df["id"].values

        result = {"inputs": inputs, "partner_indices": partner_indices, "ids": ids}

        # 4. Generate Targets (if available)
        if mode in ["train", "val"]:
            targets = self.generate_targets(df)
            result["targets"] = targets

        # 5. Save Cache
        print(f"Saving {mode} data to {cache_path}...")
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez_compressed(cache_path, **result)

        return result
