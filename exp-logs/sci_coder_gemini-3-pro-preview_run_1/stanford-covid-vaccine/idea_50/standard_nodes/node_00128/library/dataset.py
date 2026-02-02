import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
from library.config import Config
from library.features import extract_features


class RNADataset(Dataset):
    """
    PyTorch Dataset for RNA Degradation Prediction.
    Handles loading from Parquet, tokenization, feature integration (RWPE, Pair Enc),
    and target formatting.
    """

    def __init__(self, split="train", load_cached=True):
        """
        Args:
            split (str): One of 'train', 'val', 'test'.
            load_cached (bool): Whether to use cached geometric features.
        """
        self.split = split
        self.seq_len = Config.SEQ_LENGTH
        self.seq_scored = Config.SEQ_SCORED

        # 1. Load Metadata
        file_name = f"{split}.parquet"
        file_path = os.path.join(Config.METADATA_DIR, file_name)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Metadata file not found: {file_path}")

        self.df = pd.read_parquet(file_path)

        # 2. Handle Debugging (Slice Data)
        if Config.DEBUG:
            print(
                f"DEBUG Mode: Slicing {split} dataset to {Config.DEBUG_SUBSET_SIZE} samples."
            )
            self.df = self.df.iloc[: Config.DEBUG_SUBSET_SIZE].reset_index(drop=True)

        # 3. Extract/Load Geometric Features
        # We pass the dataframe to extract_features.
        # Note: extract_features handles caching based on the split name.
        # If running in debug mode, we append '_debug' to split name to avoid polluting full cache
        feature_split_name = f"{split}_debug" if Config.DEBUG else split

        # extract_features returns a dict of tensors
        features = extract_features(
            self.df, feature_split_name, load_cached=load_cached
        )
        self.pair_indices = features["pair_indices"]  # (N, L)

        # 4. Tokenize Sequences and Loop Types
        self.seq_ids = self._tokenize_sequence(self.df["sequence"].values)
        self.loop_ids = self._tokenize_loop(self.df["predicted_loop_type"].values)

        # 5. Process Targets (if available)
        self.targets = None
        self.mask = None

        if split in ["train", "val"]:
            self._process_targets()

        # Store IDs for submission
        self.ids = self.df["id"].values

    def _tokenize_sequence(self, sequences):
        """Vectorized tokenization of RNA sequences."""
        # Map chars to integers based on Config.TOKEN2ID_SEQ
        # Create a mapping array for fast lookup
        # ASCII values: A=65, C=67, G=71, U=85
        # We'll use a simple apply/map for clarity as vocab is small

        token_map = Config.TOKEN2ID_SEQ

        # Pre-allocate tensor
        n_samples = len(sequences)
        seq_tensor = torch.zeros((n_samples, self.seq_len), dtype=torch.long)

        for i, seq in enumerate(sequences):
            # Ensure length is correct
            s = list(seq[: self.seq_len])
            ids = [
                token_map.get(c, 0) for c in s
            ]  # Default to 0 (A) if unknown, though shouldn't happen
            seq_tensor[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)

        return seq_tensor

    def _tokenize_loop(self, loops):
        """Vectorized tokenization of predicted loop types."""
        token_map = Config.TOKEN2ID_LOOP

        n_samples = len(loops)
        loop_tensor = torch.zeros((n_samples, self.seq_len), dtype=torch.long)

        for i, loop_str in enumerate(loops):
            l = list(loop_str[: self.seq_len])
            ids = [token_map.get(c, 0) for c in l]  # Default to 0 (B) if unknown
            loop_tensor[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)

        return loop_tensor

    def _process_targets(self):
        """
        Extracts targets, pads them to seq_length, and creates masks.
        Targets: reactivity, deg_Mg_pH10, deg_Mg_50C
        """
        target_cols = Config.TARGET_COLS
        n_samples = len(self.df)

        # Initialize tensors
        # Shape: (N, SEQ_LENGTH, NUM_TARGETS)
        self.targets = torch.zeros(
            (n_samples, self.seq_len, len(target_cols)), dtype=torch.float32
        )

        # Mask: 1 for scored positions (0..67), 0 otherwise
        self.mask = torch.zeros((n_samples, self.seq_len), dtype=torch.float32)
        self.mask[:, : self.seq_scored] = 1.0

        for idx, col in enumerate(target_cols):
            # The column in parquet is an array of floats (length 68)
            # We stack them into a numpy array
            values = np.vstack(self.df[col].values)  # Shape (N, 68)

            # Assign to the tensor
            self.targets[:, : self.seq_scored, idx] = torch.tensor(
                values, dtype=torch.float32
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        Returns:
            dict: {
                'seq': (L,),
                'loop': (L,),
                'pair_indices': (L,),
                'targets': (L, 3) [Optional],
                'mask': (L,) [Optional],
                'id': str
            }
        """
        sample = {
            "seq": self.seq_ids[idx],
            "loop": self.loop_ids[idx],
            "pair_indices": self.pair_indices[idx],
            "id": self.ids[idx],
        }

        if self.targets is not None:
            sample["targets"] = self.targets[idx]
            sample["mask"] = self.mask[idx]

        return sample
