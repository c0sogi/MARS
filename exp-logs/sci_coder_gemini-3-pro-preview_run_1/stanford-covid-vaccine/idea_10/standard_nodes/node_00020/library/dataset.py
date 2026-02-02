import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from library.config import Config
from library.features import get_distance_encoding, get_paired_index_map


class RNADataset(Dataset):
    def __init__(self, split="train", config=Config(), load_cached_data=True):
        """
        PyTorch Dataset for RNA degradation prediction.

        Args:
            split (str): One of 'train', 'val', 'test'.
            config (Config): Configuration object containing paths and hyperparameters.
            load_cached_data (bool): Whether to try loading from cache first.
        """
        self.config = config
        self.split = split
        self.rng = np.random.default_rng(config.SEED)

        # Ensure working directory exists
        os.makedirs(config.WORKING_DIR, exist_ok=True)
        cache_path = os.path.join(config.WORKING_DIR, f"{split}_data.npz")

        # 1. Load from Cache if available
        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading cached {split} data from {cache_path}...")
            data = np.load(cache_path, allow_pickle=True)
            self.ids = data["ids"]
            self.sequences = data["sequences"]
            self.loop_types = data["loop_types"]
            self.pair_indices = data["pair_indices"]
            self.pair_distances = data["pair_distances"]

            if split in ["train", "val"]:
                self.targets = data["targets"]
            else:
                self.targets = None

        # 2. Process from Scratch
        else:
            print(f"Processing {split} data from metadata...")
            parquet_path = os.path.join(config.METADATA_DIR, f"{split}.parquet")

            if not os.path.exists(parquet_path):
                raise FileNotFoundError(f"Metadata file not found: {parquet_path}")

            df = pd.read_parquet(parquet_path)

            self.ids = df["id"].values

            # Encode Sequences
            self.sequences = []
            for seq in df["sequence"]:
                encoded = [config.TOKEN2ID.get(c, config.PAD_TOKEN_ID) for c in seq]
                self.sequences.append(encoded)
            self.sequences = np.array(self.sequences, dtype=np.int32)

            # Encode Loop Types
            self.loop_types = []
            for lt in df["predicted_loop_type"]:
                encoded = [config.LOOP2ID.get(c, 0) for c in lt]
                self.loop_types.append(encoded)
            self.loop_types = np.array(self.loop_types, dtype=np.int32)

            # Process Structure Features using library.features
            self.pair_indices = []
            self.pair_distances = []

            for struct in df["structure"]:
                # Get Paired-Base Identity indices
                p_idx = get_paired_index_map(struct, config.SEQ_LEN)
                self.pair_indices.append(p_idx)

                # Get Geometric Distance encoding
                p_dist = get_distance_encoding(struct, config.SEQ_LEN)
                self.pair_distances.append(p_dist)

            self.pair_indices = np.array(self.pair_indices, dtype=np.int32)
            self.pair_distances = np.array(self.pair_distances, dtype=np.float32)

            # Process Targets (Train/Val only)
            if split in ["train", "val"]:
                t_list = []
                # Load all target columns defined in Config to maintain index alignment
                for col in config.TARGET_COLS:
                    # df[col] contains lists/arrays. Stack them.
                    # Ensure we handle potential missing values or format issues if any,
                    # though metadata generation implies cleanliness.
                    col_data = np.vstack(df[col].values)
                    t_list.append(col_data)

                # Stack to shape (N, 68, 5)
                self.targets = np.stack(t_list, axis=2).astype(np.float32)
            else:
                self.targets = None

            # Save to Cache
            save_dict = {
                "ids": self.ids,
                "sequences": self.sequences,
                "loop_types": self.loop_types,
                "pair_indices": self.pair_indices,
                "pair_distances": self.pair_distances,
            }
            if self.targets is not None:
                save_dict["targets"] = self.targets

            np.savez(cache_path, **save_dict)
            print(f"Data cached to {cache_path}")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # Retrieve data
        seq = self.sequences[idx].copy()
        loop = self.loop_types[idx]
        pair_idx = self.pair_indices[idx]
        pair_dist = self.pair_distances[idx]

        # Initialize mask labels with -100 (ignored by CrossEntropyLoss)
        mask_labels = np.full_like(seq, -100)

        # Dynamic Masking for Reconstruction Task (Train only)
        if self.split == "train":
            # Generate random probabilities
            prob = self.rng.random(len(seq))

            # Create boolean mask based on probability
            mask_bool = prob < self.config.MASK_PROB

            # Set targets for masked positions (the original token ID)
            mask_labels[mask_bool] = seq[mask_bool]

            # Replace input tokens with MASK token
            seq[mask_bool] = self.config.MASK_TOKEN_ID

        # Prepare dictionary
        item = {
            "seq": torch.tensor(seq, dtype=torch.long),
            "loop": torch.tensor(loop, dtype=torch.long),
            "pair_idx": torch.tensor(pair_idx, dtype=torch.long),
            "pair_dist": torch.tensor(pair_dist, dtype=torch.float),
            "mask_labels": torch.tensor(mask_labels, dtype=torch.long),
        }

        if self.targets is not None:
            item["targets"] = torch.tensor(self.targets[idx], dtype=torch.float)

        return item
