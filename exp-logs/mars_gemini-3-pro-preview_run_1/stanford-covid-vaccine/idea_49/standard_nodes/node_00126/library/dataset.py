import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from library.config import Config
from library.utils import parse_structure, compute_laplacian_pe, get_positional_encoding


class RNADataset(Dataset):
    """
    PyTorch Dataset for RNA Degradation Prediction.
    Implements the Spectral-Topological Wide-Stream Residual BiGRU data pipeline.
    """

    def __init__(self, mode="train", load_cached_data=True):
        """
        Args:
            mode (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to load pre-processed data from cache.
        """
        self.mode = mode
        self.seq_map = {"A": 0, "G": 1, "C": 2, "U": 3}
        self.loop_map = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}

        # Ensure working directory exists
        os.makedirs(Config.WORKING_DIR, exist_ok=True)

        # Define cache path
        self.cache_path = os.path.join(Config.WORKING_DIR, f"cached_{mode}.pt")

        # Caching Logic
        if load_cached_data and os.path.exists(self.cache_path):
            print(f"Loading cached {mode} data from {self.cache_path}...")
            self.data = torch.load(self.cache_path)
        else:
            print(f"Processing {mode} data from scratch...")
            self.data = self._process_data()
            print(f"Saving processed {mode} data to {self.cache_path}...")
            torch.save(self.data, self.cache_path)

    def _process_data(self):
        """
        Reads Parquet files, processes features (Seq, Loop, PairDist, LPE) and targets.
        Returns a dictionary of tensors.
        """
        # Load raw data
        if self.mode == "train":
            df = pd.read_parquet(Config.TRAIN_PATH)
        elif self.mode == "val":
            df = pd.read_parquet(Config.VAL_PATH)
        elif self.mode == "test":
            df = pd.read_parquet(Config.TEST_PATH)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # Limit for debugging if configured
        if Config.MAX_DEBUG_SAMPLES is not None:
            df = df.iloc[: Config.MAX_DEBUG_SAMPLES]

        # Initialize lists
        sequences = []
        loop_types = []
        pair_dists_enc = []
        lpes = []
        targets_list = []
        masks = []
        ids = []

        # Iterate over samples
        for idx, row in df.iterrows():
            seq_len = Config.SEQ_LEN  # 107
            pred_len = Config.PRED_LEN  # 68

            # 1. Sequence Tokenization
            seq_ints = [self.seq_map.get(c, 0) for c in row["sequence"]]
            sequences.append(torch.tensor(seq_ints, dtype=torch.long))

            # 2. Loop Type Tokenization
            loop_ints = [self.loop_map.get(c, 0) for c in row["predicted_loop_type"]]
            loop_types.append(torch.tensor(loop_ints, dtype=torch.long))

            # 3. Structure Features
            structure = row["structure"]

            # 3a. Signed Pairing Distance
            pairs = parse_structure(structure)
            dists = np.zeros(seq_len, dtype=np.float32)
            for i in range(seq_len):
                if i in pairs:
                    dists[i] = pairs[i] - i
                else:
                    dists[i] = 0.0

            # Convert to tensor and get sinusoidal encoding
            dists_tensor = torch.tensor(dists, dtype=torch.float32)
            # Shape: (seq_len, EMBED_DIM_PAIR)
            pe = get_positional_encoding(dists_tensor, Config.EMBED_DIM_PAIR)
            pair_dists_enc.append(pe)

            # 3b. Laplacian Positional Encoding (LPE)
            # Shape: (seq_len, LPE_DIM=8)
            lpe = compute_laplacian_pe(structure, seq_len, k=Config.LPE_DIM)
            lpes.append(torch.tensor(lpe, dtype=torch.float32))

            # 4. Targets & Mask
            mask = torch.zeros(seq_len, dtype=torch.float32)
            mask[:pred_len] = 1.0
            masks.append(mask)

            if self.mode in ["train", "val"]:
                # Extract targets: reactivity, deg_Mg_pH10, deg_Mg_50C
                # These are lists of length 68
                t_matrix = []
                for col in Config.TARGET_COLS:
                    val_list = row[col]
                    # Ensure it's a list or array
                    if isinstance(val_list, np.ndarray):
                        val_list = val_list.tolist()
                    t_matrix.append(val_list)

                # Transpose to (68, 3)
                t_matrix = np.array(t_matrix).T

                # Pad to (107, 3)
                pad_len = seq_len - len(t_matrix)
                if pad_len > 0:
                    t_padded = np.pad(
                        t_matrix,
                        ((0, pad_len), (0, 0)),
                        mode="constant",
                        constant_values=0.0,
                    )
                else:
                    t_padded = t_matrix

                targets_list.append(torch.tensor(t_padded, dtype=torch.float32))
            else:
                # Test mode: Dummy targets
                targets_list.append(
                    torch.zeros((seq_len, Config.NUM_TARGETS), dtype=torch.float32)
                )

            ids.append(row["id"])

        # Stack everything
        data_dict = {
            "sequence": torch.stack(sequences),  # (N, 107)
            "loop_type": torch.stack(loop_types),  # (N, 107)
            "pair_dist": torch.stack(pair_dists_enc),  # (N, 107, 64)
            "lpe": torch.stack(lpes),  # (N, 107, 8)
            "targets": torch.stack(targets_list),  # (N, 107, 3)
            "mask": torch.stack(masks),  # (N, 107)
            "ids": ids,  # List of strings
        }

        return data_dict

    def __len__(self):
        return len(self.data["sequence"])

    def __getitem__(self, idx):
        return {
            "sequence": self.data["sequence"][idx],
            "loop_type": self.data["loop_type"][idx],
            "pair_dist": self.data["pair_dist"][idx],
            "lpe": self.data["lpe"][idx],
            "targets": self.data["targets"][idx],
            "mask": self.data["mask"][idx],
            "id": self.data["ids"][idx],
        }
