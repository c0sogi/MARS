import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import numpy as np
from library.config import Config


class NotebookEmbeddingDataset(Dataset):
    def __init__(self, split="train", max_size=None):
        """
        Dataset for loading pre-computed notebook embeddings.

        Args:
            split (str): One of 'train', 'val', 'test'.
            max_size (int, optional): Limit the dataset size for debugging.
        """
        self.split = split
        self.max_size = max_size

        # Select paths based on split
        if split == "train":
            self.parquet_path = Config.TRAIN_CACHE_PATH
            self.metadata_path = Config.TRAIN_METADATA_PATH
        elif split == "val":
            self.parquet_path = Config.VAL_CACHE_PATH
            self.metadata_path = Config.VAL_METADATA_PATH
        elif split == "test":
            self.parquet_path = Config.TEST_CACHE_PATH
            self.metadata_path = Config.TEST_METADATA_PATH
        else:
            raise ValueError(f"Invalid split: {split}")

        self.df = self._load_data()

    def _load_data(self):
        # Check if parquet file exists
        if not pd.io.common.file_exists(self.parquet_path):
            raise FileNotFoundError(
                f"Parquet file not found at {self.parquet_path}. "
                "Please run the data preprocessor first."
            )

        df = pd.read_parquet(self.parquet_path)

        # For Train/Val, we need the ground truth cell_order from metadata
        if self.split != "test":
            if not pd.io.common.file_exists(self.metadata_path):
                raise FileNotFoundError(
                    f"Metadata file not found at {self.metadata_path}"
                )

            df_meta = pd.read_csv(self.metadata_path)
            # Merge cell_order. We assume 'id' is unique and present in both.
            df = df.merge(df_meta[["id", "cell_order"]], on="id", how="left")

        # Optional subsampling for debugging
        if self.max_size is not None and self.max_size < len(df):
            df = df.iloc[: self.max_size].reset_index(drop=True)

        return df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load embeddings (stored as lists in Parquet)
        # Fix: Handle case where Parquet/Pandas returns numpy array of objects
        code_data = row["code_embeddings"]
        if isinstance(code_data, np.ndarray) and code_data.dtype == object:
            code_data = code_data.tolist()

        md_data = row["markdown_embeddings"]
        if isinstance(md_data, np.ndarray) and md_data.dtype == object:
            md_data = md_data.tolist()

        # Shape: (Seq_Len, Hidden_Dim)
        code_embs = torch.tensor(code_data, dtype=torch.float32)
        md_embs = torch.tensor(md_data, dtype=torch.float32)

        code_ids = row["code_cell_ids"]
        md_ids = row["markdown_cell_ids"]

        # Handle potential empty sequences
        if code_embs.numel() == 0:
            code_embs = torch.empty((0, Config.INPUT_DIM), dtype=torch.float32)
        if md_embs.numel() == 0:
            md_embs = torch.empty((0, Config.INPUT_DIM), dtype=torch.float32)

        labels = []

        if self.split != "test":
            # --- Label Generation Logic ---
            # Goal: For each MD cell, find the rank (index) of the *next* code cell.
            # If it's after the last code cell, rank = num_code_cells.

            cell_order_str = row["cell_order"]
            if pd.isna(cell_order_str):
                # Fallback if cell_order is missing (should not happen in clean data)
                cell_order = []
            else:
                cell_order = cell_order_str.split()

            # Map code IDs to their 0-indexed rank
            # Note: code_ids in the parquet are already sorted by execution order for Train/Val
            code_rank_map = {cid: i for i, cid in enumerate(code_ids)}
            num_code = len(code_ids)

            # Efficient backward pass to determine labels
            next_code_rank = num_code
            md_id_set = set(md_ids)
            position_to_label = {}

            for cid in reversed(cell_order):
                if cid in code_rank_map:
                    next_code_rank = code_rank_map[cid]
                elif cid in md_id_set:
                    position_to_label[cid] = next_code_rank

            # Collect labels in the order of md_ids
            labels = [position_to_label.get(mid, num_code) for mid in md_ids]
            labels = torch.tensor(labels, dtype=torch.long)

            # --- Shuffling ---
            # Shuffle markdown cells to simulate the set-based input nature
            # and ensure the model learns permutation invariance.
            perm = torch.randperm(len(md_ids))
            md_embs = md_embs[perm]
            labels = labels[perm]
            md_ids = [md_ids[i] for i in perm.tolist()]

        else:
            # Test split: No labels, no shuffling (preserve input order)
            labels = torch.full((len(md_ids),), -1, dtype=torch.long)

        return {
            "id": row["id"],
            "code_embeddings": code_embs,
            "markdown_embeddings": md_embs,
            "labels": labels,
            "code_cell_ids": code_ids,
            "markdown_cell_ids": md_ids,
        }

    @staticmethod
    def collate_fn(batch):
        """
        Custom collate function to pad sequences and create masks.
        """
        ids = [item["id"] for item in batch]
        code_ids = [item["code_cell_ids"] for item in batch]
        md_ids = [item["markdown_cell_ids"] for item in batch]

        code_embs = [item["code_embeddings"] for item in batch]
        md_embs = [item["markdown_embeddings"] for item in batch]
        labels = [item["labels"] for item in batch]

        # Pad sequences
        # Output shape: (Batch, Max_Len, Dim)
        code_padded = pad_sequence(code_embs, batch_first=True, padding_value=0.0)
        md_padded = pad_sequence(md_embs, batch_first=True, padding_value=0.0)

        # Pad labels with -100 (ignore index for CrossEntropy)
        labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)

        # Create Masks (True = Valid, False = Padding)
        # Calculate lengths
        code_lens = torch.tensor([len(x) for x in code_embs], dtype=torch.long)
        md_lens = torch.tensor([len(x) for x in md_embs], dtype=torch.long)

        B_code, max_code = code_padded.shape[:2]
        B_md, max_md = md_padded.shape[:2]

        # Expand arange to create boolean masks
        code_mask = torch.arange(max_code).expand(
            B_code, max_code
        ) < code_lens.unsqueeze(1)
        md_mask = torch.arange(max_md).expand(B_md, max_md) < md_lens.unsqueeze(1)

        return {
            "ids": ids,
            "code_embeddings": code_padded,
            "markdown_embeddings": md_padded,
            "labels": labels_padded,
            "code_mask": code_mask,
            "markdown_mask": md_mask,
            "code_cell_ids": code_ids,
            "markdown_cell_ids": md_ids,
        }
