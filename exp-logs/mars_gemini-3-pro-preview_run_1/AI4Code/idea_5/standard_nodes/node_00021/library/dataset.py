import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from library.config import Config
from library.utils import set_seed
from library.preprocess import precompute_features


class NotebookDataset(Dataset):
    def __init__(self, data_path, load_cached_data=True):
        """
        Dataset for loading pre-computed notebook features.

        Args:
            data_path (str): Path to the parquet file containing features.
            load_cached_data (bool): Flag to indicate if cached data should be used.
                                     If False or file missing, triggers computation.
        """
        self.data_path = data_path

        # Caching Logic
        if not load_cached_data or not os.path.exists(data_path):
            print(
                f"Data not found or reload requested at {data_path}. Triggering precomputation..."
            )
            # We assume precompute_features handles the logic for all splits
            # or we rely on the caller to have run it.
            # However, strictly following requirements, we trigger it here.
            precompute_features(load_cached_data=load_cached_data)

        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"Failed to load data from {data_path} after precomputation attempt."
            )

        # Load data using pyarrow engine for efficiency with list columns
        self.df = pd.read_parquet(data_path, engine="pyarrow")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Process Code Embeddings
        # Parquet stores lists of arrays or lists of lists.
        # We convert to a list first to ensure compatibility, then to Tensor.
        c_data = row["code_embeddings"]
        if isinstance(c_data, np.ndarray):
            c_data = c_data.tolist()

        if c_data is not None and len(c_data) > 0:
            code_emb = torch.tensor(c_data, dtype=torch.float32)
        else:
            # Handle empty code cells: shape (0, hidden_dim)
            code_emb = torch.zeros((0, Config.INPUT_DIM), dtype=torch.float32)

        # Process Markdown Embeddings
        m_data = row["markdown_embeddings"]
        if isinstance(m_data, np.ndarray):
            m_data = m_data.tolist()

        if m_data is not None and len(m_data) > 0:
            md_emb = torch.tensor(m_data, dtype=torch.float32)
        else:
            md_emb = torch.zeros((0, Config.INPUT_DIM), dtype=torch.float32)

        # Process Labels
        # Labels are integers representing the index of the following code cell
        labels = row["markdown_labels"]
        if labels is not None and len(labels) > 0:
            labels = torch.tensor(labels, dtype=torch.long)
        else:
            labels = torch.tensor([], dtype=torch.long)

        return {
            "id": row["id"],
            "code_embeddings": code_emb,
            "markdown_embeddings": md_emb,
            "markdown_labels": labels,
            "code_ids": row["code_ids"],
            "markdown_ids": row["markdown_ids"],
        }


def collate_fn(batch):
    """
    Custom collate function to handle variable length sequences of code and markdown cells.
    """
    ids = [item["id"] for item in batch]
    code_ids = [item["code_ids"] for item in batch]
    markdown_ids = [item["markdown_ids"] for item in batch]

    # 1. Pad Code Embeddings
    code_embs = [item["code_embeddings"] for item in batch]
    # pad_sequence requires non-empty tensors to determine dim,
    # but we handled (0, 768) in __getitem__.
    padded_code = pad_sequence(code_embs, batch_first=True, padding_value=0.0)

    # 2. Create Code Mask
    # True for valid tokens, False for padding
    code_lens = torch.tensor([c.shape[0] for c in code_embs], dtype=torch.long)
    max_code_len = padded_code.size(1)
    if max_code_len > 0:
        code_mask = torch.arange(max_code_len).expand(
            len(batch), max_code_len
        ) < code_lens.unsqueeze(1)
    else:
        code_mask = torch.zeros((len(batch), 0), dtype=torch.bool)

    # 3. Pad Markdown Embeddings
    md_embs = [item["markdown_embeddings"] for item in batch]
    padded_md = pad_sequence(md_embs, batch_first=True, padding_value=0.0)

    # 4. Create Markdown Mask
    md_lens = torch.tensor([m.shape[0] for m in md_embs], dtype=torch.long)
    max_md_len = padded_md.size(1)
    if max_md_len > 0:
        md_mask = torch.arange(max_md_len).expand(
            len(batch), max_md_len
        ) < md_lens.unsqueeze(1)
    else:
        md_mask = torch.zeros((len(batch), 0), dtype=torch.bool)

    # 5. Pad Labels
    # Use -100 as ignore_index for CrossEntropyLoss
    labels = [item["markdown_labels"] for item in batch]
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=-100)

    return {
        "ids": ids,
        "code_features": padded_code,
        "code_mask": code_mask,
        "markdown_features": padded_md,
        "markdown_mask": md_mask,
        "labels": padded_labels,
        "code_ids": code_ids,
        "markdown_ids": markdown_ids,
        "code_lens": code_lens,
    }


def get_dataloader(split="train", batch_size=None, shuffle=None, load_cached_data=True):
    """
    Factory function to create a DataLoader for a specific split.

    Args:
        split (str): 'train', 'val', or 'test'.
        batch_size (int): Batch size override.
        shuffle (bool): Shuffle override.
        load_cached_data (bool): Whether to use cached parquet files.

    Returns:
        DataLoader: Configured PyTorch DataLoader.
    """
    if split == "train":
        data_path = Config.TRAIN_FEATURES_PATH
        default_shuffle = True
    elif split == "val":
        data_path = Config.VAL_FEATURES_PATH
        default_shuffle = False
    elif split == "test":
        data_path = Config.TEST_FEATURES_PATH
        default_shuffle = False
    else:
        raise ValueError(f"Unknown split: {split}")

    bs = batch_size if batch_size is not None else Config.BATCH_SIZE
    shuf = shuffle if shuffle is not None else default_shuffle

    dataset = NotebookDataset(data_path, load_cached_data=load_cached_data)

    loader = DataLoader(
        dataset,
        batch_size=bs,
        shuffle=shuf,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=(Config.DEVICE == "cuda"),
    )

    return loader
