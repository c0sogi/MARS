import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from library.config import Config


class NotebookDataset(Dataset):
    """
    Dataset class for the DC-CodeBERT model.
    Loads pre-computed embeddings and organizes them into:
    1. Ordered Code Sequence (Anchors)
    2. Markdown Set (Queries)
    3. Target Indices (Labels)
    """

    def __init__(self, features_path, mode="train"):
        """
        Args:
            features_path (str): Path to the parquet file containing features.
            mode (str): 'train', 'val', or 'test'.
        """
        self.mode = mode
        self.samples = []

        print(f"Loading features from {features_path}...")
        # Load the dataframe
        try:
            df = pd.read_parquet(features_path)
        except Exception as e:
            print(f"Error loading {features_path}: {e}")
            # Return empty if file doesn't exist (handled by pipeline usually)
            return

        # If in debug mode and the file is huge (not generated in debug), slice it
        if Config.DEBUG:
            unique_ids = df["id"].unique()[: Config.DEBUG_SAMPLE_SIZE]
            df = df[df["id"].isin(unique_ids)].copy()

        # Group by notebook ID
        # We process this into a list of dictionaries to save memory compared to keeping the DF
        grouped = df.groupby("id")

        print(f"Processing {len(grouped)} notebooks for {mode}...")

        for nb_id, group in grouped:
            # Separate Code and Markdown
            code_df = group[group["cell_type"] == "code"].copy()
            md_df = group[group["cell_type"] == "markdown"].copy()

            # 1. Prepare Code Sequence (Anchors)
            # Must be sorted by rank to maintain execution context
            if self.mode != "test":
                code_df = code_df.sort_values("rank")
            else:
                # In test, we assume code cells are in order in the file
                # (as per competition description: code cells are in original order)
                # If rank is -1, we rely on the file order (index)
                pass

            # Extract embeddings as numpy array
            if len(code_df) > 0:
                code_embeddings = np.stack(code_df["embedding"].values)
                code_ranks = code_df["rank"].values
            else:
                # Handle edge case: notebook with no code cells
                code_embeddings = np.zeros((0, Config.HIDDEN_DIM), dtype=np.float32)
                code_ranks = np.array([])

            # 2. Prepare Markdown Set (Queries)
            if len(md_df) > 0:
                md_embeddings = np.stack(md_df["embedding"].values)
                md_ids = md_df["cell_id"].values
                md_ranks = md_df["rank"].values
            else:
                md_embeddings = np.zeros((0, Config.HIDDEN_DIM), dtype=np.float32)
                md_ids = np.array([])
                md_ranks = np.array([])

            # 3. Generate Labels
            # Label = index of the code cell that immediately follows the markdown cell.
            # Equivalently: number of code cells that appear *before* the markdown cell.
            # Range: [0, num_code_cells]
            labels = []
            if self.mode != "test":
                # Vectorized calculation of insertion indices
                # For each md_rank, count how many code_ranks are smaller
                if len(code_ranks) > 0:
                    # searchsorted finds indices where elements should be inserted to maintain order
                    labels = np.searchsorted(code_ranks, md_ranks)
                else:
                    labels = np.zeros(len(md_ranks), dtype=np.int64)
            else:
                # Dummy labels for test
                labels = np.zeros(len(md_ranks), dtype=np.int64)

            self.samples.append(
                {
                    "id": nb_id,
                    "code_embeddings": torch.tensor(
                        code_embeddings, dtype=torch.float32
                    ),
                    "md_embeddings": torch.tensor(md_embeddings, dtype=torch.float32),
                    "labels": torch.tensor(labels, dtype=torch.long),
                    "md_ids": md_ids.tolist(),
                }
            )

        # Cleanup
        del df
        del grouped

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    """
    Custom collator to handle variable length sequences of code and markdown cells.
    """
    # Unpack batch
    code_embeddings = [item["code_embeddings"] for item in batch]
    md_embeddings = [item["md_embeddings"] for item in batch]
    labels = [item["labels"] for item in batch]
    md_ids = [item["md_ids"] for item in batch]
    nb_ids = [item["id"] for item in batch]

    # 1. Pad Code Embeddings (Batch, Max_Code_Len, Hidden)
    # batch_first=True -> (B, L, D)
    padded_code = pad_sequence(code_embeddings, batch_first=True, padding_value=0.0)

    # Create Code Attention Mask (1 for valid, 0 for pad)
    # Shape: (Batch, Max_Code_Len)
    code_lens = torch.tensor([len(x) for x in code_embeddings], dtype=torch.long)
    max_code_len = padded_code.size(1)
    code_mask = torch.arange(max_code_len).expand(
        len(batch), max_code_len
    ) < code_lens.unsqueeze(1)

    # 2. Pad Markdown Embeddings (Batch, Max_Md_Len, Hidden)
    padded_md = pad_sequence(md_embeddings, batch_first=True, padding_value=0.0)

    # Create Markdown Attention Mask
    md_lens = torch.tensor([len(x) for x in md_embeddings], dtype=torch.long)
    max_md_len = padded_md.size(1)
    md_mask = torch.arange(max_md_len).expand(
        len(batch), max_md_len
    ) < md_lens.unsqueeze(1)

    # 3. Pad Labels (Batch, Max_Md_Len)
    # Use -100 for ignore_index in CrossEntropyLoss
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=-100)

    return {
        "id": nb_ids,
        "code_embeddings": padded_code,
        "code_mask": code_mask,
        "md_embeddings": padded_md,
        "md_mask": md_mask,
        "labels": padded_labels,
        "md_ids": md_ids,
    }


def get_dataloader(
    features_path, batch_size=Config.BATCH_SIZE, shuffle=True, mode="train"
):
    """
    Factory function to create a DataLoader.

    Args:
        features_path (str): Path to parquet file.
        batch_size (int): Batch size.
        shuffle (bool): Whether to shuffle the data.
        mode (str): 'train', 'val', or 'test'.

    Returns:
        DataLoader: Configured PyTorch DataLoader.
    """
    dataset = NotebookDataset(features_path, mode=mode)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True if torch.cuda.is_available() else False,
    )
