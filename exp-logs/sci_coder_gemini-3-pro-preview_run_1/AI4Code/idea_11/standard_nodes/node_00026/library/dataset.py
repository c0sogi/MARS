import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from library.config import Config


class CachedNotebookDataset(Dataset):
    """
    Dataset class for the Corrected Dual-Context Anchor Network (DC-AN).
    Loads pre-computed embeddings from Parquet files and structures them
    into (Code Sequence, Markdown Set) pairs with corresponding ranking labels.
    """

    def __init__(self, features_path, split="train"):
        """
        Args:
            features_path (str): Path to the Parquet file containing cached features.
            split (str): 'train', 'val', or 'test'. Determines label generation behavior.
        """
        self.split = split
        self.samples = []

        if not os.path.exists(features_path):
            raise FileNotFoundError(f"Features file not found at {features_path}")

        print(f"Loading features from {features_path}...")
        # Load the dataframe
        df = pd.read_parquet(features_path)

        # Handle Debug Mode: Slice the dataframe directly to save processing time
        if Config.DEBUG:
            print(
                f"DEBUG mode active: limiting dataset to {Config.SAMPLE_SIZE} notebooks."
            )
            # We filter by unique IDs to ensure we get complete notebooks
            unique_ids = df["id"].unique()[: Config.SAMPLE_SIZE]
            df = df[df["id"].isin(unique_ids)].copy()

        print(f"Constructing dataset for {split} split...")

        # Sort by ID and Rank to ensure deterministic order and correct label generation
        # For 'test', rank is -1, so the sort order relies on the original insertion order (stable sort)
        # or the 'cell_id' if ranks are tied, but we primarily trust the input order for code cells.
        if split != "test":
            df = df.sort_values(by=["id", "rank"])
        else:
            # Ensure we don't scramble the test set structure unintentionally,
            # though code cell order is fixed in input.
            df = df.sort_values(by=["id"])

        # Group by Notebook ID
        grouped = df.groupby("id")

        for nb_id, group in grouped:
            # Separate Code and Markdown cells
            code_df = group[group["cell_type"] == "code"]
            md_df = group[group["cell_type"] == "markdown"]

            # Extract Embeddings
            # Convert list of floats (from Parquet) to Tensor
            if len(code_df) > 0:
                # np.stack is necessary because the column contains lists/arrays
                code_embeddings = torch.tensor(
                    np.stack(code_df["embedding"].tolist()), dtype=torch.float32
                )
            else:
                # Handle edge case: notebook with no code cells
                # Shape: (0, 768) - assuming MPNet dim
                code_embeddings = torch.empty((0, 768), dtype=torch.float32)

            if len(md_df) > 0:
                md_embeddings = torch.tensor(
                    np.stack(md_df["embedding"].tolist()), dtype=torch.float32
                )
            else:
                md_embeddings = torch.empty((0, 768), dtype=torch.float32)

            sample = {
                "id": nb_id,
                "code_embeddings": code_embeddings,
                "md_embeddings": md_embeddings,
                "md_cell_ids": md_df["cell_id"].tolist(),
                "code_cell_ids": code_df["cell_id"].tolist(),
            }

            # Generate Labels for Training/Validation
            if split != "test":
                code_ranks = code_df["rank"].values
                md_ranks = md_df["rank"].values
                labels = []

                # For each markdown cell, determine the index of the code cell that immediately follows it.
                # If the markdown cell is after all code cells, the label is len(code_cells) (pointing to EOS).
                # np.searchsorted finds the first index where elements > value.
                # Since code_ranks is sorted, this gives exactly the index in the code sequence (0 to N_code).
                if len(code_ranks) > 0:
                    labels = np.searchsorted(code_ranks, md_ranks)
                else:
                    # If no code cells, all labels are 0 (which is the EOS index for length 0)
                    labels = np.zeros(len(md_ranks), dtype=int)

                sample["labels"] = torch.tensor(labels, dtype=torch.long)

            self.samples.append(sample)

        print(f"Loaded {len(self.samples)} notebooks.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    @staticmethod
    def collate_fn(batch):
        """
        Custom collate function to handle variable lengths of code sequences and markdown sets.
        """
        ids = [s["id"] for s in batch]
        md_cell_ids = [s["md_cell_ids"] for s in batch]
        code_cell_ids = [s["code_cell_ids"] for s in batch]

        # --- Process Code Anchors (Sequence) ---
        code_seqs = [s["code_embeddings"] for s in batch]
        # Calculate lengths for dynamic EOS insertion in the model
        code_lens = torch.tensor([len(seq) for seq in code_seqs], dtype=torch.long)
        # Pad sequence: (Batch, Max_Code_Len, Dim)
        code_emb_padded = pad_sequence(code_seqs, batch_first=True, padding_value=0.0)

        # Create Code Attention Mask
        # True indicates padding (ignored), False indicates valid token
        batch_size = len(batch)
        max_code_len = code_emb_padded.size(1)
        code_mask = torch.ones((batch_size, max_code_len), dtype=torch.bool)
        for i, l in enumerate(code_lens):
            code_mask[i, :l] = False

        # --- Process Markdown Queries (Set) ---
        md_seqs = [s["md_embeddings"] for s in batch]
        md_lens = torch.tensor([len(seq) for seq in md_seqs], dtype=torch.long)
        # Pad sequence: (Batch, Max_MD_Len, Dim)
        md_emb_padded = pad_sequence(md_seqs, batch_first=True, padding_value=0.0)

        # Create Markdown Attention Mask (for Set Transformer)
        max_md_len = md_emb_padded.size(1)
        md_mask = torch.ones((batch_size, max_md_len), dtype=torch.bool)
        for i, l in enumerate(md_lens):
            md_mask[i, :l] = False

        # --- Process Labels ---
        labels_padded = None
        if "labels" in batch[0]:
            labels_list = [s["labels"] for s in batch]
            # Pad labels with -100 (standard ignore_index for CrossEntropyLoss)
            labels_padded = pad_sequence(
                labels_list, batch_first=True, padding_value=-100
            )

        return {
            "ids": ids,
            "md_cell_ids": md_cell_ids,
            "code_cell_ids": code_cell_ids,
            "code_embeddings": code_emb_padded,
            "code_mask": code_mask,
            "code_lens": code_lens,
            "md_embeddings": md_emb_padded,
            "md_mask": md_mask,
            "md_lens": md_lens,
            "labels": labels_padded,
        }
