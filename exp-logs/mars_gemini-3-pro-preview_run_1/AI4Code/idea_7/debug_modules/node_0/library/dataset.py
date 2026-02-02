import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from library.config import Config


class NotebookDataset(Dataset):
    """
    Dataset for loading precomputed notebook features.
    Groups cells by notebook ID, separates Code and Markdown,
    and computes relative ordering labels.
    """

    def __init__(self, features_path, is_test=False):
        self.config = Config
        self.is_test = is_test
        self.data = []

        if not os.path.exists(features_path):
            raise FileNotFoundError(
                f"Features file not found at {features_path}. Please run Preprocessor first."
            )

        print(f"Loading features from {features_path}...")
        # Load the dataframe
        df = pd.read_parquet(features_path)

        # Group by notebook ID to reconstruct notebook structure
        print("Grouping data by notebook ID...")
        grouped = df.groupby("id")

        print(f"Constructing dataset with {len(grouped)} notebooks...")

        for nb_id, group in grouped:
            # Separate Code and Markdown
            code_df = group[group["cell_type"] == "code"].copy()
            md_df = group[group["cell_type"] == "markdown"].copy()

            # Handle Code Cell Ordering
            if self.is_test:
                # For test set, code cells are provided in the correct relative order in the source.
                # We enforce a sequential rank to ensure stability.
                # The groupby preserves the insertion order from the preprocessor.
                code_df["rank"] = range(len(code_df))
            else:
                # For train/val, sort by ground truth rank
                code_df = code_df.sort_values("rank")

            # Parse Embeddings (Bytes -> Float32)
            # We use a list comprehension for robustness, though vectorization is possible
            if len(code_df) > 0:
                code_embeddings = np.stack(
                    [
                        np.frombuffer(b, dtype=np.float32)
                        for b in code_df["embedding"].values
                    ]
                )
                code_ranks = code_df["rank"].values.astype(np.float32)
            else:
                code_embeddings = np.zeros(
                    (0, self.config.EMBEDDING_DIM), dtype=np.float32
                )
                code_ranks = np.array([], dtype=np.float32)

            if len(md_df) > 0:
                md_embeddings = np.stack(
                    [
                        np.frombuffer(b, dtype=np.float32)
                        for b in md_df["embedding"].values
                    ]
                )
                md_ids = md_df["cell_id"].tolist()
                if not self.is_test:
                    md_ranks = md_df["rank"].values.astype(np.float32)
                else:
                    md_ranks = None
            else:
                md_embeddings = np.zeros(
                    (0, self.config.EMBEDDING_DIM), dtype=np.float32
                )
                md_ids = []
                md_ranks = np.array([], dtype=np.float32) if not self.is_test else None

            # Apply Truncation based on Config
            if len(code_embeddings) > self.config.MAX_CODE_SEQ_LEN:
                code_embeddings = code_embeddings[: self.config.MAX_CODE_SEQ_LEN]
                code_ranks = code_ranks[: self.config.MAX_CODE_SEQ_LEN]

            if len(md_embeddings) > self.config.MAX_MD_SEQ_LEN:
                md_embeddings = md_embeddings[: self.config.MAX_MD_SEQ_LEN]
                md_ids = md_ids[: self.config.MAX_MD_SEQ_LEN]
                if md_ranks is not None:
                    md_ranks = md_ranks[: self.config.MAX_MD_SEQ_LEN]

            # Skip empty notebooks
            if len(code_embeddings) == 0 and len(md_embeddings) == 0:
                continue

            self.data.append(
                {
                    "id": nb_id,
                    "code_embeddings": code_embeddings,
                    "code_ranks": code_ranks,
                    "md_embeddings": md_embeddings,
                    "md_ranks": md_ranks,
                    "md_ids": md_ids,
                }
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        code_emb = torch.tensor(sample["code_embeddings"], dtype=torch.float32)
        md_emb = torch.tensor(sample["md_embeddings"], dtype=torch.float32)

        # Compute Labels
        # Target: Index of the code cell that immediately follows the markdown cell.
        # Range: 0 to len(code_emb).
        # len(code_emb) represents the position after the last code cell (EOS).
        labels = torch.zeros(len(md_emb), dtype=torch.long)

        if not self.is_test and len(code_emb) > 0:
            c_ranks = sample["code_ranks"]
            m_ranks = sample["md_ranks"]

            # np.searchsorted(a, v, side='right') returns the index i such that a[i-1] <= v < a[i]
            # We want the first code cell with rank > markdown rank.
            indices = np.searchsorted(c_ranks, m_ranks, side="right")
            labels = torch.tensor(indices, dtype=torch.long)
        elif not self.is_test and len(code_emb) == 0:
            # If no code cells, all markdown cells map to index 0 (which is effectively EOS here)
            labels = torch.zeros(len(md_emb), dtype=torch.long)

        return {
            "id": sample["id"],
            "code_emb": code_emb,
            "md_emb": md_emb,
            "labels": labels,
            "md_ids": sample["md_ids"],
        }


def custom_collate_fn(batch):
    """
    Collates a list of notebook samples into a batch.
    Pads sequences and generates attention masks.
    """
    ids = [x["id"] for x in batch]
    md_ids = [x["md_ids"] for x in batch]

    # --- Process Code Sequence (Anchors) ---
    code_seqs = [x["code_emb"] for x in batch]
    code_lens = torch.tensor([len(x) for x in code_seqs], dtype=torch.long)

    # Pad code sequences
    padded_code = pad_sequence(code_seqs, batch_first=True, padding_value=0.0)

    # --- Process Markdown Set (Queries) ---
    md_seqs = [x["md_emb"] for x in batch]
    md_lens = torch.tensor([len(x) for x in md_seqs], dtype=torch.long)

    # Pad markdown sequences
    padded_md = pad_sequence(md_seqs, batch_first=True, padding_value=0.0)

    # --- Process Labels ---
    labels_list = [x["labels"] for x in batch]
    # Use -100 for padding in labels (standard ignore_index for CrossEntropy)
    padded_labels = pad_sequence(labels_list, batch_first=True, padding_value=-100)

    # --- Generate Masks ---
    # Mask is True for valid tokens, False for padding
    B, Max_Code = padded_code.shape[:2]
    B, Max_Md = padded_md.shape[:2]

    # Create masks using broadcasting
    # shape: (B, Max_Len)
    code_mask = torch.arange(Max_Code, device=padded_code.device).expand(
        B, Max_Code
    ) < code_lens.unsqueeze(1)
    md_mask = torch.arange(Max_Md, device=padded_md.device).expand(
        B, Max_Md
    ) < md_lens.unsqueeze(1)

    return {
        "ids": ids,
        "md_ids": md_ids,
        "code_emb": padded_code,
        "md_emb": padded_md,
        "code_mask": code_mask,
        "md_mask": md_mask,
        "labels": padded_labels,
        "code_lens": code_lens,
        "md_lens": md_lens,
    }
