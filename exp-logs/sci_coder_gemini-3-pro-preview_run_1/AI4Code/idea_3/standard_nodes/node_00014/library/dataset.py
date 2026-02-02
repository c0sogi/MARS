import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from library.config import FEATURE_DIR, MAX_ANCHOR_SEQ_LEN
from library.preprocessor import Preprocessor


class NotebookDataset(Dataset):
    def __init__(self, split="train", load_cached_data=True, debug=False):
        """
        PyTorch Dataset for the Contextualized Anchor-Attention Network.

        Args:
            split (str): 'train', 'val', or 'test'.
            load_cached_data (bool): If True, attempts to load existing parquet file.
                                     If False or file missing, triggers preprocessing.
            debug (bool): If True, uses a smaller subset of data.
        """
        self.split = split
        self.debug = debug

        # Determine file path
        filename = f"{split}_features.parquet"
        file_path = os.path.join(FEATURE_DIR, filename)

        # Trigger preprocessing if needed
        if not load_cached_data or not os.path.exists(file_path):
            preprocessor = Preprocessor()
            file_path = preprocessor.process_split(
                split=split, load_cached_data=load_cached_data, debug=debug
            )

        # Load data
        # We use pandas to read the parquet file
        df = pd.read_parquet(file_path)

        # Pre-process into memory-efficient structure
        self.samples = []

        # Group by notebook ID.
        # sort=False is faster and preserves the order from metadata (which is randomized/stratified there)
        grouped = df.groupby("id", sort=False)

        for nb_id, group in grouped:
            # Separate Code and Markdown
            # Code cells act as Anchors
            code_mask = group["cell_type"] == "code"
            md_mask = group["cell_type"] == "markdown"

            # Ensure code cells are sorted by their rank
            code_df = group[code_mask].sort_values("rank_in_code")
            md_df = group[md_mask]

            # Extract embeddings
            # The parquet file stores lists, so we stack them into numpy arrays
            # Handle empty cases gracefully (though unlikely in valid data)
            if not code_df.empty:
                code_emb = np.stack(code_df["embedding"].values)
            else:
                code_emb = np.zeros((0, 384), dtype=np.float32)

            if not md_df.empty:
                md_emb = np.stack(md_df["embedding"].values)
                labels = md_df["label"].values.astype(np.int64)
            else:
                md_emb = np.zeros((0, 384), dtype=np.float32)
                labels = np.zeros((0,), dtype=np.int64)

            self.samples.append(
                {"id": nb_id, "code_emb": code_emb, "md_emb": md_emb, "labels": labels}
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        code_emb = sample["code_emb"]
        md_emb = sample["md_emb"]
        labels = sample["labels"]

        # Handle Max Sequence Length for Anchors (Code Cells)
        # If code sequence is too long, we truncate it.
        # This requires adjusting labels that might point beyond the truncation point.
        if len(code_emb) > MAX_ANCHOR_SEQ_LEN:
            code_emb = code_emb[:MAX_ANCHOR_SEQ_LEN]
            # Clamp labels to the new EOS index (which is MAX_ANCHOR_SEQ_LEN)
            # Labels pointing to indices >= MAX_ANCHOR_SEQ_LEN will now point to the end
            labels = np.minimum(labels, MAX_ANCHOR_SEQ_LEN)

        # Convert to Tensors
        item = {
            "code_emb": torch.tensor(code_emb, dtype=torch.float32),
            "md_emb": torch.tensor(md_emb, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.long),
            "id": sample["id"],  # Useful for inference/debugging
        }

        return item
