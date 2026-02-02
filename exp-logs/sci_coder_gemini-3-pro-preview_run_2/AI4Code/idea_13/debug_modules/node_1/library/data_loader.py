import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from library.config import Config
from library.utils import preprocess_data


def load_metadata(split="train"):
    """
    Loads the metadata CSV for the specified split.

    Args:
        split (str): One of 'train', 'val', 'test'.

    Returns:
        pd.DataFrame: The metadata dataframe.
    """
    if split == "train":
        path = Config.TRAIN_METADATA_PATH
    elif split == "val":
        path = Config.VAL_METADATA_PATH
    elif split == "test":
        path = Config.TEST_METADATA_PATH
    else:
        raise ValueError(f"Unknown split: {split}")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found at {path}")

    return pd.read_csv(path)


class NotebookProcessor:
    """
    Handles loading and processing of notebook data from JSON to flat DataFrame.
    Wraps the library.utils.preprocess_data function with specific configuration logic.
    """

    def __init__(self, config=Config):
        self.config = config

    def load_data(self, split="train", load_cached_data=True):
        """
        Loads the processed dataframe for the given split.

        Args:
            split (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to use cached parquet files.

        Returns:
            pd.DataFrame: Processed dataframe with cell-level information.
        """
        # Determine paths based on split
        if split == "train":
            meta_path = self.config.TRAIN_METADATA_PATH
            out_path = self.config.TRAIN_DATAFRAME_PATH
        elif split == "val":
            meta_path = self.config.VAL_METADATA_PATH
            out_path = self.config.VAL_DATAFRAME_PATH
        elif split == "test":
            meta_path = self.config.TEST_METADATA_PATH
            out_path = self.config.TEST_DATAFRAME_PATH
        else:
            raise ValueError(f"Unknown split: {split}")

        # Check for debug overrides
        debug = self.config.DEBUG
        debug_size = self.config.DEBUG_SAMPLE_SIZE

        # Call the provided utility function
        df = preprocess_data(
            metadata_path=meta_path,
            output_path=out_path,
            load_cached_data=load_cached_data,
            debug=debug,
            debug_size=debug_size,
        )

        # Ensure global index is preserved for feature alignment
        # We reset index to ensure 0..N-1 alignment with SVD matrices
        df = df.reset_index(drop=True)

        return df


class MetricLearningDataset(Dataset):
    """
    PyTorch Dataset for Stage 2: Supervised Metric Learning.
    Generates pairs of (Markdown, Code) cells.
    - Positive Pair: Markdown cell and an adjacent Code cell.
    - Negative Pair: Markdown cell and a random non-adjacent Code cell from the same notebook.
    """

    def __init__(self, df, features, mode="train"):
        """
        Args:
            df (pd.DataFrame): The dataframe containing cell metadata (must be aligned with features).
            features (np.ndarray or torch.Tensor): The SVD feature matrix (N_cells x SVD_dim).
            mode (str): 'train' or 'val'. If 'train', generates negatives.
        """
        self.features = (
            torch.tensor(features, dtype=torch.float32)
            if not torch.is_tensor(features)
            else features
        )
        self.mode = mode
        self.pairs = []  # List of (idx_markdown, idx_code, label)

        print(f"Generating pairs for {mode} set...")
        self._generate_pairs(df)
        print(f"Generated {len(self.pairs)} pairs.")

    def _generate_pairs(self, df):
        # 1. Setup indices and shifting for adjacency check
        # We need to preserve the original index to map back to self.features
        df = df.copy()
        df["original_index"] = df.index

        # Sort by notebook and rank to ensure correct adjacency
        # Note: For test set, rank might be NaN, but this dataset is for training/val where rank exists.
        if "rank" not in df.columns or df["rank"].isnull().all():
            # Fallback for inference or if ranks missing: cannot generate supervised pairs
            return

        df = df.sort_values(["id", "rank"])

        # Shift to find neighbors
        # We group by 'id' to prevent boundary leakage between notebooks
        grouped = df.groupby("id")
        df["prev_type"] = grouped["cell_type"].shift(1)
        df["next_type"] = grouped["cell_type"].shift(-1)
        df["prev_idx"] = grouped["original_index"].shift(1)
        df["next_idx"] = grouped["original_index"].shift(-1)

        # 2. Identify Positives (Markdown with adjacent Code)
        # Condition: Current is Markdown
        is_md = df["cell_type"] == "markdown"

        # Neighbor is Code
        prev_is_code = df["prev_type"] == "code"
        next_is_code = df["next_type"] == "code"

        # Extract indices
        pos_prev = df[is_md & prev_is_code][["original_index", "prev_idx", "id"]]
        pos_next = df[is_md & next_is_code][["original_index", "next_idx", "id"]]

        # Rename columns for consistency
        pos_prev.columns = ["md_idx", "code_idx", "id"]
        pos_next.columns = ["md_idx", "code_idx", "id"]

        # Combine all positive pairs
        positives = pd.concat([pos_prev, pos_next], ignore_index=True)

        # Convert to integer indices
        positives["md_idx"] = positives["md_idx"].astype(int)
        positives["code_idx"] = positives["code_idx"].astype(int)

        # Add to pairs list with label 1
        # We store as list of tuples for speed in __getitem__
        pos_list = list(
            zip(positives["md_idx"], positives["code_idx"], [1] * len(positives))
        )
        self.pairs.extend(pos_list)

        # 3. Identify Negatives (Markdown with random Code from SAME notebook)
        if self.mode == "train":
            num_neg = Config.METRIC_NEGATIVES_PER_POSITIVE

            # Pre-compute map: notebook_id -> list of code cell indices
            # This is much faster than filtering for every pair
            code_cells = df[df["cell_type"] == "code"]
            nb_to_code = (
                code_cells.groupby("id")["original_index"].apply(np.array).to_dict()
            )

            # We iterate through positives to generate negatives
            # This ensures we have balanced anchors
            neg_pairs = []

            # Convert positives to a set of tuples for fast lookup to avoid accidental collisions
            # (md_idx, code_idx)
            pos_set = set(zip(positives["md_idx"], positives["code_idx"]))

            # Iterate over unique (md_idx, nb_id) to avoid over-sampling if a MD has 2 code neighbors
            # Actually, we want negatives for every positive sample we generated
            # to maintain the batch structure for contrastive loss.

            # Optimization: Vectorized sampling is hard with variable group sizes.
            # We use a loop but optimize the lookup.
            rng = np.random.default_rng(Config.SEED)

            # We can process by notebook groups to minimize dictionary lookups
            # But iterating 2M pairs is okay.

            for md_idx, pos_code_idx, nb_id in zip(
                positives["md_idx"], positives["code_idx"], positives["id"]
            ):
                if nb_id not in nb_to_code:
                    continue

                candidates = nb_to_code[nb_id]
                if len(candidates) < 2:
                    continue

                # Sample k candidates
                # We sample k+1 to be safe against picking the positive one
                samples = rng.choice(
                    candidates, size=min(len(candidates), num_neg + 2), replace=False
                )

                count = 0
                for s_idx in samples:
                    if s_idx == pos_code_idx:
                        continue
                    if (md_idx, s_idx) in pos_set:
                        continue

                    neg_pairs.append((md_idx, s_idx, 0))
                    count += 1
                    if count >= num_neg:
                        break

            self.pairs.extend(neg_pairs)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        """
        Returns:
            x1 (tensor): Feature vector of the Markdown cell (Anchor).
            x2 (tensor): Feature vector of the Code cell (Positive/Negative).
            label (tensor): 1.0 if adjacent, 0.0 otherwise.
        """
        md_idx, code_idx, label = self.pairs[idx]

        x1 = self.features[md_idx]
        x2 = self.features[code_idx]
        y = torch.tensor(label, dtype=torch.float32)

        return x1, x2, y
