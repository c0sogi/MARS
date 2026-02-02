import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from library.config import Config


class NotebookSequenceDataset(Dataset):
    """
    PyTorch Dataset for the Deep Sequence-Aware Position Regressor.

    Each sample corresponds to a single markdown cell (Query) and the sequence
    of code cells (Context) from the same notebook.

    Target: The relative rank of the markdown cell within the code sequence [0, 1].
    """

    def __init__(
        self, split: str, load_cached_data: bool = True, debug_limit: int = None
    ):
        """
        Args:
            split (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to load processed samples from disk cache.
            debug_limit (int): Limit number of notebooks for debugging.
        """
        self.split = split
        self.max_seq_len = Config.MAX_SEQ_LEN
        self.embed_dim = Config.EMBED_DIM

        # Ensure working directory exists
        os.makedirs(Config.WORKING_DIR, exist_ok=True)

        # Path for caching the processed list of samples
        # We cache the list of dictionaries to avoid re-grouping/processing pandas dataframes
        cache_filename = f"dataset_samples_{split}.pt"
        if debug_limit:
            cache_filename = f"dataset_samples_{split}_debug{debug_limit}.pt"
        self.cache_path = os.path.join(Config.WORKING_DIR, cache_filename)

        if load_cached_data and os.path.exists(self.cache_path):
            print(f"Loading processed dataset from {self.cache_path}...")
            try:
                self.samples = torch.load(self.cache_path)
            except Exception as e:
                print(f"Failed to load dataset cache: {e}. Reprocessing...")
                self.samples = self._process_and_cache(debug_limit)
        else:
            self.samples = self._process_and_cache(debug_limit)

        print(f"Dataset ({split}) loaded with {len(self.samples)} samples.")

    def _process_and_cache(self, debug_limit):
        """
        Loads features from Parquet, processes them into explicit features for LightGBM, and saves to cache.
        Cite solution_lesson_node_00004: Use similarity-weighted positional statistics.
        """
        print(f"Processing dataset for split: {self.split}...")
        from sklearn.metrics.pairwise import cosine_similarity

        # Identify source file
        if self.split == "train":
            parquet_path = Config.TRAIN_CACHE_PATH
        elif self.split == "val":
            parquet_path = Config.VAL_CACHE_PATH
        elif self.split == "test":
            parquet_path = Config.TEST_CACHE_PATH
        else:
            raise ValueError(f"Unknown split: {self.split}")

        if not os.path.exists(parquet_path):
            raise FileNotFoundError(
                f"Feature file not found: {parquet_path}. Run feature extraction first."
            )

        # Load Data
        df = pd.read_parquet(parquet_path)

        if debug_limit:
            nb_ids = df["notebook_id"].unique()[:debug_limit]
            df = df[df["notebook_id"].isin(nb_ids)]

        samples = []

        # Helper to preserve order for test set
        df["orig_index"] = np.arange(len(df))

        # Group by notebook
        grouped = df.groupby("notebook_id")

        for nb_id, group in grouped:
            # Separate Code and Markdown
            code_df = group[group["cell_type"] == "code"]
            md_df = group[group["cell_type"] == "markdown"]

            # Determine Context Sequence (Code Cells)
            if self.split in ["train", "val"]:
                code_df = code_df.sort_values("rank")
            else:
                code_df = code_df.sort_values("orig_index")

            # Extract embeddings
            if len(code_df) > 0:
                code_embeddings = np.stack(code_df["embedding"].values).astype(
                    np.float32
                )
                code_ranks = code_df["rank"].values
            else:
                code_embeddings = np.zeros((0, self.embed_dim), dtype=np.float32)
                code_ranks = np.array([])

            n_code = len(code_embeddings)

            # Create a sample for each markdown cell
            for _, row in md_df.iterrows():
                md_emb = np.array(row["embedding"], dtype=np.float32)
                cell_id = row["cell_id"]

                # Calculate Target
                target = 0.0
                if self.split in ["train", "val"]:
                    md_rank = row["rank"]
                    if n_code > 0:
                        n_before = np.sum(code_ranks < md_rank)
                        target = float(n_before) / float(n_code)
                    else:
                        target = 0.5
                else:
                    target = -1.0

                # Feature Engineering (Cite solution_lesson_node_00004)
                if n_code > 0:
                    # Compute cosine similarity: (1, embed_dim) x (n_code, embed_dim).T -> (1, n_code)
                    sims = cosine_similarity(md_emb.reshape(1, -1), code_embeddings)[0]

                    sim_max = float(sims.max())
                    best_match_loc = float(np.argmax(sims)) / n_code
                    sim_mean = float(sims.mean())
                    sim_std = float(sims.std())

                    # Center of Mass
                    # Use ReLU to ignore negative similarities for weighting
                    weights = np.maximum(sims, 0)
                    if weights.sum() > 1e-6:
                        com = np.average(np.arange(n_code), weights=weights) / n_code
                    else:
                        com = 0.5
                else:
                    # Fallback for notebooks with no code cells
                    sim_max = 0.0
                    best_match_loc = 0.5
                    sim_mean = 0.0
                    sim_std = 0.0
                    com = 0.5

                samples.append(
                    {
                        "nb_id": nb_id,
                        "cell_id": cell_id,
                        "features": np.array(
                            [sim_max, best_match_loc, sim_mean, sim_std, com],
                            dtype=np.float32,
                        ),
                        "target": target,
                    }
                )

        # Save to cache
        print(f"Saving processed dataset to {self.cache_path}...")
        torch.save(samples, self.cache_path)

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Return features as tensor for compatibility with DataLoader (though we might convert back to numpy later)
        features = torch.tensor(sample["features"], dtype=torch.float32)
        label = torch.tensor(sample["target"], dtype=torch.float32)
        ids = (sample["nb_id"], sample["cell_id"])

        return {
            "features": features,
            "label": label,
            "ids": ids,
        }
