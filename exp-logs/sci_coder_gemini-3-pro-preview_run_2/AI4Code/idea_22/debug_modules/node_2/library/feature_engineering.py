import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from library.config import Config
from library.vectorization import TextPipeline


class FeatureExtractor:
    """
    Implements Content-Aware Neighbor Projection for feature engineering.
    Extracts positional and semantic features by analyzing the relationship
    between markdown cells and their nearest code cell neighbors in SVD space.
    """

    def __init__(self):
        self.working_dir = Config.WORKING_DIR
        os.makedirs(self.working_dir, exist_ok=True)

    def extract_features(self, df, text_pipeline, mode, load_cached_data=True):
        """
        Main method to generate or load features for a dataset.

        Args:
            df (pd.DataFrame): The dataframe containing notebook cells (from NotebookLoader).
            text_pipeline (TextPipeline): A fitted TextPipeline instance.
            mode (str): One of 'train', 'val', 'test'. Used for cache naming and logic.
            load_cached_data (bool): Whether to attempt loading from cache.

        Returns:
            pd.DataFrame: A dataframe containing the extracted features.
        """
        # Determine cache path based on mode
        if mode == "train":
            cache_path = Config.TRAIN_FEATURES_PATH
        elif mode == "val":
            cache_path = Config.VAL_FEATURES_PATH
        elif mode == "test":
            cache_path = Config.TEST_FEATURES_PATH
        else:
            raise ValueError(f"Invalid mode: {mode}")

        # 1. Check Cache
        if load_cached_data and os.path.exists(cache_path):
            print(f"[{mode}] Loading cached features from {cache_path}")
            return pd.read_parquet(cache_path)

        print(f"[{mode}] Generating features from scratch...")

        # 2. Pre-compute SVD vectors for all cells
        # This is much faster than doing it per-notebook
        print(f"[{mode}] Transforming all cells to SVD space...")
        all_sources = df["source"].astype(str).fillna("")
        all_svd = text_pipeline.transform_cells(all_sources)

        # Map original dataframe index to SVD array index
        # Reset index to ensure alignment if df has gaps
        df = df.reset_index(drop=True)

        # 3. Process Notebooks
        features_list = []

        # Group by notebook to process interactions
        # We assume df is sorted or grouped by 'id' effectively
        grouped = df.groupby("id")

        print(f"[{mode}] Processing {df['id'].nunique()} notebooks...")

        # Constants from Config
        K = Config.NEIGHBOR_K
        CONTENT_DIMS = Config.CONTENT_PROJECTION_DIMS

        for nb_id, group in grouped:
            # Indices in the global SVD matrix
            global_indices = group.index.values

            # Local data
            cell_types = group["cell_type"].values
            cell_ids = group["cell_id"].values

            # Identify Code and Markdown indices (local to the group)
            is_code = cell_types == "code"
            is_md = cell_types == "markdown"

            code_indices = np.where(is_code)[0]
            md_indices = np.where(is_md)[0]

            # Skip notebooks with no code cells (rare edge case)
            if len(code_indices) == 0:
                # Assign default features for all markdown cells in this notebook
                for local_idx in md_indices:
                    row_dict = self._get_default_features(
                        nb_id, cell_ids[local_idx], CONTENT_DIMS
                    )
                    features_list.append(row_dict)
                continue

            # Skip notebooks with no markdown cells
            if len(md_indices) == 0:
                continue

            # Get Code Cell Ranks
            # For Train/Val: Use the ground truth 'pct_rank' provided in df
            # For Test: Infer ranks based on sequence order (Code cells are ordered)
            total_cells = len(group)

            if mode == "test":
                # In test set, code cells are in correct order.
                # We assign them equidistant ranks across the notebook length.
                # rank 0, 1, 2... for code cells
                raw_code_ranks = np.arange(len(code_indices))
                # Normalize by total cells - 1
                if total_cells > 1:
                    code_ranks = raw_code_ranks / (total_cells - 1.0)
                else:
                    code_ranks = np.zeros(len(code_indices))
            else:
                # Use ground truth
                code_ranks = group.iloc[code_indices]["pct_rank"].values

            # Get SVD vectors
            # shape: (n_code, n_components)
            code_svd = all_svd[global_indices[code_indices]]
            # shape: (n_md, n_components)
            md_svd = all_svd[global_indices[md_indices]]

            # Compute Cosine Similarity Matrix: (n_md, n_code)
            # We want to find which code cells are closest to each markdown cell
            sim_matrix = cosine_similarity(md_svd, code_svd)

            # Context Features
            md_ratio = len(md_indices) / total_cells

            # Iterate over each markdown cell in this notebook
            for i, local_md_idx in enumerate(md_indices):
                cell_id = cell_ids[local_md_idx]

                # Get similarities for this markdown cell
                sims = sim_matrix[i]

                # Find Top-K nearest code cells
                # argsort is ascending, so take last K and reverse
                if len(sims) >= K:
                    top_k_indices = np.argsort(sims)[-K:][::-1]
                else:
                    # If fewer than K code cells, take all sorted by sim
                    top_k_indices = np.argsort(sims)[::-1]

                # --- Feature 1: Positional Anchors ---
                # Get the ranks of these neighbors
                neighbor_ranks = code_ranks[top_k_indices]

                mean_rank = np.mean(neighbor_ranks)
                std_rank = np.std(neighbor_ranks) if len(neighbor_ranks) > 1 else 0.0

                # --- Feature 2: Content Anchors ---
                # Get the SVD vector of the single nearest neighbor (Top-1)
                best_match_idx = top_k_indices[0]
                best_match_svd = code_svd[best_match_idx]

                # Extract top dimensions
                content_features = best_match_svd[:CONTENT_DIMS]

                # --- Feature 3: Self Content ---
                # Also include the markdown cell's own SVD (top dims)
                self_svd_features = md_svd[i][:CONTENT_DIMS]

                # Construct Feature Dictionary
                feat_dict = {
                    "id": nb_id,
                    "cell_id": cell_id,
                    "total_cells": total_cells,
                    "md_ratio": md_ratio,
                    "neighbor_rank_mean": mean_rank,
                    "neighbor_rank_std": std_rank,
                }

                # Add neighbor content features
                for d in range(CONTENT_DIMS):
                    feat_dict[f"neighbor_content_{d}"] = content_features[d]

                # Add self content features
                for d in range(CONTENT_DIMS):
                    feat_dict[f"md_content_{d}"] = self_svd_features[d]

                # Add target if available (for training)
                if mode != "test":
                    feat_dict["target_rank"] = group.iloc[local_md_idx]["pct_rank"]

                features_list.append(feat_dict)

        # 4. Create DataFrame
        feature_df = pd.DataFrame(features_list)

        # 5. Save to Cache
        print(f"[{mode}] Saving {len(feature_df)} rows to {cache_path}")
        feature_df.to_parquet(cache_path, index=False)

        return feature_df

    def _get_default_features(self, nb_id, cell_id, content_dims):
        """
        Helper to generate default features for edge cases (e.g., no code cells).
        """
        feat_dict = {
            "id": nb_id,
            "cell_id": cell_id,
            "total_cells": 1,
            "md_ratio": 1.0,
            "neighbor_rank_mean": 0.5,
            "neighbor_rank_std": 0.0,
        }
        for d in range(content_dims):
            feat_dict[f"neighbor_content_{d}"] = 0.0
            feat_dict[f"md_content_{d}"] = 0.0

        return feat_dict
