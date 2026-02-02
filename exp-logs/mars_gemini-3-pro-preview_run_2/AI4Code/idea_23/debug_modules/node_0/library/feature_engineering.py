import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from library.config import Config


class FeatureEngine:
    """
    Handles feature engineering for the Stage 2 Stacking Model.
    Generates content-aware and positional features based on SVD vectors and code anchors.
    """

    def __init__(self):
        self.config = Config
        self.working_dir = self.config.WORKING_DIR
        os.makedirs(self.working_dir, exist_ok=True)

    def create_stage2_features(
        self,
        df: pd.DataFrame,
        svd_features: np.ndarray,
        ridge_predictions: dict,
        mode: str = "train",
        load_cached_data: bool = True,
    ) -> pd.DataFrame:
        """
        Generates features for the Stage 2 LightGBM model.

        Args:
            df: DataFrame containing cell metadata (id, cell_id, cell_type, rank).
            svd_features: Numpy array of SVD vectors corresponding to df rows.
            ridge_predictions: Dictionary mapping cell_id to Stage 1 predicted rank.
            mode: 'train' or 'test'.
            load_cached_data: Whether to load from parquet cache.

        Returns:
            DataFrame containing features for Markdown cells.
        """
        cache_path = os.path.join(self.working_dir, f"{mode}_stage2_features.parquet")

        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading cached Stage 2 features from {cache_path}")
            return pd.read_parquet(cache_path)

        print(f"Generating Stage 2 features for {mode} set...")

        # ---------------------------------------------------------
        # 1. Prepare Data for Vectorized Processing
        # ---------------------------------------------------------
        # We need to process data notebook by notebook. Sorting by ID allows us
        # to use fast numpy slicing instead of slow pandas groupby.

        # Create a mapping to track original indices (to retrieve SVD vectors correctly)
        df["original_index"] = np.arange(len(df))

        # Sort DataFrame by ID
        df_sorted = df.sort_values("id").reset_index(drop=True)

        # Reorder SVD features to match the sorted DataFrame
        sorted_indices = df_sorted["original_index"].values
        svd_sorted = svd_features[sorted_indices]

        # Extract columns to numpy arrays for speed
        ids = df_sorted["id"].values
        cell_types = df_sorted["cell_type"].values
        cell_ids = df_sorted["cell_id"].values
        ranks = df_sorted["rank"].values  # Note: -1.0 for test set

        # Map Ridge predictions to the sorted array
        # Default to 0.5 if prediction missing (robustness)
        ridge_preds_arr = np.array(
            [ridge_predictions.get(cid, 0.5) for cid in cell_ids], dtype=np.float32
        )

        # ---------------------------------------------------------
        # 2. Iterate Through Notebooks
        # ---------------------------------------------------------
        # Identify start and end indices for each notebook
        unique_ids, start_indices = np.unique(ids, return_index=True)
        end_indices = np.append(start_indices[1:], len(ids))

        feature_rows = []

        # Hyperparameters
        K = self.config.TOP_K_POS_ANCHORS
        N_DIMS = self.config.ANCHOR_CONTENT_DIMS

        # Loop over each notebook slice
        for start, end in zip(start_indices, end_indices):
            # Extract notebook-specific data
            nb_cell_types = cell_types[start:end]
            nb_svd = svd_sorted[start:end]
            nb_ranks = ranks[start:end]
            nb_cell_ids = cell_ids[start:end]
            nb_ridge = ridge_preds_arr[start:end]

            # Identify Code and Markdown indices within this slice
            is_md = nb_cell_types == "markdown"
            is_code = nb_cell_types == "code"

            md_indices = np.where(is_md)[0]
            code_indices = np.where(is_code)[0]

            # Skip if no markdown cells (nothing to predict)
            if len(md_indices) == 0:
                continue

            # Metadata Features
            total_cells = end - start
            md_count = len(md_indices)
            md_ratio = md_count / total_cells if total_cells > 0 else 0.0

            # -----------------------------------------------------
            # Case A: No Code Cells (Edge Case)
            # -----------------------------------------------------
            if len(code_indices) == 0:
                # Fallback to defaults since we can't anchor
                for idx in md_indices:
                    row = {
                        "cell_id": nb_cell_ids[idx],
                        "ridge_pred": nb_ridge[idx],
                        "neigh_mean_rank": 0.5,
                        "neigh_std_rank": 0.288,  # Std of uniform dist [0,1]
                        "nb_total_cells": total_cells,
                        "nb_md_ratio": md_ratio,
                        "target": nb_ranks[idx],
                    }
                    # Zero out SVD features
                    for d in range(N_DIMS):
                        row[f"anchor_svd_{d}"] = 0.0
                        row[f"own_svd_{d}"] = nb_svd[idx, d]
                    feature_rows.append(row)
                continue

            # -----------------------------------------------------
            # Case B: Standard Case (Code Anchors Available)
            # -----------------------------------------------------
            # Compute Euclidean distance matrix: Markdown (rows) x Code (cols)
            dists = cdist(nb_svd[md_indices], nb_svd[code_indices], metric="euclidean")

            # Find indices of nearest code cells for each markdown cell
            # argsort along axis 1 (columns)
            sorted_args = np.argsort(dists, axis=1)

            for i, md_idx in enumerate(md_indices):
                # Get the local indices of the top K nearest code cells
                nearest_code_local_indices = sorted_args[i, :K]
                nearest_code_global_indices = code_indices[nearest_code_local_indices]

                # --- Feature 1: Positional Anchors (Mean/Std Rank) ---
                if mode == "test":
                    # In test mode, we don't have ground truth ranks.
                    # We approximate the code skeleton's rank distribution as uniform [0, 1].
                    # Since code_indices are in correct relative order, their index
                    # represents their relative position.
                    n_code = len(code_indices)
                    if n_code > 1:
                        # Map the index in the code sequence to a 0..1 float
                        neighbor_ranks = nearest_code_local_indices / (n_code - 1)
                    else:
                        neighbor_ranks = np.array([0.0])
                else:
                    # In train mode, use the actual ground truth ranks of the code cells
                    neighbor_ranks = nb_ranks[nearest_code_global_indices]

                mean_rank = np.mean(neighbor_ranks)
                std_rank = np.std(neighbor_ranks) if len(neighbor_ranks) > 1 else 0.0

                # --- Feature 2: Anchor Content Injection ---
                # Extract SVD components of the single nearest code neighbor (Top-1)
                top1_local_idx = sorted_args[i, 0]
                top1_global_idx = code_indices[top1_local_idx]
                anchor_svd_vec = nb_svd[top1_global_idx, :N_DIMS]

                # --- Feature 3: Own Content ---
                own_svd_vec = nb_svd[md_idx, :N_DIMS]

                # Construct Feature Row
                row = {
                    "cell_id": nb_cell_ids[md_idx],
                    "ridge_pred": nb_ridge[md_idx],
                    "neigh_mean_rank": mean_rank,
                    "neigh_std_rank": std_rank,
                    "nb_total_cells": total_cells,
                    "nb_md_ratio": md_ratio,
                    "target": nb_ranks[md_idx],
                }

                # Add dense SVD features
                for d in range(N_DIMS):
                    row[f"anchor_svd_{d}"] = anchor_svd_vec[d]
                    row[f"own_svd_{d}"] = own_svd_vec[d]

                feature_rows.append(row)

        # ---------------------------------------------------------
        # 3. Finalize and Save
        # ---------------------------------------------------------
        features_df = pd.DataFrame(feature_rows)

        # Optimize memory usage by downcasting floats
        float_cols = [c for c in features_df.columns if c not in ["cell_id"]]
        features_df[float_cols] = features_df[float_cols].astype(np.float32)

        print(f"Saving features to {cache_path}")
        features_df.to_parquet(cache_path, index=False)

        return features_df
