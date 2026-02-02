import os
import gc
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from library.config import (
    CACHE_DIR,
    NUM_NEIGHBORS,
    NEIGHBOR_RANKS_TO_KEEP,
    SMOOTHING_K,
    SEED,
)


class NeighborhoodFeatureExtractor:
    """
    Implements the Gated Multi-Resolution Neighborhood logic.
    Extracts features based on the similarity between markdown cells and code cells
    within the same notebook, using both Lexical (TF-IDF) and Latent (SVD) views.
    """

    def __init__(self):
        self.num_neighbors = NUM_NEIGHBORS
        self.ranks_to_keep = NEIGHBOR_RANKS_TO_KEEP
        self.smoothing_k = SMOOTHING_K

    def _compute_resolution_stats(
        self, sim_scores: np.ndarray, code_ranks: np.ndarray, prefix: str
    ) -> dict:
        """
        Computes the instance and smoothed features for a single markdown cell
        given its similarity scores to code cells and the code cells' ranks.

        Args:
            sim_scores: Array of similarity scores to code cells.
            code_ranks: Array of normalized ranks of those code cells.
            prefix: Feature prefix ('lex' or 'lat').

        Returns:
            dict: Feature dictionary.
        """
        # Sort by similarity descending
        sort_idx = np.argsort(sim_scores)[::-1]

        # Top-K neighbors
        top_k_idx = sort_idx[: self.num_neighbors]
        top_k_sims = sim_scores[top_k_idx]
        top_k_ranks = code_ranks[top_k_idx]

        feats = {}

        # Instance Features (Specific Neighbors)
        for r_idx in self.ranks_to_keep:
            if r_idx < len(top_k_sims):
                feats[f"{prefix}_sim_{r_idx}"] = float(top_k_sims[r_idx])
                feats[f"{prefix}_rank_{r_idx}"] = float(top_k_ranks[r_idx])
            else:
                # Fallback if not enough code cells
                feats[f"{prefix}_sim_{r_idx}"] = 0.0
                feats[f"{prefix}_rank_{r_idx}"] = 0.5  # Neutral rank

        # Smoothed Features (Statistics over Top-K)
        # We use min(len, smoothing_k) for the window
        smooth_len = min(len(top_k_sims), self.smoothing_k)
        if smooth_len > 0:
            smooth_ranks = top_k_ranks[:smooth_len]
            feats[f"{prefix}_mean_rank"] = float(np.mean(smooth_ranks))
            feats[f"{prefix}_std_rank"] = float(np.std(smooth_ranks))
        else:
            feats[f"{prefix}_mean_rank"] = 0.5
            feats[f"{prefix}_std_rank"] = 0.28  # Approx uniform std

        return feats

    def extract_features(
        self,
        df: pd.DataFrame,
        tfidf_matrix: sparse.spmatrix,
        svd_matrix: np.ndarray,
        partition_name: str,
        load_cached_data: bool = True,
    ) -> pd.DataFrame:
        """
        Main method to extract neighborhood features.

        Args:
            df: DataFrame containing cell metadata.
            tfidf_matrix: Sparse TF-IDF matrix (rows align with df).
            svd_matrix: Dense SVD matrix (rows align with df).
            partition_name: 'train', 'val', or 'test'.
            load_cached_data: Whether to load from cache.

        Returns:
            pd.DataFrame: DataFrame containing the extracted features for MARKDOWN cells.
        """
        os.makedirs(CACHE_DIR, exist_ok=True)
        cache_path = os.path.join(
            CACHE_DIR, f"{partition_name}_neighborhood_features.parquet"
        )

        if load_cached_data and os.path.exists(cache_path):
            print(
                f"Loading {partition_name} neighborhood features from cache: {cache_path}"
            )
            return pd.read_parquet(cache_path)

        print(f"Computing {partition_name} neighborhood features from scratch...")

        # Prepare storage
        feature_rows = []

        # Helper column to map back to global matrix indices
        # We assume df index is continuous 0..N-1 matching the matrices
        # But to be safe, we create an explicit index map
        df = df.copy()
        df["global_idx"] = np.arange(len(df))

        # Group by notebook
        grouped = df.groupby("notebook_id")

        # Iterate over notebooks
        for nb_id, group in grouped:
            # Separate Code and Markdown
            code_mask = group["cell_type"] != "markdown"
            md_mask = group["cell_type"] == "markdown"

            # If no markdown cells, nothing to predict
            if not md_mask.any():
                continue

            md_indices = group.loc[md_mask, "global_idx"].values
            md_cell_ids = group.loc[md_mask, "cell_id"].values

            # If no code cells, we cannot compute neighbors
            # Assign defaults
            if not code_mask.any():
                for i, cell_id in enumerate(md_cell_ids):
                    f = {"cell_id": cell_id, "notebook_id": nb_id}
                    for prefix in ["lex", "lat"]:
                        for r_idx in self.ranks_to_keep:
                            f[f"{prefix}_sim_{r_idx}"] = 0.0
                            f[f"{prefix}_rank_{r_idx}"] = 0.5
                        f[f"{prefix}_mean_rank"] = 0.5
                        f[f"{prefix}_std_rank"] = 0.28
                    feature_rows.append(f)
                continue

            code_indices = group.loc[code_mask, "global_idx"].values

            # Determine Code Ranks (Skeleton)
            # We infer ranks based on the order in the dataframe (assumed correct skeleton)
            # This works for both Train (ground truth) and Test (provided order)
            num_code = len(code_indices)
            if num_code > 1:
                local_ranks = np.arange(num_code) / (num_code - 1)
            else:
                local_ranks = np.zeros(num_code)

            # --- LEXICAL VIEW (TF-IDF) ---
            # Compute Sim: (N_md, Vocab) x (N_code, Vocab)^T -> (N_md, N_code)
            md_tfidf = tfidf_matrix[md_indices]
            code_tfidf = tfidf_matrix[code_indices]
            lex_sim_matrix = cosine_similarity(md_tfidf, code_tfidf)

            # --- LATENT VIEW (SVD) ---
            # Compute Sim: (N_md, Components) x (N_code, Components)^T
            md_svd = svd_matrix[md_indices]
            code_svd = svd_matrix[code_indices]
            lat_sim_matrix = cosine_similarity(md_svd, code_svd)

            # Extract Features for each MD cell
            for i in range(len(md_indices)):
                f = {"cell_id": md_cell_ids[i], "notebook_id": nb_id}

                # Lexical
                f.update(
                    self._compute_resolution_stats(
                        lex_sim_matrix[i], local_ranks, "lex"
                    )
                )

                # Latent
                f.update(
                    self._compute_resolution_stats(
                        lat_sim_matrix[i], local_ranks, "lat"
                    )
                )

                feature_rows.append(f)

        # Create DataFrame
        features_df = pd.DataFrame(feature_rows)

        # Save
        print(f"Saving {partition_name} neighborhood features to {cache_path}")
        features_df.to_parquet(cache_path, index=False)

        # Cleanup
        del df
        gc.collect()

        return features_df

    def construct_stage2_features(
        self,
        neighborhood_features: pd.DataFrame,
        ridge_predictions: pd.DataFrame,
        df_metadata: pd.DataFrame,
        partition_name: str,
        load_cached_data: bool = True,
    ) -> pd.DataFrame:
        """
        Combines neighborhood features, Ridge predictions, and metadata into the final feature matrix.

        Args:
            neighborhood_features: Output from extract_features.
            ridge_predictions: DataFrame with ['cell_id', 'ridge_pred'].
            df_metadata: Original dataframe with metadata (total_cells, etc.).
            partition_name: Name for caching.
            load_cached_data: Whether to load from cache.

        Returns:
            pd.DataFrame: Merged dataframe ready for LightGBM.
        """
        os.makedirs(CACHE_DIR, exist_ok=True)
        cache_path = os.path.join(
            CACHE_DIR, f"{partition_name}_stage2_features.parquet"
        )

        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading {partition_name} Stage 2 features from cache: {cache_path}")
            return pd.read_parquet(cache_path)

        print(f"Constructing {partition_name} Stage 2 features...")

        # 1. Prepare Metadata
        # We need to calculate Markdown Ratio for each notebook
        # df_metadata contains all cells.
        nb_stats = (
            df_metadata.groupby("notebook_id")
            .agg(
                total_cells=("cell_id", "count"),
                num_md=("cell_type", lambda x: (x == "markdown").sum()),
            )
            .reset_index()
        )
        nb_stats["md_ratio"] = nb_stats["num_md"] / nb_stats["total_cells"]

        # Filter df_metadata to only markdown cells (since we only predict for markdown)
        md_meta = df_metadata[df_metadata["cell_type"] == "markdown"].copy()

        # Merge notebook stats
        md_meta = md_meta.merge(
            nb_stats[["notebook_id", "md_ratio"]], on="notebook_id", how="left"
        )

        # 2. Merge Neighborhood Features
        # Inner join ensures we only keep cells we have features for
        merged = md_meta.merge(
            neighborhood_features, on=["cell_id", "notebook_id"], how="inner"
        )

        # 3. Merge Ridge Predictions
        # Left join to be safe, fillna with 0.5 (neutral rank) or similar
        merged = merged.merge(ridge_predictions, on="cell_id", how="left")

        # Handle missing ridge preds if any
        if "ridge_pred" in merged.columns:
            merged["ridge_pred"] = merged["ridge_pred"].fillna(0.5)

        # 4. Final Cleanup
        # Drop non-feature columns that are not needed for training (but keep IDs/Target)
        # We keep: cell_id, notebook_id, rank (target), ridge_pred, md_ratio, total_cells, and all lex/lat features
        # We drop: source, cell_type, ancestor_id, filepath, cell_order, parent_id
        cols_to_drop = [
            "source",
            "cell_type",
            "ancestor_id",
            "filepath",
            "cell_order",
            "parent_id",
        ]
        merged = merged.drop(columns=[c for c in cols_to_drop if c in merged.columns])

        print(f"Saving {partition_name} Stage 2 features to {cache_path}")
        merged.to_parquet(cache_path, index=False)

        return merged
