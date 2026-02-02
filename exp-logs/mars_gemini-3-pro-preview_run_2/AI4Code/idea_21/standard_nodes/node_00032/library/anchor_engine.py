import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from library.config import Config


class AnchorExtractor:
    """
    Implements the Multi-Resolution Neighborhood Anchoring engine.
    Extracts features based on the similarity between Markdown cells (queries)
    and Code cells (anchors) within the same notebook using both Lexical (Sparse)
    and Latent (Dense) vector representations.
    """

    def __init__(self):
        self.config = Config
        self.cache_dir = self.config.WORKING_DIR
        self.top_k = self.config.ANCHOR_PARAMS["top_k"]

        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_cache_path(self, split_name):
        return os.path.join(self.cache_dir, f"{split_name}_anchor_features.parquet")

    def extract_features(
        self, df_cells, tfidf_matrix, svd_matrix, split_name, load_cached_data=True
    ):
        """
        Extracts anchor features for the given dataset.

        Args:
            df_cells (pd.DataFrame): DataFrame containing cell metadata. Must have 'id', 'cell_id', 'cell_type'.
                                     The index of this DataFrame must align with the rows of the matrices.
            tfidf_matrix (scipy.sparse.csr_matrix): Global TF-IDF matrix.
            svd_matrix (np.ndarray): Global SVD matrix.
            split_name (str): 'train', 'val', or 'test' for caching purposes.
            load_cached_data (bool): Whether to attempt loading from cache.

        Returns:
            pd.DataFrame: DataFrame containing feature columns for Markdown cells.
                          Columns: [cell_id, lex_nn1_rank, lex_nn1_sim, ..., lat_topk_mean, ...]
        """
        cache_path = self._get_cache_path(split_name)

        # 1. Try to load from cache
        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading anchor features for {split_name} from cache: {cache_path}")
            try:
                return pd.read_parquet(cache_path)
            except Exception as e:
                print(f"Failed to load cache: {e}. Reprocessing...")

        print(f"Extracting anchor features for {split_name}...")

        # 2. Preparation
        # Ensure df_cells has a proper index to map back to matrix rows
        # We assume df_cells is unchanged from when vectors were created.
        # We add a 'matrix_idx' column to track the global row index.
        df_cells = df_cells.copy()
        df_cells["matrix_idx"] = np.arange(len(df_cells))

        # Group by notebook ID
        # We rely on the fact that Code cells are in the correct order in the input DataFrame
        # (as guaranteed by DataManager for train, and inherent in test JSON structure for code).
        grouped = df_cells.groupby("id", sort=False)

        results = []

        # 3. Iteration over notebooks
        # Note: Using a loop here. For 140k notebooks, this takes time but is necessary
        # for per-notebook similarity context.

        for nb_id, group in grouped:
            # Separate Code and Markdown
            code_mask = (
                group["cell_type"] != "markdown"
            )  # Treat 'code' and 'raw' as anchors if any
            md_mask = group["cell_type"] == "markdown"

            df_code = group[code_mask]
            df_md = group[md_mask]

            # If no markdown cells, nothing to predict for this notebook
            if len(df_md) == 0:
                continue

            # If no code cells, we cannot calculate anchors.
            # Fill with defaults (0.5 rank, 0.0 sim).
            if len(df_code) == 0:
                for _, row in df_md.iterrows():
                    res = {"cell_id": row["cell_id"], "id": nb_id}
                    self._add_default_features(res)
                    results.append(res)
                continue

            # Assign normalized ranks to code cells (0.0 to 1.0)
            n_code = len(df_code)
            if n_code == 1:
                code_ranks = np.array([0.0])
            else:
                code_ranks = np.linspace(0, 1, n_code)

            # Get global indices
            code_indices = df_code["matrix_idx"].values
            md_indices = df_md["matrix_idx"].values

            # --- Lexical (Sparse) Resolution ---
            # Slice matrices
            # cosine_similarity accepts sparse matrices efficiently
            code_vecs_sparse = tfidf_matrix[code_indices]
            md_vecs_sparse = tfidf_matrix[md_indices]

            # Compute Similarity (Markdown x Code)
            # Result shape: (n_md, n_code)
            sim_lex = cosine_similarity(md_vecs_sparse, code_vecs_sparse)

            # --- Latent (Dense) Resolution ---
            code_vecs_dense = svd_matrix[code_indices]
            md_vecs_dense = svd_matrix[md_indices]

            sim_lat = cosine_similarity(md_vecs_dense, code_vecs_dense)

            # --- Feature Extraction ---
            # Iterate over each markdown cell in this notebook
            for i, (idx, row) in enumerate(df_md.iterrows()):
                res = {"cell_id": row["cell_id"], "id": nb_id}

                # Process Lexical
                self._extract_resolution_features(
                    res, sim_lex[i], code_ranks, prefix="lex", include_instances=True
                )

                # Process Latent
                self._extract_resolution_features(
                    res, sim_lat[i], code_ranks, prefix="lat", include_instances=False
                )

                results.append(res)

        # 4. Finalize
        df_features = pd.DataFrame(results)

        # Optimize types
        float_cols = [c for c in df_features.columns if c not in ["cell_id", "id"]]
        df_features[float_cols] = df_features[float_cols].astype(np.float32)

        # 5. Save to Cache
        print(f"Saving anchor features to {cache_path}")
        df_features.to_parquet(cache_path, index=False)

        return df_features

    def _extract_resolution_features(
        self, res_dict, similarities, ranks, prefix, include_instances
    ):
        """
        Helper to extract features from a similarity array for a single markdown cell.

        Args:
            res_dict (dict): Dictionary to update.
            similarities (np.array): Array of similarity scores against all code cells.
            ranks (np.array): Array of normalized ranks for the code cells.
            prefix (str): Prefix for feature names (e.g., 'lex' or 'lat').
            include_instances (bool): Whether to extract specific NN ranks/scores.
        """
        # Sort neighbors by similarity descending
        # argsort gives ascending, so we reverse
        sorted_indices = np.argsort(similarities)[::-1]

        # --- Instance Features (Top 1, 2, 3) ---
        if include_instances:
            for k in [1, 2, 3]:
                if len(sorted_indices) >= k:
                    idx = sorted_indices[k - 1]
                    res_dict[f"{prefix}_nn{k}_rank"] = ranks[idx]
                    res_dict[f"{prefix}_nn{k}_sim"] = similarities[idx]
                else:
                    # Fallback if fewer code cells than k
                    res_dict[f"{prefix}_nn{k}_rank"] = 0.5
                    res_dict[f"{prefix}_nn{k}_sim"] = 0.0

        # --- Smoothed Features (Top K) ---
        # Take top K indices
        top_k_indices = sorted_indices[: self.top_k]

        if len(top_k_indices) > 0:
            top_k_ranks = ranks[top_k_indices]
            res_dict[f"{prefix}_topk_mean"] = np.mean(top_k_ranks)
            res_dict[f"{prefix}_topk_std"] = np.std(top_k_ranks)
        else:
            res_dict[f"{prefix}_topk_mean"] = 0.5
            res_dict[f"{prefix}_topk_std"] = 0.0

    def _add_default_features(self, res_dict):
        """
        Adds default values for all features if no code cells exist.
        """
        # Lexical Instances
        for k in [1, 2, 3]:
            res_dict[f"lex_nn{k}_rank"] = 0.5
            res_dict[f"lex_nn{k}_sim"] = 0.0

        # Lexical Smoothed
        res_dict["lex_topk_mean"] = 0.5
        res_dict["lex_topk_std"] = 0.0

        # Latent Smoothed
        res_dict["lat_topk_mean"] = 0.5
        res_dict["lat_topk_std"] = 0.0
