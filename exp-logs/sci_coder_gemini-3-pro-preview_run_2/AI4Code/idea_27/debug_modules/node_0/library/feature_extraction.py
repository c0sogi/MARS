import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from library.config import Config
from library.utils import set_seed


class FeatureEngineer:
    def __init__(self, config=None):
        """
        Initializes the FeatureEngineer.

        Args:
            config: Configuration object. If None, uses default Config.
        """
        self.config = config if config else Config()
        self.working_dir = self.config.WORKING_DIR
        os.makedirs(self.working_dir, exist_ok=True)
        set_seed(42)

    def extract_features(
        self, df_corpus, tfidf_mat, svd_mat, mode="train", load_cached_data=True
    ):
        """
        Generates Stage 2 features: Intrinsic LSA + Decoupled Neighborhoods (Lexical & Latent).

        Args:
            df_corpus (pd.DataFrame): The corpus dataframe containing cell metadata and order.
                                      Must be aligned with the rows of tfidf_mat and svd_mat.
            tfidf_mat (scipy.sparse.csr_matrix): Global TF-IDF matrix for all cells.
            svd_mat (np.ndarray): Global SVD matrix for all cells.
            mode (str): 'train', 'val', or 'test' for caching naming.
            load_cached_data (bool): Whether to load from cache if available.

        Returns:
            pd.DataFrame: DataFrame containing features for markdown cells.
        """
        cache_path = os.path.join(self.working_dir, f"{mode}_features.parquet")

        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading cached {mode} features from {cache_path}...")
            return pd.read_parquet(cache_path)

        print(f"Generating {mode} features from scratch...")

        # Ensure df_corpus has a numeric index to map to matrix rows
        df_corpus = df_corpus.reset_index(drop=True)

        # Prepare list to collect feature dictionaries
        feature_rows = []

        # Group by notebook to process interactions within each notebook
        # We need to preserve original indices to slice the matrices
        nb_groups = df_corpus.groupby("id")

        # Number of SVD components to use as intrinsic features
        n_lsa_feats = min(16, self.config.SVD_COMPONENTS)
        top_k = self.config.TOP_K_NEIGHBORS

        for nb_id, group_indices in tqdm(
            nb_groups.groups.items(), desc=f"Extracting {mode} features"
        ):
            # group_indices are the row indices in df_corpus (and thus in the matrices)
            # Get the subset of the dataframe
            nb_df = df_corpus.loc[group_indices]

            # Identify Code and Markdown indices relative to the matrix
            code_mask = nb_df["cell_type"] == "code"
            md_mask = nb_df["cell_type"] == "markdown"

            code_indices = nb_df.index[code_mask].tolist()
            md_indices = nb_df.index[md_mask].tolist()

            n_code = len(code_indices)
            n_md = len(md_indices)

            # Skip if no markdown cells (nothing to predict)
            if n_md == 0:
                continue

            # If no code cells, we can't compute neighborhood features relative to code.
            # We will fill neighborhood features with defaults (0.5 or -1) later.
            has_code = n_code > 0

            # ---------------------------------------------------------
            # 1. Prepare Anchors (Code Cells)
            # ---------------------------------------------------------
            if has_code:
                # Code cells act as fixed anchors distributed 0..1
                code_ranks = np.linspace(0, 1, n_code)

                # Slice matrices for code cells
                code_tfidf = tfidf_mat[code_indices]
                code_svd = svd_mat[code_indices]

            # ---------------------------------------------------------
            # 2. Process Markdown Cells (Targets)
            # ---------------------------------------------------------
            # Slice matrices for markdown cells
            md_tfidf = tfidf_mat[md_indices]
            md_svd = svd_mat[md_indices]

            # Compute Similarities (Decoupled Neighborhoods)
            if has_code:
                # Lexical View (Sparse)
                sim_lex = cosine_similarity(md_tfidf, code_tfidf)
                # Latent View (Dense)
                sim_lat = cosine_similarity(md_svd, code_svd)

            # Iterate over each markdown cell in this notebook
            for i, global_idx in enumerate(md_indices):
                row_data = nb_df.loc[global_idx]

                feat = {
                    "id": nb_id,
                    "cell_id": row_data["cell_id"],
                }

                # Target (only for train/val)
                if "pct_rank" in row_data:
                    feat["target"] = row_data["pct_rank"]

                # --- Metadata Features ---
                feat["n_code"] = n_code
                feat["n_md"] = n_md
                feat["md_ratio"] = n_md / (n_code + n_md) if (n_code + n_md) > 0 else 0

                # --- Intrinsic Features (LSA) ---
                # Use the markdown cell's own SVD vector
                # md_svd is shape (n_md, n_components)
                current_svd_vec = md_svd[i]
                for c in range(n_lsa_feats):
                    feat[f"lsa_{c}"] = current_svd_vec[c]

                # --- Neighborhood Features ---
                if has_code:
                    # Helper to extract stats
                    def extract_neighbor_stats(sim_scores, prefix):
                        # Sort indices by similarity descending
                        sorted_args = np.argsort(sim_scores)[::-1]

                        # Take Top-K
                        top_args = sorted_args[:top_k]
                        top_sims = sim_scores[top_args]
                        top_ranks = code_ranks[top_args]

                        # Top-1
                        feat[f"{prefix}_top1_rank"] = top_ranks[0]
                        feat[f"{prefix}_top1_sim"] = top_sims[0]

                        # Aggregates
                        feat[f"{prefix}_mean_rank"] = np.mean(top_ranks)
                        feat[f"{prefix}_std_rank"] = (
                            np.std(top_ranks) if len(top_ranks) > 1 else 0.0
                        )

                        # Weighted Mean
                        w_sum = np.sum(top_sims) + 1e-9
                        feat[f"{prefix}_wmean_rank"] = (
                            np.sum(top_ranks * top_sims) / w_sum
                        )

                    # Lexical Features
                    extract_neighbor_stats(sim_lex[i], "lex")

                    # Latent Features
                    extract_neighbor_stats(sim_lat[i], "lat")
                else:
                    # Fallback if no code cells exist
                    for prefix in ["lex", "lat"]:
                        feat[f"{prefix}_top1_rank"] = 0.5
                        feat[f"{prefix}_top1_sim"] = 0.0
                        feat[f"{prefix}_mean_rank"] = 0.5
                        feat[f"{prefix}_std_rank"] = 0.0
                        feat[f"{prefix}_wmean_rank"] = 0.5

                feature_rows.append(feat)

        # Convert to DataFrame
        df_features = pd.DataFrame(feature_rows)

        # Save to cache
        print(f"Saving {mode} features to {cache_path}...")
        df_features.to_parquet(cache_path, index=False)

        return df_features
