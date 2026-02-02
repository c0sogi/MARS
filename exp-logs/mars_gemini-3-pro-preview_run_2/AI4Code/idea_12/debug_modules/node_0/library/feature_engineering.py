import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from library.config import Config


class NeighborhoodFeatureExtractor:
    """
    Implements the feature engineering pipeline for Stage 2.
    Generates Multi-View Neighborhood features (Lexical & Latent) and integrates
    Stage 1 Ridge predictions.
    """

    def __init__(self):
        self.neighbor_k = Config.NEIGHBOR_K
        self.debug = Config.DEBUG

    def extract_features(
        self,
        df: pd.DataFrame,
        text_pipeline,
        ridge_preds: pd.DataFrame,
        partition: str = "train",
        load_cached_data: bool = True,
    ) -> pd.DataFrame:
        """
        Main method to generate features. Implements caching.

        Args:
            df: DataFrame containing cell metadata and source.
            text_pipeline: Fitted TextPipeline instance.
            ridge_preds: DataFrame containing ['cell_id', 'ridge_pred'].
            partition: 'train', 'val', or 'test'. Used for cache naming.
            load_cached_data: Whether to load from cache if available.

        Returns:
            DataFrame ready for LightGBM.
        """
        # Construct cache path
        cache_filename = f"{partition}_features.parquet"
        cache_path = os.path.join(Config.WORKING_DIR, cache_filename)

        # 1. Try Loading from Cache
        if load_cached_data and os.path.exists(cache_path):
            print(f"[{partition}] Loading features from cache: {cache_path}")
            df_features = pd.read_parquet(cache_path)
            # If in debug mode, we might need to subset the cached data if it's full size
            if self.debug:
                # Filter to match the input df IDs
                valid_ids = set(df["id"].unique())
                df_features = df_features[
                    df_features["id"].isin(valid_ids)
                ].reset_index(drop=True)
            return df_features

        # 2. Compute Features from Scratch
        print(f"[{partition}] Generating features from scratch...")

        # Filter for markdown cells to initialize the result dataframe
        # We only predict ranks for markdown cells
        df_md = df[df["cell_type"] == "markdown"].copy()

        # If no markdown cells (edge case), return empty
        if len(df_md) == 0:
            return pd.DataFrame()

        # Generate Neighborhood Features
        neighborhood_features = self._compute_neighborhood_features(df, text_pipeline)

        # Merge Neighborhood Features
        # neighborhood_features contains ['cell_id', ... feats ...]
        df_features = df_md.merge(neighborhood_features, on="cell_id", how="left")

        # Merge Stage 1 Ridge Predictions
        if ridge_preds is not None and not ridge_preds.empty:
            df_features = df_features.merge(ridge_preds, on="cell_id", how="left")
            # Fill missing ridge preds (if any) with 0.5 or mean
            df_features["ridge_pred"] = df_features["ridge_pred"].fillna(0.5)
        else:
            df_features["ridge_pred"] = 0.5

        # Add Metadata Features
        # Calculate code ratio per notebook
        nb_stats = (
            df.groupby("id")
            .apply(
                lambda x: pd.Series(
                    {
                        "n_code": (x["cell_type"] == "code").sum(),
                        "n_md": (x["cell_type"] == "markdown").sum(),
                    }
                )
            )
            .reset_index()
        )
        nb_stats["code_ratio"] = nb_stats["n_code"] / (
            nb_stats["n_code"] + nb_stats["n_md"] + 1e-5
        )

        df_features = df_features.merge(nb_stats, on="id", how="left")

        # Select/Order columns
        # We keep 'pct_rank' as target if it exists (train/val), else it might be -1.0
        cols_to_keep = [
            "id",
            "cell_id",
            "pct_rank",
            "ridge_pred",
            "lex_mean",
            "lex_wmean",
            "lex_std",
            "lex_min",
            "lex_max",
            "lat_mean",
            "lat_wmean",
            "lat_std",
            "lat_min",
            "lat_max",
            "n_code",
            "n_md",
            "code_ratio",
        ]

        # Filter columns that exist
        cols_to_keep = [c for c in cols_to_keep if c in df_features.columns]
        df_features = df_features[cols_to_keep]

        # 3. Save to Cache
        if not self.debug:
            print(f"[{partition}] Saving features to cache: {cache_path}")
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            df_features.to_parquet(cache_path, index=False)

        return df_features

    def _compute_neighborhood_features(
        self, df: pd.DataFrame, text_pipeline
    ) -> pd.DataFrame:
        """
        Computes Lexical and Latent neighborhood statistics.
        """
        # 1. Global Vectorization
        # Ensure input list is aligned with DataFrame index
        print("Transforming text data...")
        sources = df["source"].astype(str).tolist()
        X_tfidf, X_svd = text_pipeline.transform(sources)

        # Create helper arrays for fast indexing
        # Map cell_id to matrix row index
        cell_id_to_idx = {cid: i for i, cid in enumerate(df["cell_id"])}

        # Group by notebook
        # We iterate over notebooks to perform local similarity search
        grouped = df.groupby("id")

        results = []

        # Iterate with progress bar
        print("Computing neighborhood features per notebook...")
        for nb_id, group in tqdm(grouped, total=len(grouped), mininterval=5.0):

            # Identify Code and Markdown cells in this notebook
            code_mask = group["cell_type"] == "code"
            md_mask = group["cell_type"] == "markdown"

            code_ids = group.loc[code_mask, "cell_id"].values
            md_ids = group.loc[md_mask, "cell_id"].values

            if len(md_ids) == 0:
                continue

            # If no code cells, we cannot compute neighbors. Return defaults.
            if len(code_ids) == 0:
                for mid in md_ids:
                    results.append(
                        {
                            "cell_id": mid,
                            "lex_mean": 0.5,
                            "lex_wmean": 0.5,
                            "lex_std": 0.28,
                            "lex_min": 0.0,
                            "lex_max": 1.0,
                            "lat_mean": 0.5,
                            "lat_wmean": 0.5,
                            "lat_std": 0.28,
                            "lat_min": 0.0,
                            "lat_max": 1.0,
                        }
                    )
                continue

            # Assign Anchor Ranks to Code Cells
            # In Train: we trust 'rank'. In Test: we trust 'code_rank' (relative order).
            # To be consistent, we always re-derive equidistant ranks based on the code sequence.
            if "code_rank" in group.columns and group["code_rank"].max() > -1:
                # Test set or pre-calculated relative rank
                code_ranks_sorted = group.loc[code_mask].sort_values("code_rank")
            else:
                # Train set fallback: use absolute rank
                code_ranks_sorted = group.loc[code_mask].sort_values("rank")

            n_code = len(code_ids)
            # Equidistant ranks: 0.0 to 1.0
            if n_code == 1:
                anchor_ranks = np.array([0.0])
            else:
                anchor_ranks = np.linspace(0.0, 1.0, n_code)

            # Map code_id -> anchor_rank
            # The sorted dataframe aligns with anchor_ranks array
            sorted_code_ids = code_ranks_sorted["cell_id"].values

            # Get Matrix Indices
            md_indices = [cell_id_to_idx[cid] for cid in md_ids]
            code_indices = [cell_id_to_idx[cid] for cid in sorted_code_ids]

            # --- Lexical View (TF-IDF) ---
            # Slice matrices: (n_md, n_features) x (n_code, n_features).T
            # Result: (n_md, n_code) similarity matrix
            md_tfidf_sub = X_tfidf[md_indices]
            code_tfidf_sub = X_tfidf[code_indices]

            # Compute Cosine Similarity (Sparse)
            # Note: TfidfVectorizer output is normalized, so dot product is cosine similarity
            sim_lex = md_tfidf_sub.dot(code_tfidf_sub.T)
            # Convert to dense for numpy ops (usually small enough per notebook)
            if isinstance(sim_lex, csr_matrix):
                sim_lex = sim_lex.toarray()
            else:
                sim_lex = np.array(sim_lex)

            # --- Latent View (SVD) ---
            md_svd_sub = X_svd[md_indices]
            code_svd_sub = X_svd[code_indices]

            # Normalize SVD vectors for Cosine Similarity
            # (TruncatedSVD output is not necessarily normalized)
            md_norm = np.linalg.norm(md_svd_sub, axis=1, keepdims=True)
            code_norm = np.linalg.norm(code_svd_sub, axis=1, keepdims=True)

            # Avoid divide by zero
            md_norm[md_norm == 0] = 1e-9
            code_norm[code_norm == 0] = 1e-9

            md_svd_sub = md_svd_sub / md_norm
            code_svd_sub = code_svd_sub / code_norm

            sim_lat = np.dot(md_svd_sub, code_svd_sub.T)

            # --- Feature Aggregation ---
            # Process each markdown cell
            for i, mid in enumerate(md_ids):
                row_feats = {"cell_id": mid}

                # Helper to calc stats
                def calc_stats(sims, ranks, prefix):
                    # Sort by similarity descending
                    top_k_idx = np.argsort(sims)[::-1][: self.neighbor_k]
                    top_k_sims = sims[top_k_idx]
                    top_k_ranks = ranks[top_k_idx]

                    # Basic Stats
                    row_feats[f"{prefix}_mean"] = np.mean(top_k_ranks)
                    row_feats[f"{prefix}_std"] = np.std(top_k_ranks)
                    row_feats[f"{prefix}_min"] = np.min(top_k_ranks)
                    row_feats[f"{prefix}_max"] = np.max(top_k_ranks)

                    # Weighted Mean
                    sum_sim = np.sum(top_k_sims)
                    if sum_sim > 1e-6:
                        w_mean = np.sum(top_k_ranks * top_k_sims) / sum_sim
                    else:
                        w_mean = np.mean(top_k_ranks)
                    row_feats[f"{prefix}_wmean"] = w_mean

                # Lexical Stats
                calc_stats(sim_lex[i], anchor_ranks, "lex")

                # Latent Stats
                calc_stats(sim_lat[i], anchor_ranks, "lat")

                results.append(row_feats)

        return pd.DataFrame(results)
