import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from library.config import Config
from library.vectorizer import TextVectorizer, get_svd_features


class AnchorFeatureGenerator:
    """
    Generates Multi-Resolution Neighborhood features for the stacking model.
    Extracts instance-based and smoothed signals from both Lexical (TF-IDF)
    and Latent (SVD) views.
    """

    def __init__(self, vectorizer: TextVectorizer):
        """
        Args:
            vectorizer: A fitted TextVectorizer instance.
        """
        self.vectorizer = vectorizer

    def extract_features(
        self, df: pd.DataFrame, split_name: str, load_cached_data: bool = True
    ) -> pd.DataFrame:
        """
        Computes anchor features for the given DataFrame.

        Args:
            df: DataFrame containing cells (must include 'id', 'cell_type', 'source', 'cell_id').
            split_name: Name of the split ('train', 'val', 'test') for caching.
            load_cached_data: Whether to load features from disk if available.

        Returns:
            pd.DataFrame: DataFrame with anchor features and metadata.
        """
        # 1. Setup Cache Path
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        cache_path = os.path.join(
            Config.WORKING_DIR, f"{split_name}_anchor_features.parquet"
        )

        # 2. Try Loading Cache
        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading cached anchor features from {cache_path}")
            return pd.read_parquet(cache_path)

        print(f"Generating anchor features for {split_name}...")

        # 3. Prepare Data and Matrices
        # Reset index to ensure alignment between DataFrame rows and Matrix rows
        df = df.reset_index(drop=True)

        # Generate/Get Global Matrices
        # Lexical: TF-IDF (Sparse)
        print("Transforming text to TF-IDF matrix...")
        # Fill NaNs to avoid errors
        texts = df["source"].fillna("").astype(str)
        tfidf_matrix = self.vectorizer.transform(texts)

        # Latent: SVD (Dense)
        print("Retrieving SVD features...")
        svd_matrix = get_svd_features(
            df, self.vectorizer, split_name, load_cached_data=load_cached_data
        )

        # 4. Iterate Over Notebooks
        # Group by notebook ID to process each context individually
        # Using .indices is faster than iterating groupby objects
        groups = df.groupby("id").indices

        features_list = []

        # Hyperparameters
        instance_neighbors = Config.ANCHOR_INSTANCE_NEIGHBORS
        smoothing_k = Config.ANCHOR_SMOOTHING_K

        # Pre-fetch columns to avoid DataFrame overhead inside loop
        cell_types = df["cell_type"].values
        cell_ids = df["cell_id"].values

        # Iterate through each notebook
        for nb_id, indices in groups.items():
            # indices: numpy array of row indices for this notebook

            # Identify Code and Markdown cells
            # Using numpy boolean indexing
            nb_types = cell_types[indices]
            is_code = nb_types == "code"
            is_md = nb_types == "markdown"

            code_indices = indices[is_code]
            md_indices = indices[is_md]

            num_code = len(code_indices)
            num_md = len(md_indices)
            total_cells = len(indices)
            md_ratio = num_md / total_cells if total_cells > 0 else 0.0

            # If no markdown cells, nothing to predict for this notebook
            if num_md == 0:
                continue

            # If no code cells, we cannot anchor. Fill with defaults.
            if num_code == 0:
                for i in range(num_md):
                    global_idx = md_indices[i]
                    feat = {
                        "id": nb_id,
                        "cell_id": cell_ids[global_idx],
                        "total_cells": total_cells,
                        "md_ratio": md_ratio,
                    }
                    # Defaults
                    for k in instance_neighbors:
                        feat[f"lex_sim_{k}"] = 0.0
                        feat[f"lex_rank_{k}"] = 0.5
                    feat[f"lex_mean_{smoothing_k}"] = 0.5
                    feat[f"lex_std_{smoothing_k}"] = 0.0
                    feat[f"lat_mean_{smoothing_k}"] = 0.5
                    feat[f"lat_std_{smoothing_k}"] = 0.0
                    features_list.append(feat)
                continue

            # Establish Code Skeleton Ranks (Normalized 0.0 to 1.0)
            # Code cells are assumed to be in correct order
            code_ranks = np.linspace(0, 1, num_code)

            # Extract Sub-Matrices for this notebook
            # Lexical
            md_tfidf_sub = tfidf_matrix[md_indices]
            code_tfidf_sub = tfidf_matrix[code_indices]

            # Latent
            md_svd_sub = svd_matrix[md_indices]
            code_svd_sub = svd_matrix[code_indices]

            # Compute Pairwise Similarities (Cosine)
            # Result shape: (num_md, num_code)
            # Note: cosine_similarity handles dense/sparse inputs
            lex_sims = cosine_similarity(md_tfidf_sub, code_tfidf_sub)
            lat_sims = cosine_similarity(md_svd_sub, code_svd_sub)

            # Process each Markdown cell
            for i in range(num_md):
                feat = {
                    "id": nb_id,
                    "cell_id": cell_ids[md_indices[i]],
                    "total_cells": total_cells,
                    "md_ratio": md_ratio,
                }

                # --- Lexical Features ---
                # Sort neighbors by similarity (descending)
                # argsort gives ascending, so we reverse
                lex_sorted_idx = np.argsort(lex_sims[i])[::-1]

                # Instance Anchors (Specific Neighbors)
                for k in instance_neighbors:
                    # k is 1-based (1st, 2nd...)
                    if k <= num_code:
                        neighbor_idx = lex_sorted_idx[k - 1]
                        feat[f"lex_sim_{k}"] = float(lex_sims[i][neighbor_idx])
                        feat[f"lex_rank_{k}"] = float(code_ranks[neighbor_idx])
                    else:
                        # Fallback
                        feat[f"lex_sim_{k}"] = 0.0
                        feat[f"lex_rank_{k}"] = 0.5

                # Smoothed Anchors (Top K Aggregate)
                top_k_lex_idx = lex_sorted_idx[:smoothing_k]
                top_k_lex_ranks = code_ranks[top_k_lex_idx]
                feat[f"lex_mean_{smoothing_k}"] = float(np.mean(top_k_lex_ranks))
                feat[f"lex_std_{smoothing_k}"] = (
                    float(np.std(top_k_lex_ranks)) if len(top_k_lex_ranks) > 1 else 0.0
                )

                # --- Latent Features ---
                # Sort neighbors by similarity (descending)
                lat_sorted_idx = np.argsort(lat_sims[i])[::-1]

                # Smoothed Anchors (Top K Aggregate)
                top_k_lat_idx = lat_sorted_idx[:smoothing_k]
                top_k_lat_ranks = code_ranks[top_k_lat_idx]
                feat[f"lat_mean_{smoothing_k}"] = float(np.mean(top_k_lat_ranks))
                feat[f"lat_std_{smoothing_k}"] = (
                    float(np.std(top_k_lat_ranks)) if len(top_k_lat_ranks) > 1 else 0.0
                )

                features_list.append(feat)

        # 5. Compile DataFrame
        print("Compiling feature DataFrame...")
        feature_df = pd.DataFrame(features_list)

        # 6. Merge Targets (if available)
        # Check if 'norm_rank' exists in the source df (labeled data)
        if "norm_rank" in df.columns:
            print("Merging target labels...")
            # Create a subset of targets to merge
            targets = df[["id", "cell_id", "norm_rank"]].drop_duplicates()
            feature_df = feature_df.merge(targets, on=["id", "cell_id"], how="left")

        # 7. Save to Cache
        print(f"Saving anchor features to {cache_path}")
        feature_df.to_parquet(cache_path, index=False)

        return feature_df
