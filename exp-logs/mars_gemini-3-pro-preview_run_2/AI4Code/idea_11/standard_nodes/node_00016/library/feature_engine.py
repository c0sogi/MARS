import os
import numpy as np
import pandas as pd
import scipy.sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from library import config, utils, data_factory


# ------------------------------------------------------------------------------
# 1. Text Vectorizer (TF-IDF)
# ------------------------------------------------------------------------------
class TextVectorizer:
    def __init__(self, params=config.TFIDF_PARAMS):
        self.params = params
        self.vectorizer = None

    def fit(self, texts):
        """Fits the TF-IDF vectorizer on a list of texts."""
        utils.log_message("Fitting TF-IDF Vectorizer...")
        self.vectorizer = TfidfVectorizer(**self.params)
        self.vectorizer.fit(texts)
        return self

    def transform(self, texts):
        """Transforms texts into a sparse TF-IDF matrix."""
        if self.vectorizer is None:
            raise ValueError("Vectorizer has not been fitted.")
        return self.vectorizer.transform(texts)

    def save(self, filepath):
        """Saves the fitted vectorizer."""
        utils.log_message(f"Saving TF-IDF Vectorizer to {filepath}...")
        joblib.dump(self.vectorizer, filepath)

    def load(self, filepath):
        """Loads a fitted vectorizer."""
        utils.log_message(f"Loading TF-IDF Vectorizer from {filepath}...")
        self.vectorizer = joblib.load(filepath)
        return self


# ------------------------------------------------------------------------------
# 2. Semantic Projector (SVD)
# ------------------------------------------------------------------------------
class SemanticProjector:
    def __init__(self, params=config.SVD_PARAMS):
        self.params = params
        self.svd = None

    def fit(self, tfidf_matrix):
        """Fits TruncatedSVD on a TF-IDF matrix."""
        utils.log_message("Fitting SVD Model...")
        self.svd = TruncatedSVD(**self.params)
        self.svd.fit(tfidf_matrix)
        return self

    def transform(self, tfidf_matrix):
        """Projects TF-IDF matrix into latent space."""
        if self.svd is None:
            raise ValueError("SVD model has not been fitted.")
        # SVD transform returns a dense numpy array
        return self.svd.transform(tfidf_matrix)

    def save(self, filepath):
        """Saves the fitted SVD model."""
        utils.log_message(f"Saving SVD Model to {filepath}...")
        joblib.dump(self.svd, filepath)

    def load(self, filepath):
        """Loads a fitted SVD model."""
        utils.log_message(f"Loading SVD Model from {filepath}...")
        self.svd = joblib.load(filepath)
        return self


# ------------------------------------------------------------------------------
# 3. Neighborhood Feature Extractor
# ------------------------------------------------------------------------------
class NeighborhoodExtractor:
    def __init__(self, top_k=config.NEIGHBOR_K):
        self.top_k = top_k

    def extract(self, df, tfidf_matrix, svd_matrix):
        """
        Extracts neighborhood features for markdown cells based on code cell proximity.

        Args:
            df (pd.DataFrame): Dataframe containing cell metadata (id, cell_type, etc.)
            tfidf_matrix (scipy.sparse.csr_matrix): Aligned TF-IDF features.
            svd_matrix (np.ndarray): Aligned SVD features.

        Returns:
            pd.DataFrame: Feature dataframe aligned with the input df.
        """
        utils.log_message("Extracting Neighborhood Features...")

        # Ensure alignment
        assert len(df) == tfidf_matrix.shape[0]
        assert len(df) == svd_matrix.shape[0]

        # Pre-allocate feature arrays
        n_samples = len(df)

        # Lexical Features
        lex_mean_rank = np.zeros(n_samples, dtype=np.float32)
        lex_weighted_rank = np.zeros(n_samples, dtype=np.float32)
        lex_max_sim = np.zeros(n_samples, dtype=np.float32)

        # Latent Features
        lat_mean_rank = np.zeros(n_samples, dtype=np.float32)
        lat_weighted_rank = np.zeros(n_samples, dtype=np.float32)
        lat_max_sim = np.zeros(n_samples, dtype=np.float32)

        # Metadata Features
        n_code_cells = np.zeros(n_samples, dtype=np.int32)
        md_ratio = np.zeros(n_samples, dtype=np.float32)

        # Group by notebook to process interactions
        # We use a groupby on the index to handle matrix slicing efficiently
        # Assuming df is sorted or grouped by 'id' is beneficial but we handle general case
        # To optimize, we get groups of indices

        # Convert 'id' to categorical codes for faster grouping if not already
        if not isinstance(df["id"].dtype, pd.CategoricalDtype):
            df["id"] = df["id"].astype("category")

        # Group indices by notebook ID
        # This gives us the row indices in the global matrix for each notebook
        notebook_groups = df.groupby("id", observed=True).indices

        # Iterate over each notebook
        # Note: This loop can be slow in pure Python.
        # For 140k notebooks, we need to be efficient.

        for nb_id, indices in notebook_groups.items():
            if len(indices) == 0:
                continue

            # Get subset of dataframe and matrices
            # indices is a numpy array of row indices
            nb_df_subset = df.iloc[indices]

            # Identify Code and Markdown indices RELATIVE to the subset
            # But we need global indices to slice the matrices
            is_code = (nb_df_subset["cell_type"] == "code").values
            is_md = ~is_code

            # Global indices
            code_indices = indices[is_code]
            md_indices = indices[is_md]

            num_code = len(code_indices)
            num_md = len(md_indices)
            total_cells = num_code + num_md

            # Set metadata features
            n_code_cells[indices] = num_code
            md_ratio[indices] = num_md / total_cells if total_cells > 0 else 0

            if num_code == 0 or num_md == 0:
                # If no code cells, ranks are 0 (or default).
                # If no md cells, nothing to predict (but we fill for consistency).
                continue

            # Calculate Code Ranks (Normalized 0.0 to 1.0)
            # We use the integer position of the code cell / num_code
            # This aligns with the target definition: rank ~ pos / num_code
            code_ranks = np.arange(num_code) / num_code

            # --- Lexical View (TF-IDF) ---
            # Slice matrices
            # sparse matrix slicing: [rows, :]
            md_tfidf = tfidf_matrix[md_indices]
            code_tfidf = tfidf_matrix[code_indices]

            # Compute Cosine Similarity: (N_md x F) dot (N_code x F).T -> (N_md x N_code)
            # Both are sparse, result is dense usually small enough for one notebook
            sim_lex = cosine_similarity(md_tfidf, code_tfidf)

            # Process each markdown cell in the notebook
            for i in range(num_md):
                sims = sim_lex[i]

                # Find Top-K neighbors
                # If num_code < K, take all
                k = min(self.top_k, num_code)

                # argpartition is faster than sort for finding top k
                if k < num_code:
                    top_k_idx = np.argpartition(sims, -k)[-k:]
                else:
                    top_k_idx = np.arange(num_code)

                top_sims = sims[top_k_idx]
                top_ranks = code_ranks[top_k_idx]

                # Features
                global_md_idx = md_indices[i]

                lex_max_sim[global_md_idx] = np.max(sims)
                lex_mean_rank[global_md_idx] = np.mean(top_ranks)

                sum_sim = np.sum(top_sims)
                if sum_sim > 1e-6:
                    lex_weighted_rank[global_md_idx] = (
                        np.sum(top_ranks * top_sims) / sum_sim
                    )
                else:
                    lex_weighted_rank[global_md_idx] = np.mean(top_ranks)  # Fallback

            # --- Latent View (SVD) ---
            md_svd = svd_matrix[md_indices]
            code_svd = svd_matrix[code_indices]

            sim_lat = cosine_similarity(md_svd, code_svd)

            for i in range(num_md):
                sims = sim_lat[i]
                k = min(self.top_k, num_code)

                if k < num_code:
                    top_k_idx = np.argpartition(sims, -k)[-k:]
                else:
                    top_k_idx = np.arange(num_code)

                top_sims = sims[top_k_idx]
                top_ranks = code_ranks[top_k_idx]

                global_md_idx = md_indices[i]

                lat_max_sim[global_md_idx] = np.max(sims)
                lat_mean_rank[global_md_idx] = np.mean(top_ranks)

                sum_sim = np.sum(top_sims)
                if sum_sim > 1e-6:
                    lat_weighted_rank[global_md_idx] = (
                        np.sum(top_ranks * top_sims) / sum_sim
                    )
                else:
                    lat_weighted_rank[global_md_idx] = np.mean(top_ranks)

        # Assemble DataFrame
        feature_df = pd.DataFrame(
            {
                "lex_mean_rank": lex_mean_rank,
                "lex_weighted_rank": lex_weighted_rank,
                "lex_max_sim": lex_max_sim,
                "lat_mean_rank": lat_mean_rank,
                "lat_weighted_rank": lat_weighted_rank,
                "lat_max_sim": lat_max_sim,
                "n_code_cells": n_code_cells,
                "md_ratio": md_ratio,
            },
            index=df.index,
        )

        # Add SVD components for the markdown cells (context)
        # We only care about SVD features for the row itself (the markdown cell)
        svd_cols = [f"svd_{i}" for i in range(svd_matrix.shape[1])]
        svd_df = pd.DataFrame(svd_matrix, columns=svd_cols, index=df.index)

        # Concatenate
        result_df = pd.concat([df, feature_df, svd_df], axis=1)

        # Filter to return only markdown cells if desired?
        # The prompt implies we need features for the regression task.
        # Usually we filter for is_code=0 later. We return everything here aligned.

        return result_df


# ------------------------------------------------------------------------------
# 4. Main Pipeline Function
# ------------------------------------------------------------------------------
def generate_features(mode, load_cached_data=True):
    """
    Orchestrates the feature generation pipeline.

    Args:
        mode (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to use cached intermediate files.

    Returns:
        pd.DataFrame: The processed dataframe with features.
    """
    # Determine paths based on mode
    if mode == "train":
        cache_path = config.CACHE_TRAIN_FEATURES
        load_func = data_factory.load_train_data
    elif mode == "val":
        cache_path = config.CACHE_VAL_FEATURES
        load_func = data_factory.load_val_data
    elif mode == "test":
        cache_path = config.CACHE_TEST_FEATURES
        load_func = data_factory.load_test_data
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # 1. Check Cache
    if load_cached_data and os.path.exists(cache_path):
        utils.log_message(f"Loading cached features for {mode} from {cache_path}...")
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception as e:
            utils.log_message(f"Failed to load feature cache: {e}. Recomputing...")

    # 2. Load Raw Data
    df = load_func(load_cached_data=load_cached_data)

    # 3. Vectorization (TF-IDF)
    # We always fit on TRAIN data, transform others.
    # If mode is train, we fit. If val/test, we load.

    vectorizer = TextVectorizer()

    if mode == "train":
        # Filter for markdown cells to fit vocabulary (optional, but cleaner)
        # However, we might want code tokens in vocab too.
        # Config says 'content', usually implies all text.
        # Let's fit on all markdown text in train to capture the language.
        train_md_text = df[df["cell_type"] == "markdown"]["source"].astype(str).tolist()

        if load_cached_data and os.path.exists(config.CACHE_TFIDF_VECTORIZER):
            vectorizer.load(config.CACHE_TFIDF_VECTORIZER)
        else:
            vectorizer.fit(train_md_text)
            vectorizer.save(config.CACHE_TFIDF_VECTORIZER)
    else:
        # Load existing vectorizer
        if not os.path.exists(config.CACHE_TFIDF_VECTORIZER):
            raise FileNotFoundError(
                "TF-IDF vectorizer not found. Run 'train' mode first."
            )
        vectorizer.load(config.CACHE_TFIDF_VECTORIZER)

    # Transform current dataset
    utils.log_message(f"Vectorizing {mode} data...")
    all_text = df["source"].astype(str).tolist()
    tfidf_matrix = vectorizer.transform(all_text)

    # 4. Semantic Projection (SVD)
    projector = SemanticProjector()

    if mode == "train":
        # Fit SVD on the TF-IDF matrix (usually just MD part or all?
        # Standard LSA fits on the corpus. We fit on the full train corpus or just MD.
        # Let's fit on the train TF-IDF matrix we just generated.
        if load_cached_data and os.path.exists(config.CACHE_SVD_MODEL):
            projector.load(config.CACHE_SVD_MODEL)
        else:
            projector.fit(tfidf_matrix)
            projector.save(config.CACHE_SVD_MODEL)
    else:
        if not os.path.exists(config.CACHE_SVD_MODEL):
            raise FileNotFoundError("SVD model not found. Run 'train' mode first.")
        projector.load(config.CACHE_SVD_MODEL)

    # Project current dataset
    utils.log_message(f"Projecting {mode} data to SVD space...")
    svd_matrix = projector.transform(tfidf_matrix)

    # 5. Neighborhood Extraction
    extractor = NeighborhoodExtractor(top_k=config.NEIGHBOR_K)
    df_features = extractor.extract(df, tfidf_matrix, svd_matrix)

    # 6. Save to Cache
    utils.log_message(f"Saving features to {cache_path}...")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df_features.to_parquet(cache_path, index=False)

    return df_features
