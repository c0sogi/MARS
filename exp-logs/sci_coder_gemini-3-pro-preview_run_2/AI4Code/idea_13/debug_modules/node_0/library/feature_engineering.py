import os
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import euclidean_distances
from joblib import Parallel, delayed
from library.config import Config
from library.utils import seed_everything


class TextVectorizer:
    """
    Manages TF-IDF and SVD vectorization for text data.
    """

    def __init__(self, config=Config):
        self.config = config
        self.tfidf = None
        self.svd = None

    def fit(self, texts, load_cached_models=True):
        """
        Fits TF-IDF and SVD models on the provided texts.

        Args:
            texts (iterable): List or Series of text strings.
            load_cached_models (bool): If True, attempts to load models from disk.
        """
        # Check if models exist
        if (
            load_cached_models
            and os.path.exists(self.config.TFIDF_PATH)
            and os.path.exists(self.config.SVD_PATH)
        ):
            print(f"Loading cached vectorizers from {self.config.WORKING_DIR}...")
            self.load()
            return self

        print("Fitting TF-IDF Vectorizer...")
        self.tfidf = TfidfVectorizer(
            max_features=self.config.VOCAB_SIZE,
            ngram_range=self.config.NGRAM_RANGE,
            min_df=self.config.MIN_DF,
            token_pattern=self.config.TOKEN_PATTERN,
            dtype=np.float32,
        )
        # Fit TF-IDF
        sparse_matrix = self.tfidf.fit_transform(texts)

        print(f"Fitting Truncated SVD (components={self.config.SVD_COMPONENTS})...")
        self.svd = TruncatedSVD(
            n_components=self.config.SVD_COMPONENTS,
            n_iter=self.config.SVD_ITER,
            random_state=self.config.SVD_RANDOM_STATE,
        )
        self.svd.fit(sparse_matrix)

        print("Saving vectorizers...")
        self.save()
        return self

    def transform(self, texts):
        """
        Transforms texts into SVD embeddings.

        Args:
            texts (iterable): List or Series of text strings.

        Returns:
            np.ndarray: Dense SVD embeddings (N_samples, SVD_COMPONENTS).
        """
        if self.tfidf is None or self.svd is None:
            raise ValueError("Models not fitted. Call fit() or load() first.")

        # Transform to sparse TF-IDF
        sparse = self.tfidf.transform(texts)
        # Transform to dense SVD
        dense = self.svd.transform(sparse)
        return dense.astype(np.float32)

    def save(self):
        """Saves models to disk."""
        os.makedirs(os.path.dirname(self.config.TFIDF_PATH), exist_ok=True)
        joblib.dump(self.tfidf, self.config.TFIDF_PATH)
        joblib.dump(self.svd, self.config.SVD_PATH)

    def load(self):
        """Loads models from disk."""
        self.tfidf = joblib.load(self.config.TFIDF_PATH)
        self.svd = joblib.load(self.config.SVD_PATH)


def _process_single_notebook(nb_id, df_nb, emb_nb, top_k):
    """
    Helper function to process anchors for a single notebook.
    """
    # Indices relative to the subset arrays
    is_code = (df_nb["cell_type"] == "code").values
    is_md = (df_nb["cell_type"] == "markdown").values

    # If no code cells or no markdown cells, return empty/default
    if not np.any(is_code) or not np.any(is_md):
        # Return dict of NaNs for MD rows
        n_rows = len(df_nb)
        return {
            "indices": df_nb.index.values,
            "anchor_mean_rank": np.full(n_rows, np.nan, dtype=np.float32),
            "anchor_weighted_rank": np.full(n_rows, np.nan, dtype=np.float32),
            "anchor_min_dist": np.full(n_rows, np.nan, dtype=np.float32),
            "anchor_nearest_rank": np.full(n_rows, np.nan, dtype=np.float32),
        }

    # Extract embeddings
    code_emb = emb_nb[is_code]
    md_emb = emb_nb[is_md]

    # Determine Code Ranks
    # For Training: we could use 'norm_rank' from metadata.
    # For Inference: 'norm_rank' is NaN.
    # Strategy: Always re-compute code ranks based on their current sequence order.
    # This assumes code cells in df_nb are in the correct relative order (which is true for Train and Test).
    n_code = len(code_emb)
    if n_code > 1:
        code_ranks = np.linspace(0.0, 1.0, n_code, dtype=np.float32)
    else:
        code_ranks = np.array([0.0], dtype=np.float32)

    # Compute Distance Matrix (MD x Code)
    # Euclidean distance
    dists = euclidean_distances(md_emb, code_emb)

    # Initialize result arrays for MD cells
    n_md = len(md_emb)
    feat_mean = np.zeros(n_md, dtype=np.float32)
    feat_weighted = np.zeros(n_md, dtype=np.float32)
    feat_min_dist = np.zeros(n_md, dtype=np.float32)
    feat_nearest = np.zeros(n_md, dtype=np.float32)

    # Loop over MD cells to find top-k anchors
    # (Vectorization over K is possible but loop is clear and fast enough for small N)
    for i in range(n_md):
        row_dists = dists[i]

        # Sort indices by distance
        sorted_idx = np.argsort(row_dists)

        # Select Top-K
        k_curr = min(len(sorted_idx), top_k)
        top_indices = sorted_idx[:k_curr]

        top_dists = row_dists[top_indices]
        top_ranks = code_ranks[top_indices]

        # 1. Mean Rank
        feat_mean[i] = np.mean(top_ranks)

        # 2. Weighted Mean Rank (Inverse Distance Weighting)
        # Add epsilon to avoid division by zero
        weights = 1.0 / (top_dists + 1e-6)
        feat_weighted[i] = np.sum(top_ranks * weights) / np.sum(weights)

        # 3. Min Distance
        feat_min_dist[i] = top_dists[0]

        # 4. Nearest Rank
        feat_nearest[i] = top_ranks[0]

    # Map back to original dataframe size
    n_total = len(df_nb)
    res_mean = np.full(n_total, np.nan, dtype=np.float32)
    res_weighted = np.full(n_total, np.nan, dtype=np.float32)
    res_min_dist = np.full(n_total, np.nan, dtype=np.float32)
    res_nearest = np.full(n_total, np.nan, dtype=np.float32)

    # Fill MD slots
    # is_md is boolean mask aligned with 0..n_total-1
    res_mean[is_md] = feat_mean
    res_weighted[is_md] = feat_weighted
    res_min_dist[is_md] = feat_min_dist
    res_nearest[is_md] = feat_nearest

    return {
        "indices": df_nb.index.values,
        "anchor_mean_rank": res_mean,
        "anchor_weighted_rank": res_weighted,
        "anchor_min_dist": res_min_dist,
        "anchor_nearest_rank": res_nearest,
    }


def generate_anchor_features(df, embeddings, top_k=Config.TOP_K_ANCHORS, n_jobs=-1):
    """
    Generates features based on the proximity of markdown cells to code cells
    in the embedding space.

    Args:
        df (pd.DataFrame): Dataframe containing 'id' and 'cell_type'.
        embeddings (np.ndarray): Feature matrix aligned with df.
        top_k (int): Number of nearest code anchors to consider.
        n_jobs (int): Number of parallel jobs.

    Returns:
        pd.DataFrame: DataFrame with anchor features, aligned with df index.
    """
    print(f"Generating anchor features (Top-K={top_k})...")

    # Ensure index is unique and sorted for reconstruction
    if not df.index.is_unique:
        df = df.reset_index(drop=True)

    # Group by notebook ID
    # We pass indices to helper to reconstruct the order later
    groups = df.groupby("id")

    # Prepare generator for parallel execution
    tasks = (
        (nb_id, df.loc[indices], embeddings[indices])
        for nb_id, indices in groups.indices.items()
    )

    results = Parallel(n_jobs=n_jobs)(
        delayed(_process_single_notebook)(nb_id, df_nb, emb_nb, top_k)
        for nb_id, df_nb, emb_nb in tqdm(tasks, total=len(groups))
    )

    # Reassemble results
    # Each result is a dict with 'indices' and feature arrays
    all_indices = []
    feat_cols = {
        "anchor_mean_rank": [],
        "anchor_weighted_rank": [],
        "anchor_min_dist": [],
        "anchor_nearest_rank": [],
    }

    for res in results:
        all_indices.append(res["indices"])
        for k in feat_cols:
            feat_cols[k].append(res[k])

    # Concatenate
    flat_indices = np.concatenate(all_indices)
    flat_feats = {k: np.concatenate(v) for k, v in feat_cols.items()}

    # Create DataFrame
    df_feats = pd.DataFrame(flat_feats, index=flat_indices)

    # Sort by index to match original df order
    df_feats = df_feats.sort_index()

    return df_feats
