import os
import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from library.config import Config
from library.utils import preprocess_text


class VectorizationPipeline:
    """
    Manages TF-IDF and SVD models for text vectorization.
    Persists models to the working directory.
    """

    def __init__(self):
        self.tfidf_path = os.path.join(Config.WORKING_DIR, "tfidf_vectorizer.joblib")
        self.svd_path = os.path.join(Config.WORKING_DIR, "svd_model.joblib")
        self.tfidf = None
        self.svd = None

    def fit(self, texts):
        """
        Fits TF-IDF and SVD models on the provided texts.
        """
        print("Fitting TF-IDF Vectorizer...")
        self.tfidf = TfidfVectorizer(
            max_features=Config.VOCAB_SIZE,
            ngram_range=Config.NGRAM_RANGE,
            sublinear_tf=True,
            token_pattern=Config.TOKEN_PATTERN,
            strip_accents=None,  # Preserving accents as per strategy
        )
        tfidf_matrix = self.tfidf.fit_transform(texts)

        print(f"Fitting Truncated SVD (components={Config.SVD_COMPONENTS})...")
        self.svd = TruncatedSVD(
            n_components=Config.SVD_COMPONENTS, random_state=Config.SVD_RANDOM_STATE
        )
        self.svd.fit(tfidf_matrix)

        self.save_models()

    def transform(self, texts):
        """
        Transforms texts into TF-IDF and SVD representations.
        Returns:
            tfidf_matrix (sparse), svd_matrix (dense)
        """
        if self.tfidf is None or self.svd is None:
            self.load_models()

        # Handle case where texts might be empty or contain NaNs
        texts = [t if isinstance(t, str) else "" for t in texts]

        tfidf_matrix = self.tfidf.transform(texts)
        svd_matrix = self.svd.transform(tfidf_matrix)
        return tfidf_matrix, svd_matrix

    def save_models(self):
        joblib.dump(self.tfidf, self.tfidf_path)
        joblib.dump(self.svd, self.svd_path)
        print(f"Vectorization models saved to {Config.WORKING_DIR}")

    def load_models(self):
        if os.path.exists(self.tfidf_path) and os.path.exists(self.svd_path):
            self.tfidf = joblib.load(self.tfidf_path)
            self.svd = joblib.load(self.svd_path)
        else:
            raise FileNotFoundError(
                "Vectorization models not found. Please fit them first."
            )


class AnchorFeatureGenerator:
    """
    Generates Multi-View Symbolic and Semantic Anchor features.
    """

    def __init__(self):
        self.pipeline = VectorizationPipeline()

    def _compute_jaccard_similarity(self, set_a, set_b):
        """
        Computes Jaccard similarity between two sets of identifiers.
        """
        if not set_a and not set_b:
            return 0.0
        intersection = len(set_a.intersection(set_b))
        union = len(set_a.union(set_b))
        return intersection / union if union > 0 else 0.0

    def _process_notebook(self, nb_df, tfidf_matrix, svd_matrix):
        """
        Computes anchor features for a single notebook.
        """
        # Separate Code and Markdown
        code_mask = nb_df["cell_type"] == "code"
        md_mask = nb_df["cell_type"] == "markdown"

        code_df = nb_df[code_mask]
        md_df = nb_df[md_mask]

        if len(code_df) == 0:
            # Edge case: No code cells. Return default rank (0.5)
            features = pd.DataFrame(index=md_df.index)
            features["lexical_anchor"] = 0.5
            features["latent_anchor"] = 0.5
            features["symbolic_anchor"] = 0.5
            # Add SVD features
            md_svd = svd_matrix[md_mask.values]
            for i in range(Config.SVD_COMPONENTS):
                features[f"svd_{i}"] = md_svd[:, i]
            return features

        if len(md_df) == 0:
            return pd.DataFrame()

        # Matrices for this notebook
        # Note: tfidf_matrix and svd_matrix correspond to the rows of nb_df
        # We need to map local indices correctly

        # Get indices relative to the passed matrices (which are subsets of the full split)
        # However, to simplify, we assume the matrices passed correspond exactly to nb_df rows in order

        code_indices = np.where(code_mask.values)[0]
        md_indices = np.where(md_mask.values)[0]

        code_tfidf = tfidf_matrix[code_indices]
        md_tfidf = tfidf_matrix[md_indices]

        code_svd = svd_matrix[code_indices]
        md_svd = svd_matrix[md_indices]

        code_ranks = code_df["pct_rank"].values

        # 1. Lexical Anchors (TF-IDF Cosine)
        # Shape: (n_md, n_code)
        lex_sim = cosine_similarity(md_tfidf, code_tfidf)

        # 2. Latent Anchors (SVD Cosine)
        lat_sim = cosine_similarity(md_svd, code_svd)

        # 3. Symbolic Anchors (Jaccard)
        # Extract sets once
        code_ids = [set(x) for x in code_df["identifiers"].values]
        md_ids = [set(x) for x in md_df["identifiers"].values]

        n_md = len(md_df)
        n_code = len(code_df)
        sym_sim = np.zeros((n_md, n_code))

        for i in range(n_md):
            for j in range(n_code):
                sym_sim[i, j] = self._compute_jaccard_similarity(md_ids[i], code_ids[j])

        # Aggregate Top-K
        k = min(Config.ANCHOR_K, n_code)

        def get_top_k_mean_rank(sim_matrix):
            # argsort returns indices of sorted elements. We want descending order.
            # Take last k elements
            top_k_idx = np.argsort(sim_matrix, axis=1)[:, -k:]
            # Retrieve ranks
            # code_ranks shape (n_code,)
            # top_k_ranks shape (n_md, k)
            top_k_ranks = code_ranks[top_k_idx]
            return np.mean(top_k_ranks, axis=1)

        lex_anchors = get_top_k_mean_rank(lex_sim)
        lat_anchors = get_top_k_mean_rank(lat_sim)
        sym_anchors = get_top_k_mean_rank(sym_sim)

        # Construct Result
        features = pd.DataFrame(index=md_df.index)
        features["lexical_anchor"] = lex_anchors
        features["latent_anchor"] = lat_anchors
        features["symbolic_anchor"] = sym_anchors

        # Add SVD components for the markdown cells
        for i in range(Config.SVD_COMPONENTS):
            features[f"svd_{i}"] = md_svd[:, i]

        return features

    def generate_features(self, df_cells, split_name="train", load_cached_data=True):
        """
        Main method to generate features for a given dataframe of cells.
        Handles Caching.
        """
        cache_path = os.path.join(Config.WORKING_DIR, f"features_{split_name}.parquet")

        # 1. Try Load Cache
        if load_cached_data and os.path.exists(cache_path):
            try:
                print(f"Loading cached features from {cache_path}...")
                return pd.read_parquet(cache_path)
            except Exception as e:
                print(f"Failed to load feature cache: {e}")

        print(
            f"Generating features for {split_name} (Input shape: {df_cells.shape})..."
        )

        # 2. Vectorization
        # If train, we might need to fit. If val/test, transform only.
        # Ideally, fit is called externally or we check if models exist.
        # For this implementation, we assume if split_name=='train', we fit.

        all_sources = df_cells["source_clean"].fillna("").tolist()

        if split_name == "train":
            # Check if models exist to avoid re-fitting if we are just re-running feature gen
            # But strictly, if we are generating train features, we should ensure fit.
            self.pipeline.fit(all_sources)
            tfidf_mat, svd_mat = self.pipeline.transform(all_sources)
        else:
            tfidf_mat, svd_mat = self.pipeline.transform(all_sources)

        # 3. Compute Anchors Group-wise
        # We need to iterate by notebook to perform pairwise comparisons
        # To make this efficient, we can groupby notebook_id

        feature_dfs = []

        # Get unique notebook IDs
        nb_ids = df_cells["notebook_id"].unique()

        # Map global indices to notebook groups
        # We can use groupby on the dataframe, but we need to slice the sparse matrices.
        # Groupby preserves order if sort=False? Not necessarily.
        # Safer approach:

        # Create an array of indices
        df_cells = df_cells.reset_index(drop=True)  # Ensure continuous index
        df_cells["global_idx"] = df_cells.index

        grouped = df_cells.groupby("notebook_id", sort=False)

        count = 0
        total = len(grouped)

        for nb_id, group in grouped:
            global_indices = group["global_idx"].values

            # Slice matrices
            nb_tfidf = tfidf_mat[global_indices]
            nb_svd = svd_mat[global_indices]

            # Compute features
            nb_feats = self._process_notebook(group, nb_tfidf, nb_svd)

            # We need to preserve the original index or ID to merge back
            # nb_feats index is the original index from group (which is from df_cells)
            feature_dfs.append(nb_feats)

            count += 1
            if count % 1000 == 0:
                print(f"Processed {count}/{total} notebooks...")

        # Concatenate all features
        if feature_dfs:
            full_features = pd.concat(feature_dfs)
        else:
            full_features = pd.DataFrame()

        # Merge back with key identifiers from df_cells (like cell_id, notebook_id, rank)
        # We only computed features for markdown cells.
        # The output should ideally be aligned with the markdown rows of df_cells.

        # Filter df_cells to markdown only to align
        md_cells = df_cells[df_cells["cell_type"] == "markdown"]

        # Ensure alignment
        # full_features index corresponds to md_cells index
        result = md_cells[["notebook_id", "cell_id", "rank", "pct_rank"]].join(
            full_features
        )

        # 4. Save Cache
        try:
            result.to_parquet(cache_path, index=False)
            print(f"Saved features to {cache_path}")
        except Exception as e:
            print(f"Warning: Could not save feature cache: {e}")

        return result
