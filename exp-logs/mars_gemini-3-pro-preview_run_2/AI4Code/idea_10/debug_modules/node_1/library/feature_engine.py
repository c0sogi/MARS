import os
import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from library.config import Config


class DualViewVectorizer:
    """
    Manages the Lexical (TF-IDF) and Latent (SVD) vectorization models.
    """

    def __init__(self):
        self.config = Config
        self.tfidf = TfidfVectorizer(**self.config.TFIDF_PARAMS)
        self.svd = TruncatedSVD(**self.config.SVD_PARAMS)

    def fit(self, text_corpus):
        """
        Fits the TF-IDF and SVD models on the provided text corpus.
        """
        print("Fitting TF-IDF Vectorizer...")
        tfidf_matrix = self.tfidf.fit_transform(text_corpus)

        print("Fitting Truncated SVD...")
        self.svd.fit(tfidf_matrix)

        self.save_models()

    def transform(self, text_corpus):
        """
        Transforms text into both Sparse (TF-IDF) and Dense (SVD) representations.
        Returns:
            sparse_matrix, dense_matrix
        """
        sparse_matrix = self.tfidf.transform(text_corpus)
        dense_matrix = self.svd.transform(sparse_matrix)
        return sparse_matrix, dense_matrix

    def save_models(self):
        """Saves models to the cache directory."""
        joblib.dump(self.tfidf, self.config.TFIDF_MODEL_PATH)
        joblib.dump(self.svd, self.config.SVD_MODEL_PATH)
        print(f"Models saved to {self.config.CACHE_DIR}")

    def load_models(self):
        """Loads models from the cache directory."""
        if os.path.exists(self.config.TFIDF_MODEL_PATH) and os.path.exists(
            self.config.SVD_MODEL_PATH
        ):
            self.tfidf = joblib.load(self.config.TFIDF_MODEL_PATH)
            self.svd = joblib.load(self.config.SVD_MODEL_PATH)
            print("Models loaded from cache.")
            return True
        return False


class FeatureEngineer:
    """
    Generates features for the ranking task using Dual-View Anchoring.
    """

    def __init__(self):
        self.config = Config
        self.vectorizer = DualViewVectorizer()

    def generate_features(self, df, split="train", load_cached_data=True):
        """
        Main pipeline to generate features.

        Args:
            df (pd.DataFrame): Input dataframe from DataManager.
            split (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to load from parquet cache.

        Returns:
            pd.DataFrame: DataFrame with features and targets.
        """
        # Define cache path based on split
        if split == "train":
            cache_path = self.config.TRAIN_FEATS_PATH
        elif split == "val":
            cache_path = self.config.VAL_FEATS_PATH
        else:
            cache_path = self.config.TEST_FEATS_PATH

        # 1. Check Cache
        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading {split} features from cache: {cache_path}")
            return pd.read_parquet(cache_path)

        print(f"Generating features for {split} set...")

        # 2. Vectorization
        # We process all text to get vectors.
        # Ensure we only fit on TRAIN split.
        all_text = df["source"].astype(str).fillna("").tolist()

        if split == "train":
            # Filter for markdown cells only for fitting to capture the vocabulary of the target domain
            # Or fit on everything. Fitting on markdown is usually better for this specific task
            # as we want to map markdown to code.
            # However, to ensure code tokens are in vocab, fitting on all is safer or just markdown.
            # Given the task, markdown vocab is crucial. Let's fit on markdown only
            # to avoid code syntax dominating the IDF.
            md_text = (
                df[df["cell_type"] == "markdown"]["source"]
                .astype(str)
                .fillna("")
                .tolist()
            )
            if not self.vectorizer.load_models():
                self.vectorizer.fit(md_text)
        else:
            if not self.vectorizer.load_models():
                raise FileNotFoundError(
                    "Vectorizer models not found. Run training split first."
                )

        # Transform all cells
        print("Transforming text data...")
        sparse_vectors, dense_vectors = self.vectorizer.transform(all_text)

        # 3. Feature Extraction (Notebook-wise)
        # We need to group by notebook to perform anchoring

        # Create a helper dataframe to map index back to vectors
        # We can't put sparse matrices in a DF easily, so we keep them separate and index by integer location

        # Add index to df to track vector rows
        df = df.copy()
        df["vec_idx"] = range(len(df))

        features_list = []

        # Group by notebook
        # Using a loop is necessary here because interaction is within-group
        grouped = df.groupby("id")

        print("Extracting anchor features...")
        # Note: No tqdm as per instructions

        for nb_id, group in grouped:
            # Separate Code and Markdown
            code_mask = group["cell_type"] == "code"
            md_mask = group["cell_type"] == "markdown"

            code_df = group[code_mask]
            md_df = group[md_mask]

            n_code = len(code_df)
            n_md = len(md_df)

            # If no markdown cells, nothing to predict (shouldn't happen in valid data)
            if n_md == 0:
                continue

            # Prepare Markdown Vectors
            md_indices = md_df["vec_idx"].values
            md_sparse = sparse_vectors[md_indices]
            md_dense = dense_vectors[md_indices]

            # Prepare Code Vectors & Ranks
            if n_code > 0:
                code_indices = code_df["vec_idx"].values
                code_sparse = sparse_vectors[code_indices]
                code_dense = dense_vectors[code_indices]

                # Assign equidistant ranks to code cells [0.0, ..., 1.0]
                # Assuming code cells are in correct relative order (as per task description)
                if n_code == 1:
                    code_ranks = np.array([0.0])
                else:
                    code_ranks = np.linspace(0.0, 1.0, n_code)

                # Compute Similarities
                # Shape: (n_md, n_code)
                sim_lexical = cosine_similarity(md_sparse, code_sparse)
                sim_latent = cosine_similarity(md_dense, code_dense)

                # Find Max Sim and corresponding Rank
                # Lexical
                max_lex_idx = np.argmax(sim_lexical, axis=1)
                max_lex_sim = sim_lexical[np.arange(n_md), max_lex_idx]
                neighbor_lex_rank = code_ranks[max_lex_idx]

                # Latent
                max_lat_idx = np.argmax(sim_latent, axis=1)
                max_lat_sim = sim_latent[np.arange(n_md), max_lat_idx]
                neighbor_lat_rank = code_ranks[max_lat_idx]

            else:
                # Fallback if no code cells exist in notebook
                max_lex_sim = np.zeros(n_md)
                neighbor_lex_rank = np.full(n_md, 0.5)
                max_lat_sim = np.zeros(n_md)
                neighbor_lat_rank = np.full(n_md, 0.5)

            # Construct features for this batch
            for i in range(n_md):
                row = md_df.iloc[i]

                feat = {
                    "id": row["id"],
                    "cell_id": row["cell_id"],
                    "rank": row["rank"],  # Target (NaN for test)
                    "ancestor_id": row.get("ancestor_id", row["id"]),
                    # Anchor Features
                    "lexical_max_sim": float(max_lex_sim[i]),
                    "lexical_neighbor_rank": float(neighbor_lex_rank[i]),
                    "latent_max_sim": float(max_lat_sim[i]),
                    "latent_neighbor_rank": float(neighbor_lat_rank[i]),
                    # Metadata
                    "cell_char_len": len(row["source"]),
                    "cell_word_len": len(row["source"].split()),
                    "notebook_code_count": n_code,
                    "notebook_md_count": n_md,
                    "notebook_total_count": n_code + n_md,
                }
                features_list.append(feat)

        # Create DataFrame
        feat_df = pd.DataFrame(features_list)

        # Optimize dtypes
        float_cols = [
            "lexical_max_sim",
            "lexical_neighbor_rank",
            "latent_max_sim",
            "latent_neighbor_rank",
            "rank",
        ]
        for c in float_cols:
            if c in feat_df.columns:
                feat_df[c] = feat_df[c].astype(np.float32)

        int_cols = [
            "cell_char_len",
            "cell_word_len",
            "notebook_code_count",
            "notebook_md_count",
        ]
        for c in int_cols:
            if c in feat_df.columns:
                feat_df[c] = feat_df[c].astype(np.int32)

        # 4. Save to Cache
        print(f"Saving features to {cache_path}")
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        feat_df.to_parquet(cache_path, index=False)

        return feat_df
