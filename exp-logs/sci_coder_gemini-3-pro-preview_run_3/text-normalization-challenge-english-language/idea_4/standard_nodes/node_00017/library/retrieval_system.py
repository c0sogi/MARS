import os
import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from typing import List, Dict, Union, Optional, Tuple

from library.config import Config
from library.text_utils import is_hard_token


class SimilarityIndex:
    """
    Implements the retrieval mechanism for the Retrieval-Augmented Generation (RAG) system.

    This class indexes 'hard' tokens from the training set using character-level TF-IDF
    and allows for retrieving the most structurally similar examples using K-Nearest Neighbors.
    """

    def __init__(self):
        # Character-level TF-IDF to capture morphological structure (e.g., "$3.50" vs "$4.20")
        self.vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(1, 3),  # Unigrams to Trigrams
            min_df=2,  # Ignore extremely rare character patterns
            strip_accents="unicode",
        )

        # Cosine similarity is generally effective for high-dimensional sparse TF-IDF vectors
        self.knn = NearestNeighbors(metric="cosine", n_jobs=-1)

        # DataFrame to store the reference data (raw and normalized forms)
        self.hard_samples_df: Optional[pd.DataFrame] = None
        self.is_fitted = False

    def build_index(self, load_cached_data: bool = True):
        """
        Builds or loads the similarity index from the training data.

        Args:
            load_cached_data: If True, attempts to load pre-computed models and data
                              from the cache directory defined in Config.
        """
        # Define paths from Config
        tfidf_path = Config.TFIDF_MODEL_PATH
        knn_path = Config.KNN_INDEX_PATH
        samples_path = Config.HARD_SAMPLES_PATH

        # Check if cache exists
        cache_exists = (
            os.path.exists(tfidf_path)
            and os.path.exists(knn_path)
            and os.path.exists(samples_path)
        )

        if load_cached_data and cache_exists:
            print("Loading retrieval index from cache...")
            try:
                self.vectorizer = joblib.load(tfidf_path)
                self.knn = joblib.load(knn_path)
                self.hard_samples_df = pd.read_parquet(samples_path)
                self.is_fitted = True
                print(f"Index loaded. Reference size: {len(self.hard_samples_df)}")
                return
            except Exception as e:
                print(f"Failed to load cache: {e}. Rebuilding index...")

        print("Building retrieval index from scratch...")
        self._compute_and_save_index(tfidf_path, knn_path, samples_path)
        self.is_fitted = True

    def _compute_and_save_index(
        self, tfidf_path: str, knn_path: str, samples_path: str
    ):
        """
        Computes the TF-IDF vectors and KNN index from raw training data and saves them.
        """
        # Ensure working directory exists
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        os.makedirs(Config.PROCESSED_DIR, exist_ok=True)

        # 1. Load Training Data
        if not os.path.exists(Config.TRAIN_DATA_PATH):
            raise FileNotFoundError(
                f"Training data not found at {Config.TRAIN_DATA_PATH}"
            )

        print("Loading training data for indexing...")
        df = pd.read_parquet(Config.TRAIN_DATA_PATH)

        # Optional: Debugging subsample
        if Config.MAX_TRAIN_SAMPLES is not None:
            df = df.head(Config.MAX_TRAIN_SAMPLES).copy()

        # Ensure string types
        df["before"] = df["before"].astype(str)
        df["after"] = df["after"].astype(str)
        df["class"] = df["class"].astype(str)

        # 2. Filter for "Hard" tokens
        # We use vectorized operations for speed instead of applying the function row-by-row
        # Condition 1: Class is not PLAIN or PUNCT
        cond_class = ~df["class"].isin(["PLAIN", "PUNCT"])
        # Condition 2: Text is not purely alphabetic
        cond_alpha = ~df["before"].str.isalpha()

        hard_mask = cond_class & cond_alpha
        df_hard = df[hard_mask].copy()

        print(f"Filtered {len(df_hard)} hard tokens from {len(df)} total tokens.")

        # 3. Deduplicate
        # We only need unique input patterns for the index.
        # If there are conflicts (same input, different output), we keep the first occurrence.
        # In a retrieval scenario, finding *a* valid similar example is often sufficient.
        self.hard_samples_df = df_hard.drop_duplicates(subset=["before"]).reset_index(
            drop=True
        )

        if len(self.hard_samples_df) == 0:
            print("Warning: No hard tokens found. Index will be empty.")
            # Create dummy to prevent errors
            self.hard_samples_df = pd.DataFrame(
                {"before": ["dummy"], "after": ["dummy"]}
            )

        print(f"Unique hard tokens to index: {len(self.hard_samples_df)}")

        # 4. Fit Vectorizer
        print("Fitting TF-IDF Vectorizer...")
        tfidf_matrix = self.vectorizer.fit_transform(self.hard_samples_df["before"])

        # 5. Fit KNN
        print("Fitting Nearest Neighbors Index...")
        # n_neighbors must be at least 1, but we can set a higher default for the model structure
        k = min(5, len(self.hard_samples_df))
        self.knn.set_params(n_neighbors=k)
        self.knn.fit(tfidf_matrix)

        # 6. Save Artifacts
        print("Saving artifacts...")
        joblib.dump(self.vectorizer, tfidf_path)
        joblib.dump(self.knn, knn_path)
        self.hard_samples_df.to_parquet(samples_path, index=False)

        print("Index build complete.")

    def retrieve(self, query_text: str, k: int = 1) -> List[Dict[str, str]]:
        """
        Retrieves the k nearest neighbors for a given query string.

        Args:
            query_text: The raw token text to normalize.
            k: Number of neighbors to retrieve.

        Returns:
            A list of dictionaries, each containing:
            - 'source': The raw text of the neighbor.
            - 'target': The normalized text of the neighbor.
            - 'distance': The cosine distance to the query.
        """
        if not self.is_fitted:
            raise RuntimeError("Index not fitted. Call build_index() first.")

        # Handle empty or purely whitespace queries gracefully
        if not query_text or not query_text.strip():
            return []

        # Transform query
        try:
            query_vec = self.vectorizer.transform([query_text])
        except ValueError:
            # Can happen if query contains only chars not in vocab
            return []

        # Query KNN
        # Ensure k doesn't exceed index size
        k_eff = min(k, len(self.hard_samples_df))
        distances, indices = self.knn.kneighbors(query_vec, n_neighbors=k_eff)

        results = []
        # kneighbors returns arrays of shape (n_queries, n_neighbors)
        # Since we query one item, we take the first row [0]
        row_indices = indices[0]
        row_distances = distances[0]

        for idx, dist in zip(row_indices, row_distances):
            record = self.hard_samples_df.iloc[idx]
            results.append(
                {
                    "source": record["before"],
                    "target": record["after"],
                    "distance": float(dist),
                }
            )

        return results

    def retrieve_batch(
        self, queries: List[str], k: int = 1
    ) -> List[List[Dict[str, str]]]:
        """
        Batch version of retrieve for efficient processing during training/inference.
        """
        if not self.is_fitted:
            raise RuntimeError("Index not fitted.")

        if not queries:
            return []

        # Transform all queries
        query_vecs = self.vectorizer.transform(queries)

        k_eff = min(k, len(self.hard_samples_df))
        distances, indices = self.knn.kneighbors(query_vecs, n_neighbors=k_eff)

        batch_results = []
        for i in range(len(queries)):
            row_results = []
            for j in range(k_eff):
                idx = indices[i][j]
                dist = distances[i][j]
                record = self.hard_samples_df.iloc[idx]
                row_results.append(
                    {
                        "source": record["before"],
                        "target": record["after"],
                        "distance": float(dist),
                    }
                )
            batch_results.append(row_results)

        return batch_results
