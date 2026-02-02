import os
import json
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy import sparse
from library.utils import setup_logger, load_or_create_cache, timer
from library.config import Config

# Initialize logger
logger = setup_logger("feature_engineering")


class BM25Transformer:
    """
    A simplified BM25 implementation using sklearn's CountVectorizer and sparse matrix operations.
    """

    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.vectorizer = None
        self.idf = None
        self.avgdl = 0
        self.is_fitted = False

    def fit(self, corpus):
        """
        Fit the BM25 model on a corpus of text.
        Args:
            corpus (list/Series): List of document strings.
        """
        self.vectorizer = CountVectorizer(binary=False)
        X = self.vectorizer.fit_transform(corpus)

        # Calculate IDF
        N = X.shape[0]
        # Document frequency
        df = np.bincount(X.indices, minlength=X.shape[1])
        # Standard BM25 IDF formula with smoothing
        self.idf = np.log((N - df + 0.5) / (df + 0.5) + 1)

        # Calculate Average Document Length
        doc_lengths = np.array(X.sum(axis=1)).flatten()
        self.avgdl = np.mean(doc_lengths)
        self.is_fitted = True
        return self

    def score(self, queries, documents):
        """
        Compute BM25 scores for pairs of (query, document).
        This is optimized for pair-wise scoring rather than retrieval.
        Args:
            queries (list/Series): List of query strings.
            documents (list/Series): List of document strings.
        Returns:
            np.array: BM25 scores.
        """
        if not self.is_fitted:
            raise ValueError("BM25Transformer must be fitted before scoring.")

        # Transform both queries and docs using the same vectorizer
        q_vecs = self.vectorizer.transform(queries)
        d_vecs = self.vectorizer.transform(documents)

        # Get document lengths for normalization
        d_len = np.array(d_vecs.sum(axis=1)).flatten()
        len_norm = (1 - self.b) + self.b * (d_len / self.avgdl)

        # We need to compute the score for each pair (q_i, d_i).
        # Score = sum( IDF * (TF * (k1+1)) / (TF + k1 * len_norm) )
        # Since we can't easily do efficient sparse row-wise operations for varying TFs in a batch
        # without expanding, we iterate or use a simplified approach.
        # Given the constraints, we process in batches or use a dot product approximation
        # if we assume binary query term presence.

        # Precise implementation for paired data:
        scores = []
        # Convert to CSR for efficient row slicing
        q_vecs_csr = q_vecs.tocsr()
        d_vecs_csr = d_vecs.tocsr()

        # To optimize, we can operate on the non-zero elements directly.
        # However, a simple loop might be too slow.
        # Vectorized approach:
        # 1. Calculate the denominator term for all non-zero elements in D
        # 2. Multiply by numerator term
        # 3. Filter by Q terms

        # Approximation: For NQ, queries are short.
        # We can calculate term-wise scores and sum.

        # Let's use a simpler approach:
        # Score = Sum_{t in q} IDF_t * (TF_td * (k1+1)) / (TF_td + k1 * len_norm_d)

        # We can perform this calculation efficiently if we treat it as a sparse operation.
        # However, len_norm depends on the document index.

        # Fallback to a loop with optimization for this baseline
        # (Assuming dataset size allows or batching is used externally)
        n_samples = len(queries)
        result = np.zeros(n_samples)

        for i in range(n_samples):
            q_indices = q_vecs_csr[i].indices
            d_indices = d_vecs_csr[i].indices
            d_data = d_vecs_csr[i].data

            # Intersection of terms
            common_terms, d_idx, _ = np.intersect1d(
                d_indices, q_indices, return_indices=True, return_intersection=True
            )

            if len(common_terms) == 0:
                continue

            tf = d_data[d_idx]
            denom = tf + self.k1 * len_norm[i]
            numer = tf * (self.k1 + 1)
            term_scores = self.idf[common_terms] * (numer / denom)
            result[i] = np.sum(term_scores)

        return result

    def save_state(self, path):
        """Save vocabulary and IDF to disk."""
        state = {
            "vocabulary": self.vectorizer.vocabulary_,
            "idf": self.idf.tolist(),
            "avgdl": self.avgdl,
        }
        with open(path, "w") as f:
            json.dump(state, f)

    def load_state(self, path):
        """Load state from disk."""
        with open(path, "r") as f:
            state = json.load(f)

        self.vectorizer = CountVectorizer(vocabulary=state["vocabulary"])
        # Dummy fit to initialize attributes
        self.vectorizer.fit([""])
        self.idf = np.array(state["idf"])
        self.avgdl = state["avgdl"]
        self.is_fitted = True


class FeatureGenerator:
    """
    Handles feature engineering for the NQ dataset.
    """

    def __init__(self, config: Config):
        self.config = config
        self.tfidf = TfidfVectorizer(
            lowercase=True, stop_words="english", max_features=config.MAX_VOCAB_SIZE
        )
        self.bm25 = BM25Transformer()
        self.is_fitted = False

    def _compute_jaccard(self, str1, str2):
        """Compute Jaccard similarity between two strings."""
        set1 = set(str(str1).lower().split())
        set2 = set(str(str2).lower().split())
        if not set1 or not set2:
            return 0.0
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union

    def _extract_tag(self, text):
        """Extract the first HTML tag from text."""
        match = re.match(
            r"<(P|Table|Ul|Ol|Dl|H1|H2|H3|H4|H5|H6)>", str(text), re.IGNORECASE
        )
        if match:
            return match.group(1).upper()
        return "OTHER"

    def fit(self, df: pd.DataFrame):
        """
        Fit vectorizers on the provided dataframe.
        Args:
            df: DataFrame containing 'question_text' and 'candidate_text' columns.
        """
        logger.info("Fitting TF-IDF and BM25 vectorizers...")

        # Combine questions and candidates for a richer vocabulary,
        # or just use candidates as the corpus.
        # Using a sample to speed up if dataset is huge.
        corpus = df["candidate_text"].fillna("").tolist()

        # Fit TF-IDF
        self.tfidf.fit(corpus)

        # Fit BM25
        self.bm25.fit(corpus)

        self.is_fitted = True

        # Save states for inference
        self._save_state()

    def _save_state(self):
        """Save vectorizer states to cache."""
        # Save TF-IDF vocabulary and idf
        tfidf_state = {
            "vocabulary": self.tfidf.vocabulary_,
            "idf": self.tfidf.idf_.tolist(),
        }
        with open(self.config.get_cache_path("tfidf_state.json"), "w") as f:
            json.dump(tfidf_state, f)

        # Save BM25 state
        self.bm25.save_state(self.config.get_cache_path("bm25_state.json"))

    def load_state(self):
        """Load vectorizer states from cache."""
        tfidf_path = self.config.get_cache_path("tfidf_state.json")
        bm25_path = self.config.get_cache_path("bm25_state.json")

        if not os.path.exists(tfidf_path) or not os.path.exists(bm25_path):
            logger.warning("Vectorizer states not found. Call fit() first.")
            return False

        # Load TF-IDF
        with open(tfidf_path, "r") as f:
            state = json.load(f)
        self.tfidf = TfidfVectorizer(vocabulary=state["vocabulary"])
        self.tfidf.idf_ = np.array(state["idf"])
        # Mock fit
        self.tfidf.fit([""])

        # Load BM25
        self.bm25.load_state(bm25_path)

        self.is_fitted = True
        return True

    def generate_features(
        self, df: pd.DataFrame, is_training: bool = False
    ) -> pd.DataFrame:
        """
        Main method to generate features.

        Args:
            df: Input DataFrame with raw text and metadata.
            is_training: Boolean flag.

        Returns:
            DataFrame with numerical features.
        """
        if not self.is_fitted:
            if not self.load_state():
                if is_training:
                    self.fit(df)
                else:
                    raise RuntimeError(
                        "FeatureGenerator is not fitted and no cache found for inference."
                    )

        logger.info(f"Generating features for {len(df)} samples...")

        # Ensure text columns are strings
        df["question_text"] = df["question_text"].fillna("")
        df["candidate_text"] = df["candidate_text"].fillna("")

        features = pd.DataFrame(index=df.index)

        # 1. Lexical Features
        if self.config.USE_TFIDF:
            logger.info("Computing TF-IDF cosine similarity...")
            q_tfidf = self.tfidf.transform(df["question_text"])
            c_tfidf = self.tfidf.transform(df["candidate_text"])
            # Row-wise dot product
            features["tfidf_score"] = np.array(
                q_tfidf.multiply(c_tfidf).sum(axis=1)
            ).flatten()

        if self.config.USE_BM25:
            logger.info("Computing BM25 scores...")
            features["bm25_score"] = self.bm25.score(
                df["question_text"], df["candidate_text"]
            )

        if self.config.USE_JACCARD:
            logger.info("Computing Jaccard similarity...")
            features["jaccard_score"] = df.apply(
                lambda x: self._compute_jaccard(
                    x["question_text"], x["candidate_text"]
                ),
                axis=1,
            )

        # 2. Structural Features
        logger.info("Extracting structural features...")
        features["candidate_len"] = df["candidate_text"].apply(
            lambda x: len(str(x).split())
        )

        # Extract HTML tag from the start of candidate text
        if self.config.USE_HTML_TAGS:
            tags = df["candidate_text"].apply(self._extract_tag)
            # One-hot encode common tags
            common_tags = ["P", "TABLE", "UL", "OL", "H1", "H2", "DL"]
            for tag in common_tags:
                features[f"tag_is_{tag}"] = (tags == tag).astype(int)
            features["tag_is_OTHER"] = (~tags.isin(common_tags)).astype(int)

        features["is_top_level"] = df["top_level"].astype(int)

        # Normalized position (assuming start_token is available)
        # We need document length to normalize, but if not available, we use raw start_token
        # Or relative rank if grouped.
        features["start_token"] = df["start_token"]

        # 3. Contextual Features (Lag/Lead)
        logger.info("Adding contextual features...")
        # Ensure data is sorted by document and position
        df_sorted = df.sort_values(by=["example_id", "start_token"])
        # Map sorted index back to original features index
        features = features.loc[df_sorted.index]

        # Group by example_id to prevent shifting across documents
        grouped = features.groupby(df_sorted["example_id"])

        # Shift scores
        score_cols = (
            ["tfidf_score", "bm25_score"] if self.config.USE_BM25 else ["tfidf_score"]
        )

        for col in score_cols:
            for i in range(1, self.config.CONTEXT_WINDOW_SIZE + 1):
                features[f"prev_{col}_{i}"] = grouped[col].shift(i).fillna(0)
                features[f"next_{col}_{i}"] = grouped[col].shift(-i).fillna(0)

        # Restore original order
        features = features.loc[df.index]

        return features

    def process_and_cache_features(
        self, df: pd.DataFrame, split_name: str, load_cached_data: bool
    ) -> pd.DataFrame:
        """
        Wrapper to handle caching of generated features.

        Args:
            df: Input DataFrame.
            split_name: 'train', 'val', or 'test'.
            load_cached_data: Whether to try loading from disk.

        Returns:
            DataFrame with features.
        """
        file_name = f"{split_name}_features.parquet"
        file_path = self.config.get_cache_path(file_name)

        def _process():
            is_training = split_name == "train"
            return self.generate_features(df, is_training=is_training)

        return load_or_create_cache(
            file_path=file_path,
            process_fn=_process,
            load_cached_data=load_cached_data,
            file_type="parquet",
        )
