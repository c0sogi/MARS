import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from library.config import (
    WORKING_DIR,
    SBERT_MODEL_NAME,
    TFIDF_VOCAB_SIZE,
    TFIDF_NGRAM_RANGE,
    RANDOM_STATE,
)
from library.utils import set_seed

# Ensure deterministic behavior
set_seed(RANDOM_STATE)


class SBERTEncoder:
    def __init__(self, model_name=SBERT_MODEL_NAME):
        self.model_name = model_name
        self.device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") != "" else "cpu"
        # We delay model loading to the process method or first usage to avoid overhead if cached
        self.model = None

    def _load_model(self):
        if self.model is None:
            # print(f"Loading SBERT model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name, device=self.device)

    def encode(self, texts, batch_size=32):
        self._load_model()
        # Ensure texts are strings
        texts = [str(t) if not pd.isna(t) else "" for t in texts]
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings

    def compute_history_centroids(self, subreddits_series):
        """
        Computes the centroid embedding of a user's subreddit history.

        Args:
            subreddits_series: A pandas Series where each entry is a list of subreddit strings.

        Returns:
            numpy.ndarray: Array of shape (N, embedding_dim)
        """
        self._load_model()

        # 1. Identify all unique subreddits
        all_subreddits = set()
        for sub_list in subreddits_series:
            if isinstance(sub_list, list):
                all_subreddits.update(sub_list)

        unique_subs = list(all_subreddits)
        if not unique_subs:
            # Handle edge case where no subreddits exist in the entire series
            # Return zero vectors of appropriate dimension (384 for MiniLM)
            dummy_emb = self.model.encode(["dummy"], show_progress_bar=False)[0]
            return np.zeros((len(subreddits_series), len(dummy_emb)))

        # 2. Encode all unique subreddits
        # Prepend "subreddit " to give context to the model, though raw names often work
        # Here we just encode the name directly as it's a specific entity
        sub_embeddings = self.encode(unique_subs, batch_size=64)
        sub_map = {name: emb for name, emb in zip(unique_subs, sub_embeddings)}

        # 3. Compute centroids
        centroids = []
        embedding_dim = sub_embeddings.shape[1]

        for sub_list in subreddits_series:
            if isinstance(sub_list, list) and len(sub_list) > 0:
                # Gather embeddings for this user
                user_sub_embs = [sub_map[s] for s in sub_list if s in sub_map]
                if user_sub_embs:
                    # Mean pooling
                    centroid = np.mean(user_sub_embs, axis=0)
                    # Normalize centroid to unit length for consistent cosine similarity later
                    norm = np.linalg.norm(centroid)
                    if norm > 1e-9:
                        centroid = centroid / norm
                    centroids.append(centroid)
                else:
                    centroids.append(np.zeros(embedding_dim))
            else:
                centroids.append(np.zeros(embedding_dim))

        return np.array(centroids)

    def compute_consistency_score(self, emb_a, emb_b):
        """
        Computes row-wise cosine similarity between two embedding matrices.
        """
        # Since embeddings are normalized (from encode and compute_history_centroids),
        # cosine similarity is just the dot product.
        # However, we'll use explicit calculation to be safe against zero vectors.

        # Dot product
        dot = np.sum(emb_a * emb_b, axis=1)

        # Norms
        norm_a = np.linalg.norm(emb_a, axis=1)
        norm_b = np.linalg.norm(emb_b, axis=1)

        # Avoid division by zero
        denominator = norm_a * norm_b
        similarity = np.divide(
            dot, denominator, out=np.zeros_like(dot), where=denominator != 0
        )

        return similarity.reshape(-1, 1)

    def process(self, df_train, df_val, df_test):
        """
        Generates SBERT features for all splits.
        """
        # Combine to ensure unique subreddits are covered globally if needed,
        # though processing separately is fine as long as the map is built per call
        # or we build a global map. Here we process each df independently for simplicity,
        # assuming the model is static.

        results = {}

        for split_name, df in [("train", df_train), ("val", df_val), ("test", df_test)]:
            # Text Columns
            # Prioritize edit_aware, fallback to request_text
            if "request_text_edit_aware" in df.columns:
                text_col = "request_text_edit_aware"
            else:
                text_col = "request_text"

            titles = df["request_title"].fillna("").astype(str).tolist()
            texts = df[text_col].fillna("").astype(str).tolist()

            # Encode Content
            title_embeddings = self.encode(titles)
            text_embeddings = self.encode(texts)

            # Encode History
            # Ensure 'requester_subreddits_at_request' is a list
            # It should have been parsed by data_loader, but we double check
            if "requester_subreddits_at_request" in df.columns:
                history_col = df["requester_subreddits_at_request"]
            else:
                history_col = pd.Series([[]] * len(df))

            history_centroids = self.compute_history_centroids(history_col)

            # Compute Consistency
            # Title vs History
            title_hist_sim = self.compute_consistency_score(
                title_embeddings, history_centroids
            )
            # Text vs History
            text_hist_sim = self.compute_consistency_score(
                text_embeddings, history_centroids
            )

            results[split_name] = {
                "title_emb": title_embeddings,
                "text_emb": text_embeddings,
                "history_emb": history_centroids,
                "title_hist_sim": title_hist_sim,
                "text_hist_sim": text_hist_sim,
            }

        return results


class TFIDFEncoder:
    def __init__(self, vocab_size=TFIDF_VOCAB_SIZE, ngram_range=TFIDF_NGRAM_RANGE):
        self.vocab_size = vocab_size
        self.ngram_range = ngram_range
        self.vectorizer = TfidfVectorizer(
            max_features=self.vocab_size,
            ngram_range=self.ngram_range,
            stop_words="english",
            dtype=np.float32,
        )

    def process(self, df_train, df_val, df_test):
        # Prepare corpus
        def get_corpus(df):
            if "request_text_edit_aware" in df.columns:
                text_col = "request_text_edit_aware"
            else:
                text_col = "request_text"

            # Concatenate Title and Text
            t = df["request_title"].fillna("").astype(str)
            b = df[text_col].fillna("").astype(str)
            return (t + " " + b).tolist()

        train_corpus = get_corpus(df_train)
        val_corpus = get_corpus(df_val)
        test_corpus = get_corpus(df_test)

        # Fit on Train
        self.vectorizer.fit(train_corpus)

        # Transform
        X_train = self.vectorizer.transform(train_corpus).toarray()
        X_val = self.vectorizer.transform(val_corpus).toarray()
        X_test = self.vectorizer.transform(test_corpus).toarray()

        return {"train": X_train, "val": X_val, "test": X_test}


def generate_text_features(df_train, df_val, df_test, load_cached_data=True):
    """
    Orchestrates the generation of SBERT and TF-IDF features with caching.

    Returns:
        tuple: (sbert_features, tfidf_features)

        sbert_features is a dict with keys 'train', 'val', 'test', each containing a dict of arrays.
        tfidf_features is a dict with keys 'train', 'val', 'test', each containing a numpy array.
    """
    os.makedirs(WORKING_DIR, exist_ok=True)

    # File paths
    sbert_path = os.path.join(WORKING_DIR, "sbert_features.npz")
    tfidf_path = os.path.join(WORKING_DIR, "tfidf_features.npz")

    # ---------------------------------------------------------
    # SBERT Logic
    # ---------------------------------------------------------
    sbert_data = None
    if load_cached_data and os.path.exists(sbert_path):
        try:
            # Load npz
            loaded = np.load(sbert_path, allow_pickle=True)
            # Reconstruct dictionary structure
            sbert_data = {
                "train": loaded["train"].item(),
                "val": loaded["val"].item(),
                "test": loaded["test"].item(),
            }
            # print("Loaded SBERT features from cache.")
        except Exception as e:
            # print(f"Failed to load SBERT cache: {e}")
            pass

    if sbert_data is None:
        # print("Generating SBERT features...")
        encoder = SBERTEncoder()
        sbert_data = encoder.process(df_train, df_val, df_test)

        # Save
        np.savez(
            sbert_path,
            train=sbert_data["train"],
            val=sbert_data["val"],
            test=sbert_data["test"],
        )
        # print("Saved SBERT features to cache.")

    # ---------------------------------------------------------
    # TF-IDF Logic
    # ---------------------------------------------------------
    tfidf_data = None
    if load_cached_data and os.path.exists(tfidf_path):
        try:
            loaded = np.load(tfidf_path)
            tfidf_data = {
                "train": loaded["train"],
                "val": loaded["val"],
                "test": loaded["test"],
            }
            # print("Loaded TF-IDF features from cache.")
        except Exception as e:
            # print(f"Failed to load TF-IDF cache: {e}")
            pass

    if tfidf_data is None:
        # print("Generating TF-IDF features...")
        encoder = TFIDFEncoder()
        tfidf_data = encoder.process(df_train, df_val, df_test)

        # Save
        np.savez(
            tfidf_path,
            train=tfidf_data["train"],
            val=tfidf_data["val"],
            test=tfidf_data["test"],
        )
        # print("Saved TF-IDF features to cache.")

    return sbert_data, tfidf_data
