import os
import numpy as np
import joblib
from sklearn.decomposition import NMF
from library.config import Config
from library.text_utils import SBERTEmbedder


class TopicAligner:
    """
    Wraps NMF Topic Modeling for SBERT embeddings.
    Handles non-negative transformation and persistence.
    """

    def __init__(self, n_topics=Config.NUM_TOPICS, random_state=Config.RANDOM_STATE):
        self.n_topics = n_topics
        self.random_state = random_state
        self.model = None
        self.min_val = 0.0  # Shift value for NMF compliance

    def fit(self, embeddings):
        """
        Fits NMF model on embeddings.
        Args:
            embeddings (np.ndarray): Shape (N, Dim)
        """
        # SBERT embeddings are [-1, 1]. NMF requires >= 0.
        # We shift by subtracting the global minimum (plus epsilon if needed)
        self.min_val = np.min(embeddings)
        if self.min_val < 0:
            embeddings_shifted = embeddings - self.min_val
        else:
            self.min_val = 0.0
            embeddings_shifted = embeddings

        print(f"Fitting NMF with {self.n_topics} topics...")
        self.model = NMF(
            n_components=self.n_topics,
            init="nndsvd",
            random_state=self.random_state,
            max_iter=500,
            tol=1e-4,
        )
        self.model.fit(embeddings_shifted)

        # Save model and shift value
        self._save()
        return self

    def transform(self, embeddings):
        """
        Transforms embeddings to topic distribution.
        """
        if self.model is None:
            if not self._load():
                raise RuntimeError(
                    "TopicAligner is not fitted and no cached model found."
                )

        # Apply same shift as fit
        embeddings_shifted = embeddings - self.min_val
        # Clip to 0 to ensure non-negative (handles outliers in test data)
        embeddings_shifted = np.clip(embeddings_shifted, 0, None)

        return self.model.transform(embeddings_shifted)

    def _save(self):
        """Saves model to disk."""
        payload = {"model": self.model, "min_val": self.min_val}
        joblib.dump(payload, Config.CACHE_TOPIC_MODEL)
        print(f"Topic model saved to {Config.CACHE_TOPIC_MODEL}")

    def _load(self):
        """Loads model from disk."""
        if os.path.exists(Config.CACHE_TOPIC_MODEL):
            try:
                payload = joblib.load(Config.CACHE_TOPIC_MODEL)
                self.model = payload["model"]
                self.min_val = payload["min_val"]
                print(f"Topic model loaded from {Config.CACHE_TOPIC_MODEL}")
                return True
            except Exception as e:
                print(f"Failed to load topic model: {e}")
                return False
        return False


def train_topic_model_if_needed(train_df, embedder, load_cached_data=True):
    """
    Ensures the TopicAligner is fitted.
    If cached model exists, loads it.
    Otherwise, computes embeddings for training data and fits NMF.
    """
    aligner = TopicAligner()

    # Try to load first
    if load_cached_data and aligner._load():
        return aligner

    print("Training Topic Model from scratch...")
    # We use Request embeddings to define the topic space.
    req_embs = embedder.encode_requests(
        train_df, "train", load_cached_data=load_cached_data
    )

    aligner.fit(req_embs)
    return aligner


def compute_topic_alignment(df, split_name, embedder, aligner, load_cached_data=True):
    """
    Generates topic distributions and alignment scores.
    Caches results to .npy.

    Returns:
        dict: {
            "request_topic_dist": (N, K),
            "history_topic_dist": (N, K),
            "alignment_score": (N,)
        }
    """
    cache_file = os.path.join(Config.WORKING_DIR, f"topic_features_{split_name}.npy")

    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading cached topic features for {split_name}...")
        return np.load(cache_file, allow_pickle=True).item()

    print(f"Computing topic alignment for {split_name}...")

    # 1. Get Embeddings
    req_embs = embedder.encode_requests(
        df, split_name, load_cached_data=load_cached_data
    )
    hist_embs_3d = embedder.encode_history(
        df, split_name, load_cached_data=load_cached_data
    )

    # 2. Compute Topic Distributions
    # Requests
    req_topic_dist = aligner.transform(req_embs)

    # History (N, MaxLen, Dim) -> Flatten -> Transform -> Reshape -> Average
    N, MaxLen, Dim = hist_embs_3d.shape
    hist_flat = hist_embs_3d.reshape(-1, Dim)

    # Filter out padding (zero vectors)
    # SBERT vectors are normalized, so norm ~ 0 implies padding
    norms = np.linalg.norm(hist_flat, axis=1)
    mask = norms > 1e-6

    hist_topic_dist_flat = np.zeros((N * MaxLen, aligner.n_topics), dtype=np.float32)

    if np.any(mask):
        valid_embs = hist_flat[mask]
        valid_topics = aligner.transform(valid_embs)
        hist_topic_dist_flat[mask] = valid_topics

    hist_topic_dist_3d = hist_topic_dist_flat.reshape(N, MaxLen, aligner.n_topics)

    # Average History Distribution per User
    # Count valid subreddits per user
    valid_counts = np.sum(mask.reshape(N, MaxLen), axis=1)
    valid_counts = np.maximum(valid_counts, 1)  # Avoid div by zero

    hist_sum = np.sum(hist_topic_dist_3d, axis=1)
    hist_topic_avg = hist_sum / valid_counts[:, None]

    # 3. Compute Alignment Score (Cosine Similarity)
    # Cosine Sim = (A . B) / (|A| * |B|)

    dot_prod = np.sum(req_topic_dist * hist_topic_avg, axis=1)

    norm_req = np.linalg.norm(req_topic_dist, axis=1)
    norm_hist = np.linalg.norm(hist_topic_avg, axis=1)

    # Avoid div by zero
    norm_req = np.maximum(norm_req, 1e-9)
    norm_hist = np.maximum(norm_hist, 1e-9)

    alignment_score = dot_prod / (norm_req * norm_hist)

    features = {
        "request_topic_dist": req_topic_dist,
        "history_topic_dist": hist_topic_avg,
        "alignment_score": alignment_score,
    }

    # Save to cache
    np.save(cache_file, features)

    return features
