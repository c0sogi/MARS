import os
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from library.config import (
    WORKING_DIR,
    TEXT_CONFIG,
    SEED,
    CACHE_SBERT_EMBEDDINGS,
    TEXT_COLS,
    TOP_K_CONFIG,
)

# Set seeds for reproducibility
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


class SBERTEncoder:
    def __init__(self, model_name=None, device=None):
        self.model_name = model_name or TEXT_CONFIG["sbert_model_name"]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializing SBERT model: {self.model_name} on {self.device}")
        self.model = SentenceTransformer(self.model_name, device=self.device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts, batch_size=32, show_progress_bar=False):
        """
        Encodes a list of texts into embeddings.
        """
        # Ensure texts are strings and handle NaNs
        cleaned_texts = [str(t) if pd.notnull(t) else "" for t in texts]
        embeddings = self.model.encode(
            cleaned_texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
        )
        return embeddings

    def encode_history(self, history_lists, max_seq_len=50):
        """
        Encodes lists of subreddits into centroids and padded sequences.
        Optimized to encode unique subreddits only once.
        """
        # 1. Identify unique subreddits
        unique_subs = set()
        for sub_list in history_lists:
            unique_subs.update(sub_list)

        unique_subs_list = sorted(list(unique_subs))
        sub_to_idx = {sub: i for i, sub in enumerate(unique_subs_list)}

        print(f"Encoding {len(unique_subs_list)} unique subreddits...")
        sub_embeddings = self.encode(unique_subs_list, batch_size=64)

        # 2. Map back to users
        num_samples = len(history_lists)
        centroids = np.zeros((num_samples, self.embedding_dim), dtype=np.float32)
        sequences = np.zeros(
            (num_samples, max_seq_len, self.embedding_dim), dtype=np.float32
        )

        for i, sub_list in enumerate(history_lists):
            if not sub_list:
                continue

            # Get indices for this user's subreddits
            indices = [sub_to_idx[sub] for sub in sub_list if sub in sub_to_idx]

            if not indices:
                continue

            # Get embeddings
            user_sub_embs = sub_embeddings[indices]

            # Compute Centroid
            centroids[i] = np.mean(user_sub_embs, axis=0)

            # Create Sequence (Truncate or Pad)
            # We take the most recent ones if we assume the list is chronological,
            # but lists are usually arbitrary. We'll take the first K.
            seq_len = min(len(user_sub_embs), max_seq_len)
            sequences[i, :seq_len, :] = user_sub_embs[:seq_len]

        return centroids, sequences


def generate_sbert_embeddings(train_df, val_df, test_df, load_cached_data=True):
    """
    Generates or loads SBERT embeddings for Title, Body, and History.
    Returns a dictionary containing numpy arrays.
    """
    os.makedirs(WORKING_DIR, exist_ok=True)

    if load_cached_data and os.path.exists(CACHE_SBERT_EMBEDDINGS):
        print(f"Loading SBERT embeddings from {CACHE_SBERT_EMBEDDINGS}...")
        cached_data = np.load(CACHE_SBERT_EMBEDDINGS)
        # Validate cache dimensions (Cite debug_lesson_1)
        if (
            len(cached_data["train_title"]) == len(train_df)
            and len(cached_data["val_title"]) == len(val_df)
            and len(cached_data["test_title"]) == len(test_df)
        ):
            return cached_data
        else:
            print("Cache dimension mismatch detected. Recomputing SBERT embeddings...")

    print("Generating SBERT embeddings from scratch...")
    encoder = SBERTEncoder()

    # Prepare data containers
    data_splits = {"train": train_df, "val": val_df, "test": test_df}

    output_data = {}

    for split_name, df in data_splits.items():
        print(f"Processing {split_name} split...")

        # 1. Title Embeddings
        print(f"  - Encoding Titles...")
        output_data[f"{split_name}_title"] = encoder.encode(
            df["request_title"].tolist()
        )

        # 2. Body Embeddings
        print(f"  - Encoding Bodies...")
        output_data[f"{split_name}_body"] = encoder.encode(
            df["request_text_edit_aware"].tolist()
        )

        # 3. History Embeddings (Centroid + Sequence)
        print(f"  - Encoding History...")
        # Ensure the column exists and is a list
        hist_col = "requester_subreddits_at_request"
        if hist_col in df.columns:
            # Handle potential string representation if not parsed yet (safety check)
            history_data = df[hist_col].tolist()
            # If data loader didn't parse, we might have strings.
            # Assuming data_loader works as expected, these are lists.
            centroids, sequences = encoder.encode_history(
                history_data, max_seq_len=TOP_K_CONFIG["k"]
            )
            output_data[f"{split_name}_hist_centroid"] = centroids
            output_data[f"{split_name}_hist_seq"] = sequences
        else:
            print(
                f"Warning: {hist_col} not found in {split_name} df. Filling with zeros."
            )
            n = len(df)
            dim = encoder.embedding_dim
            output_data[f"{split_name}_hist_centroid"] = np.zeros(
                (n, dim), dtype=np.float32
            )
            output_data[f"{split_name}_hist_seq"] = np.zeros(
                (n, TOP_K_CONFIG["k"], dim), dtype=np.float32
            )

    # Save to cache
    print(f"Saving SBERT embeddings to {CACHE_SBERT_EMBEDDINGS}...")
    np.savez_compressed(CACHE_SBERT_EMBEDDINGS, **output_data)

    return output_data


def generate_tfidf_features(train_df, val_df, test_df, load_cached_data=True):
    """
    Generates or loads TF-IDF features for concatenated Title + Body.
    Returns a dictionary containing numpy arrays.
    """
    cache_path = os.path.join(WORKING_DIR, "tfidf_features.npz")

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading TF-IDF features from {cache_path}...")
        cached_data = np.load(cache_path)
        # Validate cache dimensions (Cite debug_lesson_1)
        if (
            cached_data["train_tfidf"].shape[0] == len(train_df)
            and cached_data["val_tfidf"].shape[0] == len(val_df)
            and cached_data["test_tfidf"].shape[0] == len(test_df)
        ):
            return cached_data
        else:
            print("Cache dimension mismatch detected. Recomputing TF-IDF features...")

    print("Generating TF-IDF features from scratch...")

    # Text Preprocessing Helper
    def get_text(df):
        title = df["request_title"].fillna("").astype(str)
        body = df["request_text_edit_aware"].fillna("").astype(str)
        return title + " " + body

    train_text = get_text(train_df)
    val_text = get_text(val_df)
    test_text = get_text(test_df)

    # Vectorization
    vectorizer = TfidfVectorizer(
        max_features=TEXT_CONFIG["tfidf_max_features"],
        ngram_range=TEXT_CONFIG["tfidf_ngram_range"],
        stop_words="english",
        sublinear_tf=True,  # Apply log scaling to term frequency
    )

    print("Fitting TF-IDF on training data...")
    train_tfidf = vectorizer.fit_transform(train_text)

    print("Transforming validation and test data...")
    val_tfidf = vectorizer.transform(val_text)
    test_tfidf = vectorizer.transform(test_text)

    # Convert to dense arrays (feasible given dataset size ~4000 samples)
    output_data = {
        "train_tfidf": train_tfidf.toarray().astype(np.float32),
        "val_tfidf": val_tfidf.toarray().astype(np.float32),
        "test_tfidf": test_tfidf.toarray().astype(np.float32),
    }

    print(f"Saving TF-IDF features to {cache_path}...")
    np.savez_compressed(cache_path, **output_data)

    return output_data


def compute_consistency_scores(sbert_data):
    """
    Computes cosine similarity between request (Title/Body) and User History Centroid.

    Args:
        sbert_data (dict/np.lib.npyio.NpzFile): Output from generate_sbert_embeddings

    Returns:
        dict: Dictionary containing consistency scores for train/val/test
              keys: {split}_title_consistency, {split}_body_consistency
    """
    print("Computing Consistency Scores...")
    scores = {}

    splits = ["train", "val", "test"]

    for split in splits:
        # Retrieve embeddings
        title_emb = sbert_data[f"{split}_title"]
        body_emb = sbert_data[f"{split}_body"]
        hist_centroid = sbert_data[f"{split}_hist_centroid"]

        # Compute Cosine Similarity
        # sklearn cosine_similarity returns matrix (N, N), we want diagonal (N,)
        # Optimization: (A * B).sum(axis=1) / (norm(A)*norm(B))
        # Since SBERT embeddings are typically normalized, dot product is cosine similarity.
        # But let's use sklearn's utility but efficiently.

        # Actually, let's just do dot product and explicit normalization to be safe
        def cosine_sim_paired(a, b):
            # Normalize
            norm_a = np.linalg.norm(a, axis=1, keepdims=True)
            norm_b = np.linalg.norm(b, axis=1, keepdims=True)

            # Avoid division by zero
            norm_a[norm_a == 0] = 1e-9
            norm_b[norm_b == 0] = 1e-9

            return np.sum(a * b, axis=1) / (norm_a * norm_b).flatten()

        scores[f"{split}_title_consistency"] = cosine_sim_paired(
            title_emb, hist_centroid
        )
        scores[f"{split}_body_consistency"] = cosine_sim_paired(body_emb, hist_centroid)

    return scores
