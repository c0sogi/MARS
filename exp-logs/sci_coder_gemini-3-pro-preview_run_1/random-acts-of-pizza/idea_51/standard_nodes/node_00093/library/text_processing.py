import os
import ast
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from library.config import WORKING_DIR, SBERT_MODEL_NAME, EMBEDDING_DIM, RANDOM_STATE
from library.utils import save_to_cache, load_from_cache, seed_everything

# Constants
MAX_HISTORY_LEN = 50
CACHE_FILENAME = "text_features.npz"


class TextProcessor:
    """
    Wrapper for SentenceTransformer to handle model loading and encoding.
    """

    def __init__(self, model_name=SBERT_MODEL_NAME, device=None):
        self.model_name = model_name
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.model = None

    def load_model(self):
        """Lazy loading of the model to save resources if cache is hit."""
        if self.model is None:
            # print(f"Loading SBERT model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name, device=self.device)

    def encode(self, texts, batch_size=32, show_progress_bar=False):
        """
        Encodes a list of texts into embeddings.
        Handles NaN/None by converting to empty strings.
        """
        self.load_model()
        # Ensure all inputs are strings
        cleaned_texts = [str(t) if pd.notna(t) else "" for t in texts]

        embeddings = self.model.encode(
            cleaned_texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
        )
        return embeddings


def parse_subreddit_list(list_str):
    """
    Parses the string representation of the subreddit list from the CSV.
    """
    try:
        if isinstance(list_str, list):
            return list_str
        if pd.isna(list_str) or list_str == "" or list_str == "[]":
            return []
        return ast.literal_eval(list_str)
    except (ValueError, SyntaxError):
        return []


def process_text_data(train_df, val_df, test_df, load_cached_data=True):
    """
    Main function to generate text-based features.

    Outputs:
        - SBERT embeddings for Title and Body.
        - User History Centroids (Mean of subreddit embeddings).
        - User History Sequences (for MLP attention).
        - Consistency Scalars (Cosine sim between request and history).
    """
    seed_everything(RANDOM_STATE)

    # 1. Check Cache
    cached_data = load_from_cache(CACHE_FILENAME, directory=WORKING_DIR)
    if load_cached_data and cached_data is not None:
        print(
            f"Loaded text features from cache: {os.path.join(WORKING_DIR, CACHE_FILENAME)}"
        )
        return dict(cached_data)

    print("Computing text features from scratch...")

    processor = TextProcessor()

    # 2. Extract Text Columns
    train_titles = train_df["request_title"].fillna("").tolist()
    val_titles = val_df["request_title"].fillna("").tolist()
    test_titles = test_df["request_title"].fillna("").tolist()

    train_bodies = train_df["request_text_edit_aware"].fillna("").tolist()
    val_bodies = val_df["request_text_edit_aware"].fillna("").tolist()
    test_bodies = test_df["request_text_edit_aware"].fillna("").tolist()

    # 3. Encode Titles and Bodies
    # Concatenate to batch process efficiently
    all_titles = train_titles + val_titles + test_titles
    all_bodies = train_bodies + val_bodies + test_bodies

    print(f"Encoding {len(all_titles)} titles...")
    all_title_embs = processor.encode(all_titles, batch_size=64)

    print(f"Encoding {len(all_bodies)} bodies...")
    all_body_embs = processor.encode(all_bodies, batch_size=64)

    # Split back into train/val/test
    n_train = len(train_df)
    n_val = len(val_df)

    train_title_emb = all_title_embs[:n_train]
    val_title_emb = all_title_embs[n_train : n_train + n_val]
    test_title_emb = all_title_embs[n_train + n_val :]

    train_body_emb = all_body_embs[:n_train]
    val_body_emb = all_body_embs[n_train : n_train + n_val]
    test_body_emb = all_body_embs[n_train + n_val :]

    # 4. Process User History (Subreddits)
    print("Processing user history...")

    train_subs = (
        train_df["requester_subreddits_at_request"].apply(parse_subreddit_list).tolist()
    )
    val_subs = (
        val_df["requester_subreddits_at_request"].apply(parse_subreddit_list).tolist()
    )
    test_subs = (
        test_df["requester_subreddits_at_request"].apply(parse_subreddit_list).tolist()
    )

    # Identify unique subreddits across all datasets to encode once
    unique_subreddits = set()
    for subs in train_subs + val_subs + test_subs:
        unique_subreddits.update(subs)

    unique_subreddits = sorted(list(unique_subreddits))
    sub_to_idx = {sub: i for i, sub in enumerate(unique_subreddits)}

    print(f"Encoding {len(unique_subreddits)} unique subreddits...")
    if unique_subreddits:
        sub_embeddings = processor.encode(unique_subreddits, batch_size=128)
    else:
        sub_embeddings = np.zeros((0, EMBEDDING_DIM))

    # Helper function to generate history features for a split
    def process_history_split(subs_list, title_embs, body_embs):
        n_samples = len(subs_list)

        # Initialize outputs
        centroids = np.zeros((n_samples, EMBEDDING_DIM), dtype=np.float32)
        sequences = np.zeros(
            (n_samples, MAX_HISTORY_LEN, EMBEDDING_DIM), dtype=np.float32
        )
        masks = np.zeros(
            (n_samples, MAX_HISTORY_LEN), dtype=np.float32
        )  # 1 = valid, 0 = padding

        sim_title = np.zeros((n_samples, 1), dtype=np.float32)
        sim_body = np.zeros((n_samples, 1), dtype=np.float32)

        for i, subs in enumerate(subs_list):
            if not subs:
                # No history: Centroid is 0, Sim is 0 (handled by initialization)
                continue

            # Get embeddings for subreddits
            # Limit to MAX_HISTORY_LEN
            current_subs = subs[:MAX_HISTORY_LEN]
            indices = [sub_to_idx[s] for s in current_subs if s in sub_to_idx]

            if not indices:
                continue

            user_sub_embs = sub_embeddings[indices]

            # A. Centroid
            centroid = np.mean(user_sub_embs, axis=0)
            centroids[i] = centroid

            # B. Sequence & Mask
            seq_len = len(user_sub_embs)
            sequences[i, :seq_len, :] = user_sub_embs
            masks[i, :seq_len] = 1.0

            # C. Consistency Scalars (Cosine Similarity)
            # Reshape to (1, D) for sklearn
            t_vec = title_embs[i].reshape(1, -1)
            b_vec = body_embs[i].reshape(1, -1)
            c_vec = centroid.reshape(1, -1)

            # Check for zero vectors to avoid division by zero warnings/errors
            # (Though sklearn usually handles 0 vectors by returning 0)
            if np.linalg.norm(c_vec) > 1e-9:
                sim_title[i] = cosine_similarity(t_vec, c_vec)[0][0]
                sim_body[i] = cosine_similarity(b_vec, c_vec)[0][0]

        return centroids, sequences, masks, sim_title, sim_body

    # Generate features for each split
    print("Generating history features (Train)...")
    train_res = process_history_split(train_subs, train_title_emb, train_body_emb)

    print("Generating history features (Val)...")
    val_res = process_history_split(val_subs, val_title_emb, val_body_emb)

    print("Generating history features (Test)...")
    test_res = process_history_split(test_subs, test_title_emb, test_body_emb)

    # 5. Pack Results
    data_dict = {
        # Train
        "train_title_emb": train_title_emb,
        "train_body_emb": train_body_emb,
        "train_centroid": train_res[0],
        "train_history_seq": train_res[1],
        "train_history_mask": train_res[2],
        "train_sim_title": train_res[3],
        "train_sim_body": train_res[4],
        # Val
        "val_title_emb": val_title_emb,
        "val_body_emb": val_body_emb,
        "val_centroid": val_res[0],
        "val_history_seq": val_res[1],
        "val_history_mask": val_res[2],
        "val_sim_title": val_res[3],
        "val_sim_body": val_res[4],
        # Test
        "test_title_emb": test_title_emb,
        "test_body_emb": test_body_emb,
        "test_centroid": test_res[0],
        "test_history_seq": test_res[1],
        "test_history_mask": test_res[2],
        "test_sim_title": test_res[3],
        "test_sim_body": test_res[4],
    }

    # 6. Save to Cache
    save_to_cache(data_dict, CACHE_FILENAME, directory=WORKING_DIR)

    return data_dict
