import os
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from library.config import (
    WORKING_DIR,
    SBERT_MODEL_NAME,
    MAX_TEXT_LENGTH,
    MAX_HISTORY_LENGTH,
    DEVICE,
    EMBEDDING_DIM,
)


class TextEncoder:
    def __init__(self):
        """
        Initializes the TextEncoder with the SBERT model.
        """
        print(f"Initializing SBERT model: {SBERT_MODEL_NAME} on {DEVICE}...")
        self.model = SentenceTransformer(SBERT_MODEL_NAME, device=DEVICE)
        # Ensure working directory exists
        os.makedirs(WORKING_DIR, exist_ok=True)

    def _get_cache_path(self, dataset_name):
        """
        Returns the path for the cached features file.
        """
        return os.path.join(WORKING_DIR, f"sbert_features_{dataset_name}.npz")

    def encode(self, df, dataset_name, load_cached_data=True):
        """
        Generates or loads embeddings for title, body, and user history.

        Args:
            df (pd.DataFrame): The dataframe containing the data.
            dataset_name (str): Name of the dataset (e.g., 'train', 'val', 'test') for caching.
            load_cached_data (bool): Whether to attempt loading from cache.

        Returns:
            dict: A dictionary containing numpy arrays:
                  - 'title_emb': (N, 384)
                  - 'body_emb': (N, 384)
                  - 'hist_seq': (N, MAX_HISTORY_LENGTH, 384)
                  - 'hist_mask': (N, MAX_HISTORY_LENGTH)
                  - 'hist_centroid': (N, 384)
        """
        cache_path = self._get_cache_path(dataset_name)

        # 1. Try Loading from Cache
        if load_cached_data and os.path.exists(cache_path):
            print(
                f"Loading cached SBERT features for {dataset_name} from {cache_path}..."
            )
            try:
                data = np.load(cache_path)
                return {
                    "title_emb": data["title_emb"],
                    "body_emb": data["body_emb"],
                    "hist_seq": data["hist_seq"],
                    "hist_mask": data["hist_mask"],
                    "hist_centroid": data["hist_centroid"],
                }
            except Exception as e:
                print(f"Failed to load cache: {e}. Recomputing...")

        print(f"Computing SBERT features for {dataset_name}...")

        # 2. Prepare Text Data
        # Fill NaNs with empty string
        titles = df["request_title"].fillna("").astype(str).tolist()
        # Use edit_aware text to avoid leakage, fallback to request_text if missing, then empty
        if "request_text_edit_aware" in df.columns:
            bodies = df["request_text_edit_aware"].fillna("").astype(str).tolist()
        else:
            bodies = df["request_text"].fillna("").astype(str).tolist()

        # 3. Encode Title and Body
        # show_progress_bar=False to reduce clutter
        title_emb = self.model.encode(
            titles, batch_size=64, show_progress_bar=False, convert_to_numpy=True
        )
        body_emb = self.model.encode(
            bodies, batch_size=64, show_progress_bar=False, convert_to_numpy=True
        )

        # 4. Process User History (Subreddits)
        hist_seq, hist_mask, hist_centroid = self._process_history(df)

        # 5. Save to Cache
        print(f"Saving SBERT features for {dataset_name} to {cache_path}...")
        np.savez_compressed(
            cache_path,
            title_emb=title_emb,
            body_emb=body_emb,
            hist_seq=hist_seq,
            hist_mask=hist_mask,
            hist_centroid=hist_centroid,
        )

        return {
            "title_emb": title_emb,
            "body_emb": body_emb,
            "hist_seq": hist_seq,
            "hist_mask": hist_mask,
            "hist_centroid": hist_centroid,
        }

    def _process_history(self, df):
        """
        Encodes user subreddit history efficiently.
        """
        # Extract the column containing lists of subreddits
        # Ensure it's a list (it should be parsed by data_loader already)
        history_col = df["requester_subreddits_at_request"]

        # 1. Identify all unique subreddits to encode them once
        unique_subreddits = set()
        for sub_list in history_col:
            if isinstance(sub_list, list):
                unique_subreddits.update(sub_list)
            elif isinstance(sub_list, np.ndarray):
                unique_subreddits.update(sub_list.tolist())

        unique_subreddits = list(unique_subreddits)

        # Map subreddit name -> embedding
        if not unique_subreddits:
            print("Warning: No subreddits found in history column.")
            sub_emb_map = {}
        else:
            # Encode unique subreddits
            # We treat subreddits as short text descriptions
            sub_embeddings = self.model.encode(
                unique_subreddits,
                batch_size=128,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            sub_emb_map = {
                sub: emb for sub, emb in zip(unique_subreddits, sub_embeddings)
            }

        # 2. Construct Sequences and Centroids for each user
        num_samples = len(df)
        hist_seq = np.zeros(
            (num_samples, MAX_HISTORY_LENGTH, EMBEDDING_DIM), dtype=np.float32
        )
        hist_mask = np.zeros((num_samples, MAX_HISTORY_LENGTH), dtype=np.float32)
        hist_centroid = np.zeros((num_samples, EMBEDDING_DIM), dtype=np.float32)

        for i, sub_list in enumerate(history_col):
            if not isinstance(sub_list, (list, np.ndarray)) or len(sub_list) == 0:
                # No history: sequence is all zeros.
                # Cite debug_lesson_8: Ensure at least one valid token in attention mask
                # to prevent NaN in Softmax (0/0). We unmask the first padding token (zero vector).
                hist_mask[i, 0] = 1.0
                continue

            # Get embeddings for this user's subreddits
            # Filter out any that might not be in map (shouldn't happen if logic is correct)
            user_sub_embs = [sub_emb_map[s] for s in sub_list if s in sub_emb_map]

            if not user_sub_embs:
                # Cite debug_lesson_8: Handle case where subreddits exist but have no embeddings
                hist_mask[i, 0] = 1.0
                continue

            user_sub_embs = np.array(user_sub_embs, dtype=np.float32)

            # --- Centroid ---
            # Mean of all subreddit embeddings
            hist_centroid[i] = np.mean(user_sub_embs, axis=0)

            # --- Sequence ---
            # Truncate to MAX_HISTORY_LENGTH
            seq_len = min(len(user_sub_embs), MAX_HISTORY_LENGTH)

            # Fill the sequence array
            hist_seq[i, :seq_len, :] = user_sub_embs[:seq_len]

            # Set the mask (1 for valid entries, 0 for padding)
            hist_mask[i, :seq_len] = 1.0

        return hist_seq, hist_mask, hist_centroid
