import os
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from library.config import Config
from library.utils import load_json


class SemanticProcessor:
    """
    Handles SBERT-based semantic feature generation, including:
    - Title and Body embeddings
    - User History (Subreddit) embeddings and centroids (Global Persona)
    - Consistency Scalars (Cosine Similarity)
    - Sequence preparation for Dual-Query Attention
    """

    def __init__(self):
        self.model = None
        self.device = Config.DEVICE
        self.max_seq_len = 50  # Maximum history length for sequence modeling

    def _load_model(self):
        """Lazy loads the SBERT model."""
        if self.model is None:
            print(f"Loading SBERT model: {Config.SBERT_MODEL_NAME} on {self.device}")
            self.model = SentenceTransformer(
                Config.SBERT_MODEL_NAME, device=self.device
            )

    def _normalize(self, v):
        """L2 Normalization for cosine similarity calculation."""
        norm = np.linalg.norm(v, axis=1, keepdims=True)
        # Avoid division by zero
        norm[norm == 0] = 1e-10
        return v / norm

    def process_split(
        self,
        df: pd.DataFrame,
        raw_data_path: str,
        split_name: str,
        load_cached_data: bool = True,
    ):
        """
        Generates semantic features for a given dataframe split.

        Args:
            df: DataFrame containing request_ids for the split.
            raw_data_path: Path to the raw JSON file containing full data (text, lists).
            split_name: Name of the split (e.g., 'train', 'val', 'test') for caching.
            load_cached_data: Whether to load from cache if available.

        Returns:
            Dictionary containing numpy arrays of features.
        """
        cache_file = os.path.join(
            Config.IDEA_DIR, f"semantic_features_{split_name}.npz"
        )

        # 1. Check Cache
        if load_cached_data and os.path.exists(cache_file):
            print(
                f"Loading cached semantic features for '{split_name}' from {cache_file}"
            )
            try:
                with np.load(cache_file) as data:
                    # Convert NpzFile to dict to ensure data persists after context close
                    return {key: data[key] for key in data.files}
            except Exception as e:
                print(f"Failed to load cache: {e}. Recomputing...")

        # 2. Load Raw Data and Model
        print(f"Computing semantic features for '{split_name}'...")
        self._load_model()

        raw_data_list = load_json(raw_data_path)
        # Create O(1) lookup map
        id_to_data = {item["request_id"]: item for item in raw_data_list}

        # 3. Extract Data aligned with DF
        titles = []
        bodies = []
        histories = []

        # We iterate over the dataframe to ensure order matches metadata
        for rid in df[Config.ID_COL]:
            item = id_to_data.get(rid, {})
            # Use empty string/list defaults if missing (shouldn't happen in valid data)
            titles.append(item.get("request_title", ""))
            bodies.append(item.get("request_text_edit_aware", ""))
            histories.append(item.get("requester_subreddits_at_request", []))

        # 4. Encode Texts
        # SBERT encode returns numpy array by default if convert_to_numpy=True
        print(f"Encoding {len(titles)} titles...")
        title_embs = self.model.encode(
            titles,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,  # Ensure normalized for cosine sim later
        ).astype(np.float32)

        print(f"Encoding {len(bodies)} bodies...")
        body_embs = self.model.encode(
            bodies,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        # 5. Process Histories (Subreddits)
        print("Processing user histories...")

        # Identify unique subreddits to encode efficiently
        unique_subreddits = set()
        for h in histories:
            unique_subreddits.update(h)
        unique_subreddits = sorted(list(unique_subreddits))  # Sort for determinism

        sub_to_emb = {}
        if unique_subreddits:
            print(f"Encoding {len(unique_subreddits)} unique subreddits...")
            # Encode unique subreddits
            sub_embs_matrix = self.model.encode(
                unique_subreddits,
                batch_size=128,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).astype(np.float32)

            sub_to_emb = {
                sub: emb for sub, emb in zip(unique_subreddits, sub_embs_matrix)
            }

        # Construct Centroids and Sequences
        num_samples = len(df)
        emb_dim = Config.EMBEDDING_DIM

        centroid_embs = np.zeros((num_samples, emb_dim), dtype=np.float32)
        seq_embs = np.zeros((num_samples, self.max_seq_len, emb_dim), dtype=np.float32)
        seq_masks = np.zeros((num_samples, self.max_seq_len), dtype=np.float32)

        for i, h in enumerate(histories):
            if not h:
                continue

            # Gather embeddings for this user's history
            user_sub_embs = []
            for sub in h:
                if sub in sub_to_emb:
                    user_sub_embs.append(sub_to_emb[sub])

            if not user_sub_embs:
                continue

            user_sub_embs = np.array(user_sub_embs, dtype=np.float32)

            # Compute Centroid (Global Persona)
            # We take the mean.
            centroid = np.mean(user_sub_embs, axis=0)
            centroid_embs[i] = centroid

            # Construct Sequence
            # We take up to max_seq_len.
            seq_len = min(len(user_sub_embs), self.max_seq_len)
            seq_embs[i, :seq_len, :] = user_sub_embs[:seq_len]
            seq_masks[i, :seq_len] = 1.0

        # 6. Compute Consistency Scalars
        # Normalize centroids for cosine similarity
        centroid_norm = self._normalize(centroid_embs)
        # Titles and Bodies are already normalized by SBERT encode(normalize_embeddings=True)
        # But let's re-normalize to be absolutely safe against precision errors
        title_norm = self._normalize(title_embs)
        body_norm = self._normalize(body_embs)

        # Dot product for cosine similarity (since normalized)
        # shape: (N, 384) * (N, 384) -> (N, 384) -> sum -> (N,)
        topic_consistency = np.sum(
            title_norm * centroid_norm, axis=1, keepdims=True
        ).astype(np.float32)
        narrative_consistency = np.sum(
            body_norm * centroid_norm, axis=1, keepdims=True
        ).astype(np.float32)

        # 7. Save and Return
        data_dict = {
            "title_embs": title_embs,
            "body_embs": body_embs,
            "centroid_embs": centroid_embs,
            "history_seq_embs": seq_embs,
            "history_mask": seq_masks,
            "topic_consistency": topic_consistency,
            "narrative_consistency": narrative_consistency,
        }

        # Ensure directory exists
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        np.savez(cache_file, **data_dict)
        print(f"Saved semantic features to {cache_file}")

        return data_dict
