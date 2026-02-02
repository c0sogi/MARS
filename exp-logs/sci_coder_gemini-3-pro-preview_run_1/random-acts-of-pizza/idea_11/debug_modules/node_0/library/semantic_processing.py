import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from library.config import Config
from library.utils import set_seed


class SemanticEngine:
    """
    Handles semantic feature generation for the Pizza Request Prediction task.
    Generates:
    1. SBERT embeddings for request text.
    2. TF-IDF vectors for request text.
    3. Discrete Topic Ratios based on subreddit history clustering.
    4. Consistency Scores (Request vs. History similarity).
    5. Padded History Tensors for Attention-based models.
    """

    def __init__(self):
        self.config = Config
        self.cache_path = os.path.join(self.config.WORKING_DIR, "semantic_features.npz")
        set_seed(self.config.SEED)

    def process(self, train_df, val_df, test_df, load_cached_data=True):
        """
        Main processing function. Checks cache, otherwise generates features and saves them.

        Args:
            train_df, val_df, test_df (pd.DataFrame): Input dataframes.
            load_cached_data (bool): Whether to load from cache if available.

        Returns:
            dict: A dictionary containing 'train', 'val', 'test' sub-dictionaries with feature arrays.
        """
        # 1. Check Cache
        if load_cached_data and os.path.exists(self.cache_path):
            print(f"Loading semantic features from {self.cache_path}...")
            return self._load_from_cache()

        print("Generating semantic features from scratch...")

        # 2. Initialize Models
        # Note: We use the device specified in Config, but SentenceTransformer handles this internally usually.
        # We force CPU if CUDA not available, though ST usually auto-detects.
        print(f"Loading SBERT model: {self.config.SBERT_MODEL}...")
        sbert_model = SentenceTransformer(
            self.config.SBERT_MODEL, device=self.config.DEVICE
        )

        # 3. Text Embeddings (Request Text)
        print("Generating Request Text SBERT embeddings...")
        train_sbert = self._embed_text(sbert_model, train_df)
        val_sbert = self._embed_text(sbert_model, val_df)
        test_sbert = self._embed_text(sbert_model, test_df)

        # 4. Subreddit/History Analysis
        print("Processing Subreddit History (Embeddings & Topics)...")
        # Gather all unique subreddits to embed in batch
        all_subreddits = set()
        for df in [train_df, val_df, test_df]:
            for sub_list in df[self.config.SUBREDDIT_COL]:
                all_subreddits.update(sub_list)

        all_subreddits = sorted(list(all_subreddits))
        if not all_subreddits:
            # Handle edge case where no subreddits exist in data
            print("Warning: No subreddits found in history.")
            sub_emb_map = {}
        else:
            sub_embeddings = sbert_model.encode(
                all_subreddits,
                batch_size=64,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            sub_emb_map = {sub: emb for sub, emb in zip(all_subreddits, sub_embeddings)}

        # Fit K-Means on TRAIN subreddits only
        train_subreddits = set()
        for sub_list in train_df[self.config.SUBREDDIT_COL]:
            train_subreddits.update(sub_list)
        train_subreddits = sorted(list(train_subreddits))

        # Filter embeddings for clustering
        train_sub_embs = [sub_emb_map[s] for s in train_subreddits if s in sub_emb_map]

        if len(train_sub_embs) >= self.config.NUM_TOPICS:
            kmeans = KMeans(
                n_clusters=self.config.NUM_TOPICS,
                random_state=self.config.SEED,
                n_init=10,
            )
            kmeans.fit(train_sub_embs)
            # Create a map from subreddit -> cluster ID
            # We predict for ALL subreddits (including those only in test)
            all_sub_embs_matrix = np.array([sub_emb_map[s] for s in all_subreddits])
            all_clusters = kmeans.predict(all_sub_embs_matrix)
            sub_cluster_map = {
                sub: clust for sub, clust in zip(all_subreddits, all_clusters)
            }
        else:
            print(
                "Warning: Not enough unique subreddits to fit K-Means. Assigning all to topic 0."
            )
            sub_cluster_map = {sub: 0 for sub in all_subreddits}

        # Generate History Features (Topics, Consistency, Tensor)
        train_hist_feats = self._process_history(
            train_df, train_sbert, sub_emb_map, sub_cluster_map
        )
        val_hist_feats = self._process_history(
            val_df, val_sbert, sub_emb_map, sub_cluster_map
        )
        test_hist_feats = self._process_history(
            test_df, test_sbert, sub_emb_map, sub_cluster_map
        )

        # 5. TF-IDF Generation
        print("Generating TF-IDF vectors...")
        tfidf = TfidfVectorizer(
            max_features=self.config.TFIDF_MAX_FEATURES,
            ngram_range=self.config.TFIDF_NGRAM_RANGE,
            stop_words="english",
        )
        # Fit on Train, Transform All
        train_texts = train_df[self.config.TEXT_COL].fillna("").astype(str)
        val_texts = val_df[self.config.TEXT_COL].fillna("").astype(str)
        test_texts = test_df[self.config.TEXT_COL].fillna("").astype(str)

        train_tfidf = tfidf.fit_transform(train_texts).toarray().astype(np.float32)
        val_tfidf = tfidf.transform(val_texts).toarray().astype(np.float32)
        test_tfidf = tfidf.transform(test_texts).toarray().astype(np.float32)

        # 6. Save to Cache
        print(f"Saving features to {self.cache_path}...")
        np.savez(
            self.cache_path,
            # Train
            train_sbert=train_sbert,
            train_tfidf=train_tfidf,
            train_topic_ratios=train_hist_feats["topic_ratios"],
            train_consistency=train_hist_feats["consistency"],
            train_history_emb=train_hist_feats["history_emb"],
            train_history_mask=train_hist_feats["history_mask"],
            # Val
            val_sbert=val_sbert,
            val_tfidf=val_tfidf,
            val_topic_ratios=val_hist_feats["topic_ratios"],
            val_consistency=val_hist_feats["consistency"],
            val_history_emb=val_hist_feats["history_emb"],
            val_history_mask=val_hist_feats["history_mask"],
            # Test
            test_sbert=test_sbert,
            test_tfidf=test_tfidf,
            test_topic_ratios=test_hist_feats["topic_ratios"],
            test_consistency=test_hist_feats["consistency"],
            test_history_emb=test_hist_feats["history_emb"],
            test_history_mask=test_hist_feats["history_mask"],
        )

        # 7. Return Constructed Dictionary
        return self._construct_output_dict(
            train_sbert,
            train_tfidf,
            train_hist_feats,
            val_sbert,
            val_tfidf,
            val_hist_feats,
            test_sbert,
            test_tfidf,
            test_hist_feats,
        )

    def _embed_text(self, model, df):
        """Helper to embed request text."""
        texts = df[self.config.TEXT_COL].fillna("").astype(str).tolist()
        embeddings = model.encode(
            texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True
        )
        return embeddings.astype(np.float32)

    def _process_history(self, df, request_embs, sub_emb_map, sub_cluster_map):
        """
        Generates history-based features: Topic Ratios, Consistency Score, and Padded Tensor.
        """
        n_samples = len(df)
        max_len = self.config.MAX_HISTORY_LEN
        emb_dim = self.config.EMBEDDING_DIM
        num_topics = self.config.NUM_TOPICS

        # Outputs
        topic_ratios = np.zeros((n_samples, num_topics), dtype=np.float32)
        consistency = np.zeros((n_samples,), dtype=np.float32)
        history_emb = np.zeros((n_samples, max_len, emb_dim), dtype=np.float32)
        history_mask = np.zeros((n_samples, max_len), dtype=np.float32)

        sub_lists = df[self.config.SUBREDDIT_COL].tolist()

        for i, subs in enumerate(sub_lists):
            # Filter subs that have embeddings (should be all, but safety check)
            valid_subs = [s for s in subs if s in sub_emb_map]

            if not valid_subs:
                # No history: everything remains zero
                continue

            # 1. Topic Ratios
            # Count clusters
            clusters = [sub_cluster_map[s] for s in valid_subs]
            for c in clusters:
                topic_ratios[i, c] += 1
            # Normalize
            if len(clusters) > 0:
                topic_ratios[i] /= len(clusters)

            # 2. Consistency Score
            # Mean history embedding
            hist_vectors = np.array([sub_emb_map[s] for s in valid_subs])
            mean_hist_emb = np.mean(hist_vectors, axis=0)

            # Cosine similarity with request embedding
            # Reshape for sklearn: (1, dim)
            req_vec = request_embs[i].reshape(1, -1)
            hist_vec = mean_hist_emb.reshape(1, -1)

            sim = cosine_similarity(req_vec, hist_vec)[0][0]
            consistency[i] = sim

            # 3. History Tensor (Truncate/Pad)
            # Take last N subreddits (assuming list is roughly chronological or random)
            # We take up to max_len
            trunc_subs = valid_subs[:max_len]
            n_items = len(trunc_subs)

            for t, s in enumerate(trunc_subs):
                history_emb[i, t, :] = sub_emb_map[s]
                history_mask[i, t] = 1.0

        return {
            "topic_ratios": topic_ratios,
            "consistency": consistency,
            "history_emb": history_emb,
            "history_mask": history_mask,
        }

    def _load_from_cache(self):
        """Loads arrays from npz and builds the dictionary structure."""
        data = np.load(self.cache_path)

        # Reconstruct dictionaries
        train_hist = {
            "topic_ratios": data["train_topic_ratios"],
            "consistency": data["train_consistency"],
            "history_emb": data["train_history_emb"],
            "history_mask": data["train_history_mask"],
        }
        val_hist = {
            "topic_ratios": data["val_topic_ratios"],
            "consistency": data["val_consistency"],
            "history_emb": data["val_history_emb"],
            "history_mask": data["val_history_mask"],
        }
        test_hist = {
            "topic_ratios": data["test_topic_ratios"],
            "consistency": data["test_consistency"],
            "history_emb": data["test_history_emb"],
            "history_mask": data["test_history_mask"],
        }

        return self._construct_output_dict(
            data["train_sbert"],
            data["train_tfidf"],
            train_hist,
            data["val_sbert"],
            data["val_tfidf"],
            val_hist,
            data["test_sbert"],
            data["test_tfidf"],
            test_hist,
        )

    def _construct_output_dict(
        self, tr_sb, tr_tf, tr_h, va_sb, va_tf, va_h, te_sb, te_tf, te_h
    ):
        return {
            "train": {"sbert_request": tr_sb, "tfidf": tr_tf, **tr_h},
            "val": {"sbert_request": va_sb, "tfidf": va_tf, **va_h},
            "test": {"sbert_request": te_sb, "tfidf": te_tf, **te_h},
        }
