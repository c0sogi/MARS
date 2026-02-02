import os
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import scipy.sparse

from library import config, utils, data_loader


class FeaturePipeline:
    def __init__(self):
        self.sbert_model = None
        self.kmeans_model = None
        self.tfidf_vectorizer = None
        self.scaler = None
        self.imputer = None
        self.subreddit_embedding_map = {}
        self.subreddit_cluster_map = {}

        # Cache paths
        self.cache_dir = config.CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_sbert_model(self):
        if self.sbert_model is None:
            # Load SBERT model
            # Note: We use the device from config
            self.sbert_model = SentenceTransformer(
                config.SBERT_MODEL_NAME, device=config.DEVICE
            )
        return self.sbert_model

    def _encode_text(self, texts, batch_size=32):
        model = self._get_sbert_model()
        # Encode
        embeddings = model.encode(
            texts, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True
        )
        return embeddings

    def _build_subreddit_map(self, all_dfs):
        """
        Collects all unique subreddits from all splits, embeds them, and builds maps.
        """
        unique_subreddits = set()
        for df in all_dfs:
            for sub_list in df["requester_subreddits_at_request"]:
                unique_subreddits.update(sub_list)

        sorted_subs = sorted(list(unique_subreddits))
        if not sorted_subs:
            return

        embeddings = self._encode_text(sorted_subs)
        self.subreddit_embedding_map = {
            sub: emb for sub, emb in zip(sorted_subs, embeddings)
        }

    def _fit_topic_model(self, train_df):
        """
        Fits K-Means on subreddits present in the training set.
        """
        train_subs = set()
        for sub_list in train_df["requester_subreddits_at_request"]:
            train_subs.update(sub_list)

        # Filter embeddings for training subreddits
        train_sub_embeddings = []
        for sub in train_subs:
            if sub in self.subreddit_embedding_map:
                train_sub_embeddings.append(self.subreddit_embedding_map[sub])

        if not train_sub_embeddings:
            # Fallback if no subreddits
            self.kmeans_model = KMeans(
                n_clusters=config.TOPIC_CLUSTERS_K, random_state=config.SEED, n_init=10
            )
            # Fit on dummy data to avoid errors
            self.kmeans_model.fit(np.zeros((10, 384)))
            return

        X_train_subs = np.array(train_sub_embeddings)

        self.kmeans_model = KMeans(
            n_clusters=config.TOPIC_CLUSTERS_K, random_state=config.SEED, n_init=10
        )
        self.kmeans_model.fit(X_train_subs)

    def _assign_clusters(self):
        """
        Assigns a cluster to every known subreddit in the embedding map.
        """
        all_subs = list(self.subreddit_embedding_map.keys())
        all_embs = np.array([self.subreddit_embedding_map[s] for s in all_subs])

        if len(all_subs) > 0:
            clusters = self.kmeans_model.predict(all_embs)
            self.subreddit_cluster_map = {
                sub: clust for sub, clust in zip(all_subs, clusters)
            }

    def _generate_topic_features(self, df):
        """
        Generates K features representing the ratio of user history in each topic cluster.
        """
        n_samples = len(df)
        n_clusters = config.TOPIC_CLUSTERS_K
        topic_features = np.zeros((n_samples, n_clusters))

        for idx, sub_list in enumerate(df["requester_subreddits_at_request"]):
            if not sub_list:
                continue

            counts = np.zeros(n_clusters)
            total = 0
            for sub in sub_list:
                if sub in self.subreddit_cluster_map:
                    cluster = self.subreddit_cluster_map[sub]
                    counts[cluster] += 1
                    total += 1

            if total > 0:
                topic_features[idx] = counts / total

        return topic_features

    def _generate_consistency_score(self, df, request_embeddings):
        """
        Calculates Cosine Similarity between Request Embedding and History Centroid.
        """
        scores = np.zeros((len(df), 1))

        for idx, (sub_list, req_emb) in enumerate(
            zip(df["requester_subreddits_at_request"], request_embeddings)
        ):
            if not sub_list:
                # No history, neutral score (0)
                continue

            # Gather history embeddings
            hist_embs = []
            for sub in sub_list:
                if sub in self.subreddit_embedding_map:
                    hist_embs.append(self.subreddit_embedding_map[sub])

            if not hist_embs:
                continue

            # Centroid
            centroid = np.mean(hist_embs, axis=0)

            # Cosine Similarity
            # Reshape for sklearn: (1, D)
            sim = cosine_similarity(req_emb.reshape(1, -1), centroid.reshape(1, -1))[0][
                0
            ]
            scores[idx] = sim

        return scores

    def _prepare_history_tensor(self, df, max_len=50):
        """
        Prepares padded tensor of history embeddings for MLP attention.
        Returns: (N, max_len, 384) and Mask (N, max_len)
        """
        N = len(df)
        emb_dim = config.MLP_INPUT_DIM_TEXT

        # Determine max length from config or data (capping at reasonable number)
        # Using 50 as a reasonable history length for attention

        history_tensor = np.zeros((N, max_len, emb_dim), dtype=np.float32)
        mask = np.zeros((N, max_len), dtype=np.float32)  # 1 for valid, 0 for padding

        for idx, sub_list in enumerate(df["requester_subreddits_at_request"]):
            valid_embs = []
            for sub in sub_list:
                if sub in self.subreddit_embedding_map:
                    valid_embs.append(self.subreddit_embedding_map[sub])

            # Truncate or Pad
            num_items = min(len(valid_embs), max_len)

            for t in range(num_items):
                history_tensor[idx, t, :] = valid_embs[t]
                mask[idx, t] = 1.0

        return history_tensor, mask

    def _engineer_metadata(
        self, df, topic_features, consistency_scores, is_train=False
    ):
        """
        Combines raw numeric cols, ratios, topics, and consistency.
        Returns raw (imputed) for RF and normalized for MLP.
        """
        # 1. Base Numeric Columns
        # Select columns that are safe (not leakage) and numeric
        numeric_candidates = [
            "requester_account_age_in_days_at_request",
            "requester_days_since_first_post_on_raop_at_request",
            "requester_number_of_comments_at_request",
            "requester_number_of_comments_in_raop_at_request",
            "requester_number_of_posts_at_request",
            "requester_number_of_posts_on_raop_at_request",
            "requester_number_of_subreddits_at_request",
            "requester_upvotes_minus_downvotes_at_request",
            "requester_upvotes_plus_downvotes_at_request",
        ]

        # Filter for existence
        base_cols = [c for c in numeric_candidates if c in df.columns]
        X_base = df[base_cols].values.astype(np.float32)

        # 2. Engineered Ratios
        # Upvote Ratio: (Up-Down) / (Up+Down) -> approximated via columns
        # We have (U-D) and (U+D).
        # Upvotes = ((U+D) + (U-D)) / 2
        # Downvotes = ((U+D) - (U-D)) / 2
        # Ratio = U / (U+D) if U+D > 0 else 0.5
        u_plus_d = df["requester_upvotes_plus_downvotes_at_request"].values
        u_minus_d = df["requester_upvotes_minus_downvotes_at_request"].values

        # Avoid div by zero
        with np.errstate(divide="ignore", invalid="ignore"):
            upvote_ratio = (u_plus_d + u_minus_d) / (2 * u_plus_d)
            upvote_ratio[u_plus_d == 0] = 0.5

            # Comments per Post
            n_posts = df["requester_number_of_posts_at_request"].values
            n_comments = df["requester_number_of_comments_at_request"].values
            comments_per_post = n_comments / n_posts
            comments_per_post[n_posts == 0] = 0.0

        ratio_features = np.stack([upvote_ratio, comments_per_post], axis=1)

        # 3. Concatenate All
        X_full = np.hstack([X_base, ratio_features, topic_features, consistency_scores])

        # 4. Imputation
        if is_train:
            self.imputer = SimpleImputer(strategy="median")
            X_full = self.imputer.fit_transform(X_full)
        else:
            X_full = self.imputer.transform(X_full)

        # 5. Normalization for MLP (Stream B)
        # Apply Arcsinh to handle heavy tails before scaling
        X_arcsinh = np.arcsinh(X_full)

        if is_train:
            self.scaler = StandardScaler()
            X_norm = self.scaler.fit_transform(X_arcsinh)
        else:
            X_norm = self.scaler.transform(X_arcsinh)

        return X_full, X_norm

    def run(self, load_cached_data=True):
        """
        Main execution method.
        """
        # Check cache
        cache_files = {
            "train": os.path.join(self.cache_dir, "train_features.npz"),
            "val": os.path.join(self.cache_dir, "val_features.npz"),
            "test": os.path.join(self.cache_dir, "test_features.npz"),
        }

        if load_cached_data and all(os.path.exists(f) for f in cache_files.values()):
            print("Loading features from cache...")
            results = {}
            for split, path in cache_files.items():
                data = np.load(path, allow_pickle=True)
                # Reconstruct dictionary structure
                results[split] = {
                    "y": data["y"],
                    "ids": data["ids"],
                    "stream_a": {
                        "X_tfidf": scipy.sparse.csr_matrix(
                            (
                                data["sa_tfidf_data"],
                                data["sa_tfidf_indices"],
                                data["sa_tfidf_indptr"],
                            ),
                            shape=data["sa_tfidf_shape"],
                        ),
                        "X_meta": data["sa_meta"],
                    },
                    "stream_b": {
                        "X_request_emb": data["sb_req_emb"],
                        "X_history_emb": data["sb_hist_emb"],
                        "X_history_mask": data["sb_hist_mask"],
                        "X_meta": data["sb_meta"],
                    },
                }
            return results["train"], results["val"], results["test"]

        print("Generating features from scratch...")
        utils.set_seed(config.SEED)

        # Load Data
        df_train = data_loader.load_dataset("train", load_cached_data=True)
        df_val = data_loader.load_dataset("val", load_cached_data=True)
        df_test = data_loader.load_dataset("test", load_cached_data=True)

        # Debug sampling
        if config.DEBUG_SAMPLE_SIZE:
            df_train = df_train.head(config.DEBUG_SAMPLE_SIZE)
            df_val = df_val.head(config.DEBUG_SAMPLE_SIZE)
            df_test = df_test.head(config.DEBUG_SAMPLE_SIZE)

        # 1. Build Subreddit Map (Train+Val+Test)
        print("Building Subreddit Semantic Map...")
        self._build_subreddit_map([df_train, df_val, df_test])

        # 2. Fit Topic Model (Train only)
        print("Fitting Topic Model...")
        self._fit_topic_model(df_train)
        self._assign_clusters()

        # 3. TF-IDF (Stream A)
        print("Vectorizing Text (TF-IDF)...")
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=config.TFIDF_MAX_FEATURES,
            ngram_range=config.TFIDF_NGRAM_RANGE,
            stop_words="english",
        )
        # Fit on Train
        self.tfidf_vectorizer.fit(df_train["request_text_edit_aware"])

        # Process each split
        results = {}
        for split, df in zip(["train", "val", "test"], [df_train, df_val, df_test]):
            print(f"Processing {split} split...")
            is_train = split == "train"

            # --- Common ---
            # Request Embeddings
            req_embs = self._encode_text(df["request_text_edit_aware"].tolist())

            # Topic Features
            topic_feats = self._generate_topic_features(df)

            # Consistency Scores
            consistency_scores = self._generate_consistency_score(df, req_embs)

            # Metadata (Stream A & B)
            X_meta_rf, X_meta_mlp = self._engineer_metadata(
                df, topic_feats, consistency_scores, is_train=is_train
            )

            # --- Stream A (RF) ---
            # TF-IDF
            X_tfidf = self.tfidf_vectorizer.transform(df["request_text_edit_aware"])

            # --- Stream B (MLP) ---
            # History Tensor
            hist_tensor, hist_mask = self._prepare_history_tensor(df)

            # Target
            y = (
                df["requester_received_pizza"].values
                if "requester_received_pizza" in df.columns
                else np.zeros(len(df))
            )
            ids = df["request_id"].values

            # Store
            results[split] = {
                "y": y,
                "ids": ids,
                "stream_a": {"X_tfidf": X_tfidf, "X_meta": X_meta_rf},
                "stream_b": {
                    "X_request_emb": req_embs,
                    "X_history_emb": hist_tensor,
                    "X_history_mask": hist_mask,
                    "X_meta": X_meta_mlp,
                },
            }

            # Save to cache
            # Sparse matrix handling for save
            save_dict = {
                "y": y,
                "ids": ids,
                "sa_tfidf_data": X_tfidf.data,
                "sa_tfidf_indices": X_tfidf.indices,
                "sa_tfidf_indptr": X_tfidf.indptr,
                "sa_tfidf_shape": X_tfidf.shape,
                "sa_meta": X_meta_rf,
                "sb_req_emb": req_embs,
                "sb_hist_emb": hist_tensor,
                "sb_hist_mask": hist_mask,
                "sb_meta": X_meta_mlp,
            }
            np.savez_compressed(cache_files[split], **save_dict)

        return results["train"], results["val"], results["test"]
