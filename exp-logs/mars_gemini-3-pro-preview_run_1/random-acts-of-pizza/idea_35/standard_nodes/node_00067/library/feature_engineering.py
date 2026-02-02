import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from library.config import Config
from library.utils import ensure_dir, set_seed
from library.data_loader import load_dataset


class TextProcessor:
    """
    Handles text embedding generation (SBERT) and TF-IDF vectorization.
    """

    def __init__(self, sbert_model_name=Config.SBERT_MODEL):
        self.sbert_model = SentenceTransformer(sbert_model_name)
        self.tfidf_vectorizer = TfidfVectorizer(max_features=3000, stop_words="english")

    def get_sbert_embeddings(self, text_list):
        """
        Generates SBERT embeddings for a list of texts.
        """
        # Encode returns a numpy array
        return self.sbert_model.encode(
            text_list, batch_size=64, show_progress_bar=False, convert_to_numpy=True
        )

    def compute_tfidf(self, train_text, val_text, test_text):
        """
        Fits TF-IDF on training text and transforms all splits.
        """
        X_train = self.tfidf_vectorizer.fit_transform(train_text).toarray()
        X_val = self.tfidf_vectorizer.transform(val_text).toarray()
        X_test = self.tfidf_vectorizer.transform(test_text).toarray()
        return X_train, X_val, X_test


class HistoryProcessor:
    """
    Handles history-related features: Subreddit embeddings, Global Alignment Scalars,
    and Top-K binary indicators.
    """

    def __init__(
        self, top_k=Config.TOP_K_SUBREDDITS, max_history_len=Config.MAX_HISTORY_LEN
    ):
        self.top_k = top_k
        self.max_history_len = max_history_len
        self.top_k_subreddits = []
        self.subreddit_to_idx = {}
        self.subreddit_embeddings = None

    def fit_top_k(self, train_subreddits_series):
        """
        Identifies the top K most frequent subreddits from the training set.
        """
        all_subs = [s for sub_list in train_subreddits_series for s in sub_list]
        self.top_k_subreddits = (
            pd.Series(all_subs).value_counts().head(self.top_k).index.tolist()
        )

    def get_top_k_features(self, df):
        """
        Generates binary indicator matrix for top K subreddits.
        """
        X_topk = np.zeros((len(df), self.top_k))
        for i, row in df.iterrows():
            subs = set(row["requester_subreddits_at_request"])
            for j, sub in enumerate(self.top_k_subreddits):
                if sub in subs:
                    X_topk[i, j] = 1
        return X_topk

    def prepare_subreddit_embeddings(self, all_subreddits_list, text_processor):
        """
        Embeds all unique subreddits found in the dataset.
        """
        unique_subs = sorted(list(set(all_subreddits_list)))
        self.subreddit_to_idx = {sub: i for i, sub in enumerate(unique_subs)}
        self.subreddit_embeddings = text_processor.get_sbert_embeddings(unique_subs)

    def compute_history_features(self, df, title_embs, body_embs):
        """
        Computes Global Alignment Scalars (Cosine Sim) and Padded History Sequences.
        """
        global_sim_title = []
        global_sim_body = []
        history_embs_padded = []

        emb_dim = self.subreddit_embeddings.shape[1]

        for idx, row in df.iterrows():
            subs = row["requester_subreddits_at_request"]

            # Identify valid subreddits
            sub_indices = [
                self.subreddit_to_idx[s] for s in subs if s in self.subreddit_to_idx
            ]

            if not sub_indices:
                centroid = np.zeros(emb_dim)
                seq = np.zeros((self.max_history_len, emb_dim))
            else:
                cur_sub_embs = self.subreddit_embeddings[sub_indices]
                centroid = np.mean(cur_sub_embs, axis=0)

                # Pad/Truncate sequence
                seq = np.zeros((self.max_history_len, emb_dim))
                slen = min(len(cur_sub_embs), self.max_history_len)
                seq[:slen] = cur_sub_embs[:slen]

            # Compute Cosine Similarity (Global Alignment)
            # Reshape for sklearn cosine_similarity (1, D)
            if np.linalg.norm(centroid) > 1e-9:
                sim_t = cosine_similarity(
                    title_embs[idx].reshape(1, -1), centroid.reshape(1, -1)
                )[0][0]
                sim_b = cosine_similarity(
                    body_embs[idx].reshape(1, -1), centroid.reshape(1, -1)
                )[0][0]
            else:
                sim_t = 0.0
                sim_b = 0.0

            global_sim_title.append(sim_t)
            global_sim_body.append(sim_b)
            history_embs_padded.append(seq)

        return (
            np.array(global_sim_title),
            np.array(global_sim_body),
            np.array(history_embs_padded),
        )


class MetadataProcessor:
    """
    Handles numerical metadata processing for both RF and MLP streams.
    """

    def __init__(self):
        self.meta_cols = [
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
        self.scaler = StandardScaler()

    def get_rf_metadata(self, df, sim_t, sim_b):
        """
        Prepares metadata for Random Forest: Raw values + Global Alignment Scalars.
        """
        X_meta = df[self.meta_cols].fillna(0).values
        X_sim = np.stack([sim_t, sim_b], axis=1)
        return np.hstack([X_meta, X_sim])

    def fit_scaler(self, X_train):
        self.scaler.fit(X_train)

    def get_mlp_metadata(self, df, sim_t, sim_b, is_train=False):
        """
        Prepares metadata for MLP: Arcsinh transform + Scaling + Global Alignment Scalars.
        """
        # 1. Select and fill
        X_meta = df[self.meta_cols].fillna(0).values

        # 2. Arcsinh Transform
        X_meta = np.arcsinh(X_meta)

        # 3. Append Sim Scalars (before scaling? usually scalars are [0,1] or [-1,1],
        # but let's stack first then scale everything to be safe and uniform)
        X_sim = np.stack([sim_t, sim_b], axis=1)
        X_combined = np.hstack([X_meta, X_sim])

        # 4. Scale
        if is_train:
            self.fit_scaler(X_combined)
            return self.scaler.transform(X_combined)
        else:
            return self.scaler.transform(X_combined)


class FeaturePipeline:
    """
    Orchestrates the feature engineering process.
    """

    def __init__(self):
        self.cache_dir = Config.CACHE_DIR
        self.cache_file = os.path.join(self.cache_dir, "features.npz")
        self.text_processor = TextProcessor()
        self.history_processor = HistoryProcessor()
        self.metadata_processor = MetadataProcessor()

    def run(self, load_cached_data=True):
        ensure_dir(self.cache_dir)

        # 1. Check Cache
        if load_cached_data and os.path.exists(self.cache_file):
            print("Loading features from cache...")
            try:
                loaded = np.load(
                    self.cache_file, allow_pickle=True
                )  # allow_pickle required for object arrays if any, but we aim for pure numeric

                # Reconstruct dictionary structure
                data = {
                    "rf": (
                        loaded["rf_train_X"],
                        loaded["rf_train_y"],
                        loaded["rf_val_X"],
                        loaded["rf_val_y"],
                        loaded["rf_test_X"],
                        loaded["test_ids"],
                    ),
                    "mlp": {
                        "train": (
                            loaded["mlp_train_t"],
                            loaded["mlp_train_b"],
                            loaded["mlp_train_h"],
                            loaded["mlp_train_m"],
                            loaded["rf_train_y"],
                        ),
                        "val": (
                            loaded["mlp_val_t"],
                            loaded["mlp_val_b"],
                            loaded["mlp_val_h"],
                            loaded["mlp_val_m"],
                            loaded["rf_val_y"],
                        ),
                        "test": (
                            loaded["mlp_test_t"],
                            loaded["mlp_test_b"],
                            loaded["mlp_test_h"],
                            loaded["mlp_test_m"],
                            loaded["test_ids"],
                        ),
                    },
                }
                return data
            except Exception as e:
                print(f"Cache load failed: {e}. Recomputing...")

        print("Computing features from scratch...")

        # 2. Load Data
        df_train = load_dataset("train")
        df_val = load_dataset("val")
        df_test = load_dataset("test")

        # 3. Text Processing (SBERT)
        print("Generating SBERT embeddings...")
        train_title_emb = self.text_processor.get_sbert_embeddings(
            df_train["request_title"].tolist()
        )
        val_title_emb = self.text_processor.get_sbert_embeddings(
            df_val["request_title"].tolist()
        )
        test_title_emb = self.text_processor.get_sbert_embeddings(
            df_test["request_title"].tolist()
        )

        train_body_emb = self.text_processor.get_sbert_embeddings(
            df_train["request_text_edit_aware"].tolist()
        )
        val_body_emb = self.text_processor.get_sbert_embeddings(
            df_val["request_text_edit_aware"].tolist()
        )
        test_body_emb = self.text_processor.get_sbert_embeddings(
            df_test["request_text_edit_aware"].tolist()
        )

        # 4. History Processing
        print("Processing history features...")
        # Collect all subreddits for embedding vocabulary
        all_subs = set()
        for df in [df_train, df_val, df_test]:
            for sub_list in df["requester_subreddits_at_request"]:
                all_subs.update(sub_list)

        self.history_processor.prepare_subreddit_embeddings(
            list(all_subs), self.text_processor
        )

        # Compute Global Alignment & History Sequences
        train_sim_t, train_sim_b, train_hist = (
            self.history_processor.compute_history_features(
                df_train, train_title_emb, train_body_emb
            )
        )
        val_sim_t, val_sim_b, val_hist = (
            self.history_processor.compute_history_features(
                df_val, val_title_emb, val_body_emb
            )
        )
        test_sim_t, test_sim_b, test_hist = (
            self.history_processor.compute_history_features(
                df_test, test_title_emb, test_body_emb
            )
        )

        # Top-K (Fit on Train only)
        self.history_processor.fit_top_k(df_train["requester_subreddits_at_request"])
        X_topk_train = self.history_processor.get_top_k_features(df_train)
        X_topk_val = self.history_processor.get_top_k_features(df_val)
        X_topk_test = self.history_processor.get_top_k_features(df_test)

        # 5. TF-IDF (Fit on Train only)
        print("Computing TF-IDF...")
        train_text = (
            df_train["request_title"] + " " + df_train["request_text_edit_aware"]
        )
        val_text = df_val["request_title"] + " " + df_val["request_text_edit_aware"]
        test_text = df_test["request_title"] + " " + df_test["request_text_edit_aware"]

        X_tfidf_train, X_tfidf_val, X_tfidf_test = self.text_processor.compute_tfidf(
            train_text, val_text, test_text
        )

        # 6. Metadata Processing
        print("Processing metadata...")
        # RF Metadata (Raw + Sim)
        X_meta_rf_train = self.metadata_processor.get_rf_metadata(
            df_train, train_sim_t, train_sim_b
        )
        X_meta_rf_val = self.metadata_processor.get_rf_metadata(
            df_val, val_sim_t, val_sim_b
        )
        X_meta_rf_test = self.metadata_processor.get_rf_metadata(
            df_test, test_sim_t, test_sim_b
        )

        # MLP Metadata (Arcsinh + Sim + Scaled)
        X_meta_mlp_train = self.metadata_processor.get_mlp_metadata(
            df_train, train_sim_t, train_sim_b, is_train=True
        )
        X_meta_mlp_val = self.metadata_processor.get_mlp_metadata(
            df_val, val_sim_t, val_sim_b, is_train=False
        )
        X_meta_mlp_test = self.metadata_processor.get_mlp_metadata(
            df_test, test_sim_t, test_sim_b, is_train=False
        )

        # 7. Assemble Final Datasets
        # RF: Meta + TopK + TFIDF
        X_rf_train = np.hstack([X_meta_rf_train, X_topk_train, X_tfidf_train])
        X_rf_val = np.hstack([X_meta_rf_val, X_topk_val, X_tfidf_val])
        X_rf_test = np.hstack([X_meta_rf_test, X_topk_test, X_tfidf_test])

        y_train = df_train["requester_received_pizza"].astype(int).values
        y_val = df_val["requester_received_pizza"].astype(int).values
        test_ids = df_test["request_id"].values

        # 8. Save to Cache
        print("Saving features to cache...")
        np.savez(
            self.cache_file,
            # RF Data
            rf_train_X=X_rf_train,
            rf_train_y=y_train,
            rf_val_X=X_rf_val,
            rf_val_y=y_val,
            rf_test_X=X_rf_test,
            test_ids=test_ids,
            # MLP Data
            mlp_train_t=train_title_emb,
            mlp_train_b=train_body_emb,
            mlp_train_h=train_hist,
            mlp_train_m=X_meta_mlp_train,
            mlp_val_t=val_title_emb,
            mlp_val_b=val_body_emb,
            mlp_val_h=val_hist,
            mlp_val_m=X_meta_mlp_val,
            mlp_test_t=test_title_emb,
            mlp_test_b=test_body_emb,
            mlp_test_h=test_hist,
            mlp_test_m=X_meta_mlp_test,
        )

        data = {
            "rf": (X_rf_train, y_train, X_rf_val, y_val, X_rf_test, test_ids),
            "mlp": {
                "train": (
                    train_title_emb,
                    train_body_emb,
                    train_hist,
                    X_meta_mlp_train,
                    y_train,
                ),
                "val": (val_title_emb, val_body_emb, val_hist, X_meta_mlp_val, y_val),
                "test": (
                    test_title_emb,
                    test_body_emb,
                    test_hist,
                    X_meta_mlp_test,
                    test_ids,
                ),
            },
        }
        return data
