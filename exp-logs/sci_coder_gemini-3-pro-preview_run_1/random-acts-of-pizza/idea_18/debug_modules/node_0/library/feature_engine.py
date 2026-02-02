import os
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from library.config import Config
from library.utils import set_seed
from library.data_loader import load_data


class FeatureExtractor:
    def __init__(self):
        self.numeric_cols = [
            "requester_account_age_in_days_at_request",
            "requester_days_since_first_post_on_raop_at_request",
            "requester_number_of_comments_at_request",
            "requester_number_of_comments_in_raop_at_request",
            "requester_number_of_posts_at_request",
            "requester_number_of_posts_on_raop_at_request",
            "requester_number_of_subreddits_at_request",
            "requester_upvotes_minus_downvotes_at_request",
            "requester_upvotes_plus_downvotes_at_request",
            "unix_timestamp_of_request",
        ]
        self.target_col = "requester_received_pizza"

        # Initialize models/transformers placeholders
        self.sbert_model = None
        self.tfidf_vectorizer = None
        self.pca = None
        self.scaler = None
        self.imputer = None

    def _init_sbert(self):
        if self.sbert_model is None:
            # Load SBERT model on the configured device
            self.sbert_model = SentenceTransformer(
                Config.SBERT_MODEL_NAME, device=Config.DEVICE
            )

    def _get_text_features(self, df):
        """Combines title and text, filling NaNs."""
        title = df["request_title"].fillna("")
        body = df["request_text"].fillna("")
        return (title + " " + body).astype(str).tolist()

    def _engineer_metadata(self, df):
        """
        Generates 'Full-Spectrum' metadata: Raw + Ratios + Text Stats.
        Returns a DataFrame of numeric features.
        """
        # 1. Select Raw Numeric Columns
        # Ensure columns exist (intersection)
        valid_cols = [c for c in self.numeric_cols if c in df.columns]
        meta = df[valid_cols].copy()

        # 2. Engineer Ratios (Safe Division)
        # Upvote Ratio
        up_plus_down = df.get("requester_upvotes_plus_downvotes_at_request", 0)
        up_minus_down = df.get("requester_upvotes_minus_downvotes_at_request", 0)
        # derive upvotes approx: (sum + diff) / 2
        # ratio: upvotes / sum
        # Avoid div by zero
        meta["upvote_ratio"] = (up_plus_down + up_minus_down) / (
            2 * (up_plus_down + 1e-9)
        )

        # RAOP Activity Ratio
        total_posts = df.get("requester_number_of_posts_at_request", 0)
        raop_posts = df.get("requester_number_of_posts_on_raop_at_request", 0)
        meta["raop_post_ratio"] = raop_posts / (total_posts + 1e-9)

        # 3. Text Stats
        combined_text = pd.Series(self._get_text_features(df))
        meta["text_len_char"] = combined_text.apply(len)
        meta["text_len_word"] = combined_text.apply(lambda x: len(x.split()))
        meta["caps_ratio"] = combined_text.apply(
            lambda x: sum(1 for c in x if c.isupper()) / (len(x) + 1e-9)
        )

        return meta

    def _process_history_embeddings(self, df_train, df_val, df_test):
        """
        Generates SBERT embeddings for subreddits and aggregates them.
        Returns:
            - history_centroids_pca (dict): {split: np.array} for RF (PCA reduced means)
            - history_sequences (dict): {split: np.array} for MLP (Padded sequences)
        """
        self._init_sbert()

        # 1. Collect all unique subreddits
        all_subreddits = set()
        for df in [df_train, df_val, df_test]:
            for sub_list in df["requester_subreddits_at_request"]:
                all_subreddits.update(sub_list)

        all_subreddits = list(all_subreddits)
        if not all_subreddits:
            # Handle edge case where no subreddits exist
            dummy_emb = np.zeros((1, Config.SBERT_EMBEDDING_DIM))
            sub_to_emb = {}
        else:
            # 2. Encode unique subreddits
            print(f"Encoding {len(all_subreddits)} unique subreddits...")
            embeddings = self.sbert_model.encode(
                all_subreddits,
                batch_size=128,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            sub_to_emb = {sub: emb for sub, emb in zip(all_subreddits, embeddings)}

        # 3. Process each split
        history_centroids = {}
        history_sequences = {}

        splits = {"train": df_train, "val": df_val, "test": df_test}

        for name, df in splits.items():
            centroids = []
            sequences = []

            for sub_list in df["requester_subreddits_at_request"]:
                # Get embeddings for this user's history
                user_embs = [sub_to_emb[s] for s in sub_list if s in sub_to_emb]

                # A. Centroids (Mean)
                if user_embs:
                    user_embs_arr = np.array(user_embs)
                    centroids.append(np.mean(user_embs_arr, axis=0))
                else:
                    centroids.append(np.zeros(Config.SBERT_EMBEDDING_DIM))

                # B. Sequences (Pad/Truncate)
                # Truncate to MAX_HISTORY_LEN
                seq = user_embs[: Config.MAX_HISTORY_LEN]
                # Pad if shorter
                if len(seq) < Config.MAX_HISTORY_LEN:
                    padding = [np.zeros(Config.SBERT_EMBEDDING_DIM)] * (
                        Config.MAX_HISTORY_LEN - len(seq)
                    )
                    seq.extend(padding)
                sequences.append(np.array(seq))

            history_centroids[name] = np.array(centroids)
            history_sequences[name] = np.array(sequences)

        # 4. PCA Compression for RF
        print("Fitting PCA on history centroids...")
        self.pca = PCA(
            n_components=Config.PCA_COMPONENTS, random_state=Config.RANDOM_STATE
        )
        self.pca.fit(history_centroids["train"])

        history_centroids_pca = {
            k: self.pca.transform(v) for k, v in history_centroids.items()
        }

        return history_centroids_pca, history_sequences

    def run(self, load_cached_data=True):
        """
        Main execution method.
        Checks cache, loads or computes features, and saves to cache.
        """
        # Define cache paths
        cache_dir = Config.WORKING_DIR
        files = {
            "rf_train": os.path.join(cache_dir, "rf_features_train.npz"),
            "rf_val": os.path.join(cache_dir, "rf_features_val.npz"),
            "rf_test": os.path.join(cache_dir, "rf_features_test.npz"),
            "mlp_train": os.path.join(cache_dir, "mlp_features_train.npz"),
            "mlp_val": os.path.join(cache_dir, "mlp_features_val.npz"),
            "mlp_test": os.path.join(cache_dir, "mlp_features_test.npz"),
            "targets": os.path.join(cache_dir, "targets.npz"),
        }

        # Check if all cache files exist
        all_cached = all(os.path.exists(f) for f in files.values())

        if load_cached_data and all_cached:
            print("Loading features from cache...")
            data = {}
            for k, v in files.items():
                data[k] = np.load(v, allow_pickle=True)

            # Reconstruct dictionary structure
            output = {
                "train": {
                    "rf_features": sparse.load_npz(files["rf_train"]),
                    "mlp": {
                        "request_emb": data["mlp_train"]["request_emb"],
                        "history_seq": data["mlp_train"]["history_seq"],
                        "metadata": data["mlp_train"]["metadata"],
                    },
                    "y": data["targets"]["y_train"],
                },
                "val": {
                    "rf_features": sparse.load_npz(files["rf_val"]),
                    "mlp": {
                        "request_emb": data["mlp_val"]["request_emb"],
                        "history_seq": data["mlp_val"]["history_seq"],
                        "metadata": data["mlp_val"]["metadata"],
                    },
                    "y": data["targets"]["y_val"],
                },
                "test": {
                    "rf_features": sparse.load_npz(files["rf_test"]),
                    "mlp": {
                        "request_emb": data["mlp_test"]["request_emb"],
                        "history_seq": data["mlp_test"]["history_seq"],
                        "metadata": data["mlp_test"]["metadata"],
                    },
                },
            }
            return output

        # If not cached, compute from scratch
        print("Computing features from scratch...")
        df_train, df_val, df_test = load_data(load_cached_data=load_cached_data)

        # --- 1. History Features (SBERT + PCA) ---
        print("Processing history embeddings...")
        hist_pca, hist_seq = self._process_history_embeddings(df_train, df_val, df_test)

        # --- 2. Text Features (Request) ---
        print("Processing request text...")
        # A. TF-IDF for RF
        train_text = self._get_text_features(df_train)
        val_text = self._get_text_features(df_val)
        test_text = self._get_text_features(df_test)

        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=Config.TFIDF_MAX_FEATURES,
            ngram_range=Config.TFIDF_NGRAM_RANGE,
            stop_words="english",
        )
        tfidf_train = self.tfidf_vectorizer.fit_transform(train_text)
        tfidf_val = self.tfidf_vectorizer.transform(val_text)
        tfidf_test = self.tfidf_vectorizer.transform(test_text)

        # B. SBERT for MLP
        self._init_sbert()
        req_emb_train = self.sbert_model.encode(
            train_text, batch_size=64, show_progress_bar=False
        )
        req_emb_val = self.sbert_model.encode(
            val_text, batch_size=64, show_progress_bar=False
        )
        req_emb_test = self.sbert_model.encode(
            test_text, batch_size=64, show_progress_bar=False
        )

        # --- 3. Metadata Features ---
        print("Processing metadata...")
        meta_train_df = self._engineer_metadata(df_train)
        meta_val_df = self._engineer_metadata(df_val)
        meta_test_df = self._engineer_metadata(df_test)

        # A. For RF (Imputed, Raw Scale)
        self.imputer = SimpleImputer(strategy="median")
        meta_rf_train = self.imputer.fit_transform(meta_train_df)
        meta_rf_val = self.imputer.transform(meta_val_df)
        meta_rf_test = self.imputer.transform(meta_test_df)

        # B. For MLP (Arcsinh + Scaled)
        # Apply arcsinh to handle skew
        meta_mlp_train_raw = np.arcsinh(meta_rf_train)
        meta_mlp_val_raw = np.arcsinh(meta_rf_val)
        meta_mlp_test_raw = np.arcsinh(meta_rf_test)

        self.scaler = StandardScaler()
        meta_mlp_train = self.scaler.fit_transform(meta_mlp_train_raw)
        meta_mlp_val = self.scaler.transform(meta_mlp_val_raw)
        meta_mlp_test = self.scaler.transform(meta_mlp_test_raw)

        # --- 4. Assemble Stream A (RF) ---
        # Concatenate: TFIDF (Sparse) + Meta (Dense) + History PCA (Dense)
        # Convert dense to sparse for efficient hstack
        def assemble_rf(tfidf, meta, hist):
            return sparse.hstack(
                [tfidf, sparse.csr_matrix(meta), sparse.csr_matrix(hist)]
            ).tocsr()

        rf_train = assemble_rf(tfidf_train, meta_rf_train, hist_pca["train"])
        rf_val = assemble_rf(tfidf_val, meta_rf_val, hist_pca["val"])
        rf_test = assemble_rf(tfidf_test, meta_rf_test, hist_pca["test"])

        # --- 5. Targets ---
        y_train = df_train[self.target_col].astype(int).values
        y_val = df_val[self.target_col].astype(int).values

        # --- 6. Save to Cache ---
        print("Saving features to cache...")
        sparse.save_npz(files["rf_train"], rf_train)
        sparse.save_npz(files["rf_val"], rf_val)
        sparse.save_npz(files["rf_test"], rf_test)

        np.savez_compressed(
            files["mlp_train"],
            request_emb=req_emb_train,
            history_seq=hist_seq["train"],
            metadata=meta_mlp_train,
        )
        np.savez_compressed(
            files["mlp_val"],
            request_emb=req_emb_val,
            history_seq=hist_seq["val"],
            metadata=meta_mlp_val,
        )
        np.savez_compressed(
            files["mlp_test"],
            request_emb=req_emb_test,
            history_seq=hist_seq["test"],
            metadata=meta_mlp_test,
        )

        np.savez_compressed(files["targets"], y_train=y_train, y_val=y_val)

        # --- 7. Return ---
        output = {
            "train": {
                "rf_features": rf_train,
                "mlp": {
                    "request_emb": req_emb_train,
                    "history_seq": hist_seq["train"],
                    "metadata": meta_mlp_train,
                },
                "y": y_train,
            },
            "val": {
                "rf_features": rf_val,
                "mlp": {
                    "request_emb": req_emb_val,
                    "history_seq": hist_seq["val"],
                    "metadata": meta_mlp_val,
                },
                "y": y_val,
            },
            "test": {
                "rf_features": rf_test,
                "mlp": {
                    "request_emb": req_emb_test,
                    "history_seq": hist_seq["test"],
                    "metadata": meta_mlp_test,
                },
            },
        }
        return output
