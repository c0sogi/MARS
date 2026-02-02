import os
import numpy as np
import pandas as pd
import scipy.sparse
import json
import pickle
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
from library import config, utils, lsa_projection


class FeatureFactory:
    def __init__(self):
        self.cache_dir = config.WORKING_DIR
        self.tfidf_path = os.path.join(self.cache_dir, "rf_tfidf.pkl")
        self.scaler_path = os.path.join(self.cache_dir, "mlp_scaler.pkl")
        self.vocab_path = os.path.join(self.cache_dir, "history_vocab.json")
        self.rf_data_path = os.path.join(self.cache_dir, "rf_data.npz")
        self.mlp_data_path = os.path.join(self.cache_dir, "mlp_data.npz")
        self.targets_path = os.path.join(self.cache_dir, "targets.npz")

        # Initialize processors
        self.tfidf = TfidfVectorizer(
            max_features=config.TFIDF_MAX_FEATURES,
            ngram_range=config.TFIDF_NGRAM_RANGE,
            stop_words="english",
            sublinear_tf=True,
        )
        self.scaler = StandardScaler()
        self.sbert = SentenceTransformer(config.SBERT_MODEL_NAME)
        if torch.cuda.is_available():
            self.sbert = self.sbert.to("cuda")

    def _extract_metadata(self, df):
        """
        Generates 'Full-Spectrum' metadata: Raw counts + Engineered Ratios + Text Meta-features.
        """
        # 1. Base Numerical Features
        # Identify columns that are numeric and not in DROP_COLS
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        valid_cols = [c for c in numeric_cols if c not in config.DROP_COLS]

        meta_df = df[valid_cols].copy()

        # Fill NaNs with 0 for counts/ages (safe assumption for this dataset's specific missing patterns)
        meta_df = meta_df.fillna(0)

        # 2. Text Meta-Features
        text_col = "request_text_edit_aware"
        title_col = "request_title"

        # Helper for safe text len
        def get_len(x):
            return len(str(x)) if not pd.isna(x) else 0

        def get_caps(x):
            return sum(1 for c in str(x) if c.isupper())

        meta_df["text_len_char"] = df[text_col].apply(get_len)
        meta_df["title_len_char"] = df[title_col].apply(get_len)
        meta_df["text_caps_count"] = df[text_col].apply(get_caps)
        meta_df["text_caps_ratio"] = meta_df["text_caps_count"] / (
            meta_df["text_len_char"] + 1
        )

        # 3. Interaction Ratios
        # Upvote Ratio: (up - down) is net, (up + down) is total.
        # We want approx upvote ratio.
        # plus = up + down, minus = up - down. => 2*up = plus + minus => up = (plus+minus)/2
        # ratio = up / (plus + epsilon)
        plus = meta_df.get("requester_upvotes_plus_downvotes_at_request", 0)
        minus = meta_df.get("requester_upvotes_minus_downvotes_at_request", 0)

        # Avoid division by zero
        meta_df["upvote_ratio"] = ((plus + minus) / 2) / (plus + 1.0)

        # Comment to Post Ratio
        comments = meta_df.get("requester_number_of_comments_at_request", 0)
        posts = meta_df.get("requester_number_of_posts_at_request", 0)
        meta_df["comment_post_ratio"] = comments / (posts + 1.0)

        return meta_df

    def _process_history_sequences(self, train_subs, val_subs, test_subs):
        """
        Tokenizes subreddit lists for MLP attention.
        Returns: (train_seq, val_seq, test_seq), vocab_size
        """
        # Build Vocabulary from Train
        vocab = {"<PAD>": 0, "<UNK>": 1}

        # Flatten train to find unique subreddits
        all_subs = [sub for sub_list in train_subs for sub in sub_list]
        unique_subs = sorted(list(set(all_subs)))

        for sub in unique_subs:
            if sub not in vocab:
                vocab[sub] = len(vocab)

        # Save vocab
        with open(self.vocab_path, "w") as f:
            json.dump(vocab, f)

        # Determine max length (percentile to avoid outliers)
        lens = [len(x) for x in train_subs]
        max_len = int(np.percentile(lens, 95)) if lens else 10
        max_len = max(5, min(max_len, 50))  # Clamp between 5 and 50

        def vectorize(sub_lists):
            batch_size = len(sub_lists)
            seqs = np.zeros((batch_size, max_len), dtype=np.int32)
            for i, subs in enumerate(sub_lists):
                # Truncate or Pad
                subs = subs[:max_len]
                for j, sub in enumerate(subs):
                    seqs[i, j] = vocab.get(sub, vocab["<UNK>"])
            return seqs

        return (
            vectorize(train_subs),
            vectorize(val_subs),
            vectorize(test_subs),
            len(vocab),
        )

    def fit_transform(self, load_cached_data=True):
        """
        Main execution method.
        """
        # 1. Check Cache
        if (
            load_cached_data
            and os.path.exists(self.rf_data_path)
            and os.path.exists(self.mlp_data_path)
        ):
            print("Loading features from cache...")
            rf_data = np.load(self.rf_data_path, allow_pickle=True)
            mlp_data = np.load(self.mlp_data_path, allow_pickle=True)
            targets = np.load(self.targets_path, allow_pickle=True)

            # Reconstruct Sparse Matrices for RF
            def load_sparse(prefix, split):
                return scipy.sparse.csr_matrix(
                    (
                        rf_data[f"{prefix}_{split}_data"],
                        rf_data[f"{prefix}_{split}_indices"],
                        rf_data[f"{prefix}_{split}_indptr"],
                    ),
                    shape=rf_data[f"{prefix}_{split}_shape"],
                )

            return {
                "rf": {
                    "train": load_sparse("rf", "train"),
                    "val": load_sparse("rf", "val"),
                    "test": load_sparse("rf", "test"),
                },
                "mlp": {
                    "train": {
                        "text": mlp_data["train_text"],
                        "history": mlp_data["train_hist"],
                        "meta": mlp_data["train_meta"],
                    },
                    "val": {
                        "text": mlp_data["val_text"],
                        "history": mlp_data["val_hist"],
                        "meta": mlp_data["val_meta"],
                    },
                    "test": {
                        "text": mlp_data["test_text"],
                        "history": mlp_data["test_hist"],
                        "meta": mlp_data["test_meta"],
                    },
                },
                "targets": {"train": targets["train"], "val": targets["val"]},
                "dims": {
                    "vocab_size": int(mlp_data["vocab_size"]),
                    "history_dim": int(mlp_data["train_hist"].shape[1]),
                    "meta_dim": int(mlp_data["train_meta"].shape[1]),
                    "sbert_dim": int(mlp_data["train_text"].shape[1]),
                },
            }

        print("Generating features from scratch...")

        # 2. Load Data
        df_train, df_val, df_test = utils.load_data(load_cached_data=load_cached_data)

        # 3. LSA Features (Stream A)
        print("Generating LSA features...")
        lsa_train, lsa_val, lsa_test = lsa_projection.get_lsa_features(
            df_train, df_val, df_test, load_cached_data=load_cached_data
        )

        # 4. Metadata (Shared -> Processed differently)
        print("Generating Metadata...")
        meta_train_raw = self._extract_metadata(df_train)
        meta_val_raw = self._extract_metadata(df_val)
        meta_test_raw = self._extract_metadata(df_test)

        # Align columns (ensure test has same columns as train)
        common_cols = meta_train_raw.columns.tolist()
        meta_val_raw = meta_val_raw[common_cols]
        meta_test_raw = meta_test_raw[common_cols]

        # 5. Text Features
        print("Generating Text Features (TF-IDF & SBERT)...")

        # Combine Title + Body
        def combine_text(df):
            return (
                df["request_title"].fillna("")
                + " "
                + df["request_text_edit_aware"].fillna("")
            ).tolist()

        txt_train = combine_text(df_train)
        txt_val = combine_text(df_val)
        txt_test = combine_text(df_test)

        # RF: TF-IDF
        tfidf_train = self.tfidf.fit_transform(txt_train)
        tfidf_val = self.tfidf.transform(txt_val)
        tfidf_test = self.tfidf.transform(txt_test)

        # MLP: SBERT
        sbert_train = self.sbert.encode(
            txt_train, batch_size=64, show_progress_bar=False
        )
        sbert_val = self.sbert.encode(txt_val, batch_size=64, show_progress_bar=False)
        sbert_test = self.sbert.encode(txt_test, batch_size=64, show_progress_bar=False)

        # 6. History Sequences (MLP)
        print("Generating History Sequences...")
        seq_train, seq_val, seq_test, vocab_size = self._process_history_sequences(
            df_train["requester_subreddits_at_request"].tolist(),
            df_val["requester_subreddits_at_request"].tolist(),
            df_test["requester_subreddits_at_request"].tolist(),
        )

        # 7. Process Metadata for MLP (Arcsinh + Scale)
        print("Processing Metadata for MLP...")
        # Arcsinh handles skew and zeros well (similar to log(1+x) but handles negatives if any)
        meta_train_mlp = np.arcsinh(meta_train_raw.values)
        meta_val_mlp = np.arcsinh(meta_val_raw.values)
        meta_test_mlp = np.arcsinh(meta_test_raw.values)

        meta_train_mlp = self.scaler.fit_transform(meta_train_mlp)
        meta_val_mlp = self.scaler.transform(meta_val_mlp)
        meta_test_mlp = self.scaler.transform(meta_test_mlp)

        # 8. Assemble RF Input (Concatenate TF-IDF + LSA + Raw Meta)
        # Note: RF handles raw numeric scales fine, so we use meta_train_raw
        print("Assembling RF Inputs...")
        rf_train = scipy.sparse.hstack(
            [tfidf_train, lsa_train, scipy.sparse.csr_matrix(meta_train_raw.values)]
        )
        rf_val = scipy.sparse.hstack(
            [tfidf_val, lsa_val, scipy.sparse.csr_matrix(meta_val_raw.values)]
        )
        rf_test = scipy.sparse.hstack(
            [tfidf_test, lsa_test, scipy.sparse.csr_matrix(meta_test_raw.values)]
        )

        # 9. Targets
        y_train = df_train["requester_received_pizza"].astype(int).values
        y_val = df_val["requester_received_pizza"].astype(int).values

        # 10. Save to Cache
        print("Saving features to cache...")

        # Helper to save sparse
        def save_sparse_dict(matrix, prefix, split, d):
            d[f"{prefix}_{split}_data"] = matrix.data
            d[f"{prefix}_{split}_indices"] = matrix.indices
            d[f"{prefix}_{split}_indptr"] = matrix.indptr
            d[f"{prefix}_{split}_shape"] = matrix.shape

        rf_save_dict = {}
        save_sparse_dict(rf_train, "rf", "train", rf_save_dict)
        save_sparse_dict(rf_val, "rf", "val", rf_save_dict)
        save_sparse_dict(rf_test, "rf", "test", rf_save_dict)
        np.savez(self.rf_data_path, **rf_save_dict)

        np.savez(
            self.mlp_data_path,
            train_text=sbert_train,
            train_hist=seq_train,
            train_meta=meta_train_mlp,
            val_text=sbert_val,
            val_hist=seq_val,
            val_meta=meta_val_mlp,
            test_text=sbert_test,
            test_hist=seq_test,
            test_meta=meta_test_mlp,
            vocab_size=vocab_size,
        )

        np.savez(self.targets_path, train=y_train, val=y_val)

        # Save Scaler/TFIDF for inference consistency if needed later
        with open(self.tfidf_path, "wb") as f:
            pickle.dump(self.tfidf, f)
        with open(self.scaler_path, "wb") as f:
            pickle.dump(self.scaler, f)

        return {
            "rf": {"train": rf_train, "val": rf_val, "test": rf_test},
            "mlp": {
                "train": {
                    "text": sbert_train,
                    "history": seq_train,
                    "meta": meta_train_mlp,
                },
                "val": {"text": sbert_val, "history": seq_val, "meta": meta_val_mlp},
                "test": {
                    "text": sbert_test,
                    "history": seq_test,
                    "meta": meta_test_mlp,
                },
            },
            "targets": {"train": y_train, "val": y_val},
            "dims": {
                "vocab_size": vocab_size,
                "history_dim": seq_train.shape[1],
                "meta_dim": meta_train_mlp.shape[1],
                "sbert_dim": sbert_train.shape[1],
            },
        }


def create_features(load_cached_data=True):
    factory = FeatureFactory()
    return factory.fit_transform(load_cached_data=load_cached_data)
