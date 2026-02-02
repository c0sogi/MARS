import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sentence_transformers import SentenceTransformer
from library.config import Config
from library.data_loader import load_datasets
from library.utils import set_seed


class FeaturePipeline:
    def __init__(self):
        set_seed(Config.SEED)

        # Stream A: Random Forest Transformers
        self.rf_text_vectorizer = TfidfVectorizer(
            max_features=Config.RF_TFIDF_MAX_FEATURES,
            ngram_range=Config.RF_TFIDF_NGRAM_RANGE,
            stop_words="english",
            sublinear_tf=True,
        )

        # Token pattern matches any non-whitespace string to preserve subreddit names
        self.rf_subreddit_vectorizer = TfidfVectorizer(
            min_df=Config.RF_SUBREDDIT_TFIDF_MIN_DF,
            binary=Config.RF_SUBREDDIT_TFIDF_BINARY,
            token_pattern=r"(?u)\b\w+\b",
            lowercase=False,
        )

        self.rf_imputer = SimpleImputer(strategy="median")

        # Stream B: MLP Transformers
        self.mlp_scaler = StandardScaler()

    def _generate_text_content(self, df):
        """Concatenates title and body for text processing."""
        title = df["request_title"].fillna("")
        body = df["request_text_edit_aware"].fillna("")
        return (title + " " + body).tolist()

    def _generate_subreddit_string(self, df):
        """Converts list of subreddits to space-separated string."""
        # Config.SUBREDDIT_COL is a list of strings.
        return (
            df[Config.SUBREDDIT_COL]
            .apply(lambda x: " ".join(x) if isinstance(x, list) else "")
            .tolist()
        )

    def _engineer_metadata(self, df):
        """Generates full-spectrum numeric metadata."""
        # 1. Start with raw numeric columns defined in Config
        # Ensure columns exist, fill missing with 0 temporarily (imputer handles later for RF)
        meta = df[Config.NUMERIC_COLS].copy()

        # 2. Text Meta-Features
        # Use request_text_edit_aware
        texts = df["request_text_edit_aware"].fillna("").astype(str)
        meta["text_len_char"] = texts.apply(len)
        meta["text_len_word"] = texts.apply(lambda x: len(x.split()))
        meta["text_caps_ratio"] = texts.apply(
            lambda x: sum(1 for c in x if c.isupper()) / len(x) if len(x) > 0 else 0
        )

        # 3. Engineered Ratios
        # Upvote Ratio: Derived from (up+down) and (up-down)
        # up + down = total
        # up - down = diff
        # 2*up = total + diff => up = (total + diff) / 2
        total_votes = meta["requester_upvotes_plus_downvotes_at_request"]
        diff_votes = meta["requester_upvotes_minus_downvotes_at_request"]

        upvotes = (total_votes + diff_votes) / 2

        # Avoid division by zero
        meta["upvote_ratio"] = np.where(
            total_votes > 0, upvotes / total_votes, 0.5  # Default neutral ratio
        )

        return meta

    def _process_rf_features(self, df_train, df_val, df_test):
        print("Generating features for Random Forest Stream...")

        # --- 1. Text TF-IDF ---
        train_text = self._generate_text_content(df_train)
        val_text = self._generate_text_content(df_val)
        test_text = self._generate_text_content(df_test)

        print("Vectorizing Text...")
        X_train_text = self.rf_text_vectorizer.fit_transform(train_text)
        X_val_text = self.rf_text_vectorizer.transform(val_text)
        X_test_text = self.rf_text_vectorizer.transform(test_text)

        # --- 2. Direct Sparse Community Vectorization ---
        train_subs = self._generate_subreddit_string(df_train)
        val_subs = self._generate_subreddit_string(df_val)
        test_subs = self._generate_subreddit_string(df_test)

        print("Vectorizing Subreddits...")
        X_train_subs = self.rf_subreddit_vectorizer.fit_transform(train_subs)
        X_val_subs = self.rf_subreddit_vectorizer.transform(val_subs)
        X_test_subs = self.rf_subreddit_vectorizer.transform(test_subs)

        # --- 3. Metadata ---
        print("Processing Metadata...")
        meta_train = self._engineer_metadata(df_train)
        meta_val = self._engineer_metadata(df_val)
        meta_test = self._engineer_metadata(df_test)

        # Impute
        X_train_meta = self.rf_imputer.fit_transform(meta_train)
        X_val_meta = self.rf_imputer.transform(meta_val)
        X_test_meta = self.rf_imputer.transform(meta_test)

        # --- 4. Combine ---
        print("Stacking RF features...")
        X_train = sp.hstack([X_train_text, X_train_subs, X_train_meta])
        X_val = sp.hstack([X_val_text, X_val_subs, X_val_meta])
        X_test = sp.hstack([X_test_text, X_test_subs, X_test_meta])

        # Targets
        y_train = df_train[Config.TARGET_COL].astype(int).values
        y_val = df_val[Config.TARGET_COL].astype(int).values
        # Test has no target for prediction, but we return dummy or None

        return {
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "X_test": X_test,
        }

    def _process_mlp_features(self, df_train, df_val, df_test):
        print("Generating features for MLP Stream...")

        # Load SBERT model
        print(f"Loading SBERT model: {Config.SBERT_MODEL}")
        sbert = SentenceTransformer(Config.SBERT_MODEL)

        # --- 1. Request Semantics (Text Embedding) ---
        print("Encoding Request Text...")
        train_text = self._generate_text_content(df_train)
        val_text = self._generate_text_content(df_val)
        test_text = self._generate_text_content(df_test)

        # Encode
        emb_train_text = sbert.encode(
            train_text, convert_to_numpy=True, show_progress_bar=False
        )
        emb_val_text = sbert.encode(
            val_text, convert_to_numpy=True, show_progress_bar=False
        )
        emb_test_text = sbert.encode(
            test_text, convert_to_numpy=True, show_progress_bar=False
        )

        # --- 2. Attended History (Sequence of Subreddit Embeddings) ---
        print("Encoding Subreddit History...")

        # Collect all unique subreddits to encode efficiently
        all_subs = set()
        for df in [df_train, df_val, df_test]:
            for sub_list in df[Config.SUBREDDIT_COL]:
                if isinstance(sub_list, list):
                    all_subs.update(sub_list)

        unique_subs = list(all_subs)
        sub_to_idx = {sub: i for i, sub in enumerate(unique_subs)}

        print(f"Unique subreddits found: {len(unique_subs)}")
        if len(unique_subs) > 0:
            sub_embeddings = sbert.encode(
                unique_subs, convert_to_numpy=True, show_progress_bar=False
            )
        else:
            sub_embeddings = np.zeros((1, Config.SBERT_DIM))  # Fallback

        # Helper to create padded sequences
        # Determine max length (cap at 50 or 100 to save memory, though prompt implies full history)
        # Let's check max length in data
        max_len_data = 0
        for df in [df_train, df_val, df_test]:
            max_len_data = max(
                max_len_data,
                df[Config.SUBREDDIT_COL]
                .apply(lambda x: len(x) if isinstance(x, list) else 0)
                .max(),
            )

        MAX_SEQ_LEN = min(max_len_data, 100)  # Cap at 100
        print(f"Max history sequence length set to: {MAX_SEQ_LEN}")

        def create_history_tensor(df):
            n_samples = len(df)
            # Tensor: (N, L, D)
            history_emb = np.zeros(
                (n_samples, MAX_SEQ_LEN, Config.SBERT_DIM), dtype=np.float32
            )
            # Mask: (N, L) - 1 for valid, 0 for padding
            mask = np.zeros((n_samples, MAX_SEQ_LEN), dtype=np.float32)

            for i, row_subs in enumerate(df[Config.SUBREDDIT_COL]):
                if not isinstance(row_subs, list) or len(row_subs) == 0:
                    continue

                # Truncate if necessary
                current_subs = row_subs[:MAX_SEQ_LEN]

                for j, sub in enumerate(current_subs):
                    if sub in sub_to_idx:
                        idx = sub_to_idx[sub]
                        history_emb[i, j, :] = sub_embeddings[idx]
                        mask[i, j] = 1.0

            return history_emb, mask

        hist_train, mask_train = create_history_tensor(df_train)
        hist_val, mask_val = create_history_tensor(df_val)
        hist_test, mask_test = create_history_tensor(df_test)

        # --- 3. Metadata (Arcsinh + Scaled) ---
        print("Processing MLP Metadata...")
        meta_train = self._engineer_metadata(df_train)
        meta_val = self._engineer_metadata(df_val)
        meta_test = self._engineer_metadata(df_test)

        # Handle NaNs before transformation (simple fill)
        meta_train = meta_train.fillna(0)
        meta_val = meta_val.fillna(0)
        meta_test = meta_test.fillna(0)

        # Arcsinh Transform (log-like behavior for 0 and negative values)
        meta_train_asinh = np.arcsinh(meta_train)
        meta_val_asinh = np.arcsinh(meta_val)
        meta_test_asinh = np.arcsinh(meta_test)

        # Scale
        X_train_meta = self.mlp_scaler.fit_transform(meta_train_asinh)
        X_val_meta = self.mlp_scaler.transform(meta_val_asinh)
        X_test_meta = self.mlp_scaler.transform(meta_test_asinh)

        return {
            "text_train": emb_train_text,
            "hist_train": hist_train,
            "mask_train": mask_train,
            "meta_train": X_train_meta,
            "text_val": emb_val_text,
            "hist_val": hist_val,
            "mask_val": mask_val,
            "meta_val": X_val_meta,
            "text_test": emb_test_text,
            "hist_test": hist_test,
            "mask_test": mask_test,
            "meta_test": X_test_meta,
            "y_train": df_train[Config.TARGET_COL].astype(int).values,
            "y_val": df_val[Config.TARGET_COL].astype(int).values,
        }

    def run(self, load_cached_data=True):
        """
        Main execution method.
        Checks for cached features. If not found or forced reload, computes features and saves them.
        """
        # Ensure working directory exists
        os.makedirs(Config.WORKING_DIR, exist_ok=True)

        rf_path = Config.RF_FEATURES_PATH
        mlp_path = Config.MLP_FEATURES_PATH

        # Check cache
        if load_cached_data and os.path.exists(rf_path) and os.path.exists(mlp_path):
            print("Loading cached features...")
            try:
                # Load RF (Sparse matrices need careful loading if saved via savez)
                # We save sparse matrices as components in npz or use pickle.
                # Constraint: "Prohibited: Do NOT use pickle."
                # Solution: Save sparse matrices using scipy.sparse.save_npz?
                # But we have multiple matrices.
                # We will save RF data as a dictionary of arrays in npz, converting sparse to csc/csr components or
                # just save them as separate sparse npz files if needed.
                # Actually, scipy.sparse.save_npz saves ONE matrix.
                # To adhere to "Prohibited pickle", we can save indices/data/indptr arrays of sparse matrix in .npz
                # Or simpler: Re-compute RF is fast. Re-compute MLP (SBERT) is slow.
                # Let's try to load. If complex, we recompute RF and only cache MLP.
                # However, requirement says "Save the result to the cache".

                # Loading MLP features (Dense arrays)
                mlp_data = np.load(mlp_path)
                mlp_features = {k: mlp_data[k] for k in mlp_data.files}

                # Loading RF features
                # Since saving multiple sparse matrices in one file without pickle is tricky,
                # we will assume the save method used scipy.sparse.save_npz for individual files
                # OR we stored them as dense if small enough (unlikely).
                # Strategy: Save RF parts (X_train, X_val, X_test) as separate .npz files using scipy.sparse.save_npz
                # and y as .npy.
                # To keep it simple in one function call, let's look at the save block below.
                # If we implemented a custom save/load for sparse in npz, we use that.

                # Custom loader for RF sparse bundle
                rf_loader = np.load(rf_path)
                rf_features = {}
                for split in ["train", "val", "test"]:
                    # Reconstruct CSR
                    data = rf_loader[f"X_{split}_data"]
                    indices = rf_loader[f"X_{split}_indices"]
                    indptr = rf_loader[f"X_{split}_indptr"]
                    shape = rf_loader[f"X_{split}_shape"]
                    rf_features[f"X_{split}"] = sp.csr_matrix(
                        (data, indices, indptr), shape=shape
                    )

                rf_features["y_train"] = rf_loader["y_train"]
                rf_features["y_val"] = rf_loader["y_val"]

                print("Cache loaded successfully.")
                return rf_features, mlp_features

            except Exception as e:
                print(f"Failed to load cache: {e}. Recomputing...")

        # Compute from scratch
        print("Computing features from scratch...")
        df_train, df_val, df_test = load_datasets()

        rf_features = self._process_rf_features(df_train, df_val, df_test)
        mlp_features = self._process_mlp_features(df_train, df_val, df_test)

        # Save to Cache
        print("Saving features to cache...")

        # Save MLP (all dense)
        np.savez_compressed(mlp_path, **mlp_features)

        # Save RF (Sparse handling)
        # Deconstruct sparse matrices to arrays for standard npz saving
        rf_save_dict = {}
        for split in ["train", "val", "test"]:
            mat = rf_features[f"X_{split}"].tocsr()
            rf_save_dict[f"X_{split}_data"] = mat.data
            rf_save_dict[f"X_{split}_indices"] = mat.indices
            rf_save_dict[f"X_{split}_indptr"] = mat.indptr
            rf_save_dict[f"X_{split}_shape"] = mat.shape

        rf_save_dict["y_train"] = rf_features["y_train"]
        rf_save_dict["y_val"] = rf_features["y_val"]

        np.savez_compressed(rf_path, **rf_save_dict)

        return rf_features, mlp_features
