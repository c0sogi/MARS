import os
import numpy as np
import pandas as pd
import torch
import joblib
from scipy import sparse
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from library.config import Config
from library.text_utils import DualTFIDFVectorizer
from library.topic_utils import compute_topic_alignment


class FeatureBuilder:
    """
    Stateful feature engineering pipeline.
    Manages vectorizers, scalers, and imputers across train/val/test splits.
    """

    def __init__(self):
        self.tfidf_vectorizer = DualTFIDFVectorizer()
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy="median")

        # Persistence paths
        self.tfidf_path = os.path.join(Config.WORKING_DIR, "tfidf_vectorizer.pkl")
        self.scaler_path = os.path.join(Config.WORKING_DIR, "meta_scaler.pkl")
        self.imputer_path = os.path.join(Config.WORKING_DIR, "meta_imputer.pkl")

        # State flags
        self.is_tfidf_fitted = False
        self.is_scaler_fitted = False
        self.is_imputer_fitted = False

    def _save_state(self):
        """Saves fitted transformers to disk."""
        joblib.dump(self.tfidf_vectorizer, self.tfidf_path)
        joblib.dump(self.scaler, self.scaler_path)
        joblib.dump(self.imputer, self.imputer_path)

    def _load_state(self):
        """Loads fitted transformers from disk if they exist."""
        if os.path.exists(self.tfidf_path):
            self.tfidf_vectorizer = joblib.load(self.tfidf_path)
            self.is_tfidf_fitted = True
        if os.path.exists(self.scaler_path):
            self.scaler = joblib.load(self.scaler_path)
            self.is_scaler_fitted = True
        if os.path.exists(self.imputer_path):
            self.imputer = joblib.load(self.imputer_path)
            self.is_imputer_fitted = True

    def extract_tabular_features(self, df, split_name, load_cached_data=True):
        """
        Generates raw and engineered tabular features.
        Returns:
            np.ndarray: Matrix of shape (N, D)
        """
        cache_file = os.path.join(
            Config.WORKING_DIR, f"tabular_features_{split_name}.npy"
        )

        if load_cached_data and os.path.exists(cache_file):
            print(f"Loading cached tabular features for {split_name}...")
            return np.load(cache_file)

        print(f"Extracting tabular features for {split_name}...")

        # 1. Raw Numerical Features
        # Ensure columns exist, fill missing with 0 for extraction (imputation happens later)
        raw_features = df[Config.NUMERIC_META_COLS].fillna(0).values

        # 2. Engineered Ratios
        # Upvote Ratio: Up / (Up + Down)
        up = df["requester_upvotes_plus_downvotes_at_request"].fillna(0)
        diff = df["requester_upvotes_minus_downvotes_at_request"].fillna(0)
        # derived: up + down = total; up - down = diff => 2*up = total + diff => up = (total+diff)/2
        # But we have 'requester_upvotes_plus_downvotes_at_request' which is sum.
        # Let's use the sum column directly as denominator.
        # If sum is 0, ratio is 0.5 (neutral) or 0.

        # Approximate upvotes from sum and diff
        # sum = u + d, diff = u - d => u = (sum + diff) / 2
        approx_up = (up + diff) / 2

        with np.errstate(divide="ignore", invalid="ignore"):
            upvote_ratio = approx_up / up
            upvote_ratio[up == 0] = 0.5  # Default to neutral

        # Text Stats
        text_col = df[Config.TEXT_COL_BODY].fillna("").astype(str)
        text_len = text_col.apply(len).values

        def get_caps_ratio(s):
            if len(s) == 0:
                return 0.0
            return sum(1 for c in s if c.isupper()) / len(s)

        caps_ratio = text_col.apply(get_caps_ratio).values

        # 3. Lexicon Densities
        lexicon_features = []
        for category, words in Config.LEXICONS.items():
            # Simple count of any word in the list (case-insensitive)
            # This is slow but fine for dataset size < 5000
            def count_lexicon(s):
                s_lower = s.lower()
                return sum(s_lower.count(w) for w in words)

            counts = text_col.apply(count_lexicon).values
            lexicon_features.append(counts)

        lexicon_matrix = np.column_stack(lexicon_features)

        # Combine All
        # Shape: [Raw (9), UpRatio (1), Len (1), Caps (1), Lexicons (3)]
        features = np.column_stack(
            [raw_features, upvote_ratio, text_len, caps_ratio, lexicon_matrix]
        )

        # Save to cache
        np.save(cache_file, features)
        return features

    def prepare_rf_inputs(
        self, df, split_name, embedder, aligner, load_cached_data=True
    ):
        """
        Prepares features for Random Forest (Stream A).
        Combines Tabular, TF-IDF, and Topic Alignment.
        Handles fitting on 'train' and transforming on 'val'/'test'.
        """
        cache_file = os.path.join(Config.WORKING_DIR, f"rf_data_{split_name}.npz")

        if load_cached_data and os.path.exists(cache_file):
            print(f"Loading cached RF data for {split_name}...")
            data = np.load(cache_file, allow_pickle=True)
            # Handle potential None for y in test set
            y = data["y"]
            if y.shape == ():  # 0-d array check
                y = None
            return data["X"], y

        print(f"Preparing RF inputs for {split_name}...")

        # 1. Tabular Features
        tabular = self.extract_tabular_features(df, split_name, load_cached_data)

        # 2. Topic Alignment Features
        topic_data = compute_topic_alignment(
            df, split_name, embedder, aligner, load_cached_data
        )
        # We use alignment score (N,) and maybe topic dists.
        # Let's use Alignment Score + Request Topic Dist.
        # History topic dist is less direct for the specific request, but alignment captures the delta.
        align_score = topic_data["alignment_score"].reshape(-1, 1)
        req_topic_dist = topic_data["request_topic_dist"]

        # 3. TF-IDF Features
        # Manage State
        if split_name == "train":
            print("Fitting TF-IDF Vectorizer...")
            self.tfidf_vectorizer.fit(df)
            self.is_tfidf_fitted = True
        else:
            if not self.is_tfidf_fitted:
                self._load_state()
                if not self.is_tfidf_fitted:
                    raise RuntimeError(
                        "TF-IDF vectorizer not fitted. Run 'train' split first."
                    )

        title_tfidf, body_tfidf = self.tfidf_vectorizer.transform(df)

        # 4. Concatenate
        # Convert sparse to dense for concatenation (dataset is small enough)
        # If OOM issues arise, use scipy.sparse.hstack
        X_parts = [
            tabular,
            align_score,
            req_topic_dist,
            title_tfidf.toarray(),
            body_tfidf.toarray(),
        ]
        X = np.hstack(X_parts)

        # 5. Imputation (Tabular parts might have NaNs, TF-IDF is 0-filled)
        if split_name == "train":
            print("Fitting Imputer...")
            self.imputer.fit(X)
            self.is_imputer_fitted = True
            self._save_state()  # Save state after fitting everything
        else:
            if not self.is_imputer_fitted:
                self._load_state()

        X = self.imputer.transform(X)

        # 6. Target
        if "requester_received_pizza" in df.columns:
            y = df["requester_received_pizza"].values.astype(int)
        else:
            y = None

        # Save to cache
        # np.savez handles None for y by saving it as object, but cleaner to save placeholder if None
        save_y = y if y is not None else np.array(np.nan)
        np.savez(cache_file, X=X, y=save_y)

        return X, y

    def prepare_mlp_inputs(self, df, split_name, embedder, load_cached_data=True):
        """
        Prepares tensors for MLP (Stream B).
        Returns dictionary of tensors.
        """
        cache_file = os.path.join(Config.WORKING_DIR, f"mlp_data_{split_name}.pt")

        if load_cached_data and os.path.exists(cache_file):
            print(f"Loading cached MLP data for {split_name}...")
            return torch.load(cache_file)

        print(f"Preparing MLP inputs for {split_name}...")

        # 1. SBERT Embeddings
        req_emb = embedder.encode_requests(df, split_name, load_cached_data)
        hist_emb = embedder.encode_history(df, split_name, load_cached_data)

        # 2. Tabular Features (Normalized)
        tabular = self.extract_tabular_features(df, split_name, load_cached_data)

        # Apply Arcsinh
        tabular = np.arcsinh(tabular)

        # Apply StandardScaler
        if split_name == "train":
            print("Fitting Scaler...")
            self.scaler.fit(tabular)
            self.is_scaler_fitted = True
            self._save_state()
        else:
            if not self.is_scaler_fitted:
                self._load_state()
                if not self.is_scaler_fitted:
                    raise RuntimeError("Scaler not fitted. Run 'train' split first.")

        tabular = self.scaler.transform(tabular)

        # 3. Convert to Tensors
        data_dict = {
            "request_emb": torch.tensor(req_emb, dtype=torch.float32),
            "history_emb": torch.tensor(hist_emb, dtype=torch.float32),
            "meta_features": torch.tensor(tabular, dtype=torch.float32),
        }

        if "requester_received_pizza" in df.columns:
            y = df["requester_received_pizza"].values.astype(int)
            data_dict["y"] = torch.tensor(
                y, dtype=torch.float32
            )  # Float for BCEWithLogitsLoss
        else:
            data_dict["y"] = None

        # Save to cache
        torch.save(data_dict, cache_file)

        return data_dict
