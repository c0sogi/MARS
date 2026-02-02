import os
import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from library.config import (
    WORKING_DIR,
    RANDOM_SEED,
    SBERT_MODEL_NAME,
    SBERT_EMBEDDING_DIM,
    SBERT_BATCH_SIZE,
    LSA_COMPONENTS,
    TFIDF_MAX_FEATURES,
    TFIDF_NGRAM_RANGE,
    MAX_HISTORY_LENGTH,
    DEVICE,
)
from library.utils import seed_everything


class FeatureEngineer:
    def __init__(self):
        seed_everything(RANDOM_SEED)
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy="median")
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            ngram_range=TFIDF_NGRAM_RANGE,
            stop_words="english",
            sublinear_tf=True,
        )
        self.lsa_pipeline = None  # Will hold (vectorizer, svd)

    def _get_paths(self, feature_name):
        """Helper to get cache paths for a specific feature type."""
        return {
            "train": os.path.join(WORKING_DIR, f"{feature_name}_train.npz"),
            "val": os.path.join(WORKING_DIR, f"{feature_name}_val.npz"),
            "test": os.path.join(WORKING_DIR, f"{feature_name}_test.npz"),
            "model": os.path.join(WORKING_DIR, f"{feature_name}_model.pkl"),
        }

    def _save_npz(self, data, path):
        np.savez_compressed(path, data=data)

    def _load_npz(self, path):
        return np.load(path)["data"]

    def extract_metadata(self, df, split="train", fit_scaler=False):
        """
        Extracts and processes numerical metadata.
        Returns two versions:
        1. Raw/Imputed for RF (DataFrame)
        2. Arcsinh+Scaled for MLP (NP Array)
        """
        # 1. Base Feature Selection
        # Identify numeric columns suitable for features
        numeric_cols = [
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

        # Filter to available columns
        available_cols = [c for c in numeric_cols if c in df.columns]
        meta_df = df[available_cols].copy()

        # 2. Feature Engineering
        # Ratios (avoid division by zero)
        if (
            "requester_upvotes_plus_downvotes_at_request" in df.columns
            and "requester_upvotes_minus_downvotes_at_request" in df.columns
        ):
            # Reconstruct upvotes and downvotes roughly
            # net = up - down, total = up + down => 2*up = total + net
            total = df["requester_upvotes_plus_downvotes_at_request"]
            net = df["requester_upvotes_minus_downvotes_at_request"]
            upvotes = (total + net) / 2

            meta_df["upvote_ratio"] = upvotes / (total + 1e-5)

        # Text Meta-Features
        if "request_text_edit_aware" in df.columns:
            texts = df["request_text_edit_aware"].fillna("").astype(str)
            meta_df["text_len_char"] = texts.apply(len)
            meta_df["text_len_word"] = texts.apply(lambda x: len(x.split()))
            meta_df["caps_ratio"] = texts.apply(
                lambda x: sum(1 for c in x if c.isupper()) / (len(x) + 1e-5)
            )

        # 3. Processing for RF (Imputation only)
        # We perform imputation on the copy to avoid SettingWithCopy warnings on original df
        if fit_scaler:
            self.imputer.fit(meta_df)

        rf_features_arr = self.imputer.transform(meta_df)
        rf_features_df = pd.DataFrame(
            rf_features_arr, columns=meta_df.columns, index=df.index
        )

        # 4. Processing for MLP (Arcsinh + Scaling)
        # Arcsinh handles the heavy tails better than log for values that can be 0 or negative
        mlp_features_arr = np.arcsinh(rf_features_arr)

        if fit_scaler:
            self.scaler.fit(mlp_features_arr)

        mlp_features_norm = self.scaler.transform(mlp_features_arr)

        return rf_features_df, mlp_features_norm

    def generate_tfidf(self, train_df, val_df, test_df, load_cached=True):
        paths = self._get_paths("tfidf")

        if load_cached and all(os.path.exists(p) for p in paths.values()):
            print("Loading cached TF-IDF features...")
            return (
                self._load_npz(paths["train"]),
                self._load_npz(paths["val"]),
                self._load_npz(paths["test"]),
            )

        print("Generating TF-IDF features...")

        # Prepare text: Title + Body
        def get_text(df):
            t = (
                df["request_title"].fillna("")
                + " "
                + df["request_text_edit_aware"].fillna("")
            )
            return t.tolist()

        train_text = get_text(train_df)
        val_text = get_text(val_df)
        test_text = get_text(test_df)

        # Fit and Transform
        X_train = (
            self.tfidf_vectorizer.fit_transform(train_text).astype(np.float32).toarray()
        )
        X_val = self.tfidf_vectorizer.transform(val_text).astype(np.float32).toarray()
        X_test = self.tfidf_vectorizer.transform(test_text).astype(np.float32).toarray()

        # Cache
        self._save_npz(X_train, paths["train"])
        self._save_npz(X_val, paths["val"])
        self._save_npz(X_test, paths["test"])
        joblib.dump(self.tfidf_vectorizer, paths["model"])

        return X_train, X_val, X_test

    def generate_lsa(self, train_df, val_df, test_df, load_cached=True):
        """
        Generates Latent Semantic Analysis features from user subreddit history.
        Used for the Random Forest model.
        """
        paths = self._get_paths("lsa")

        if load_cached and all(os.path.exists(p) for p in paths.values()):
            print("Loading cached LSA features...")
            return (
                self._load_npz(paths["train"]),
                self._load_npz(paths["val"]),
                self._load_npz(paths["test"]),
            )

        print("Generating LSA features from history...")

        def process_subreddits(df):
            # Convert list of subreddits to space-separated string
            return (
                df["requester_subreddits_at_request"]
                .apply(
                    lambda x: (
                        " ".join(x)
                        if isinstance(x, list) and len(x) > 0
                        else "placeholder_empty"
                    )
                )
                .tolist()
            )

        train_docs = process_subreddits(train_df)
        val_docs = process_subreddits(val_df)
        test_docs = process_subreddits(test_df)

        # Pipeline: TF-IDF on subreddits -> SVD
        # We use a simple tokenizer that splits by space
        vec = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", min_df=2)
        svd = TruncatedSVD(n_components=LSA_COMPONENTS, random_state=RANDOM_SEED)

        # Fit on train
        train_dtm = vec.fit_transform(train_docs)
        train_lsa = svd.fit_transform(train_dtm).astype(np.float32)

        # Transform others
        val_dtm = vec.transform(val_docs)
        val_lsa = svd.transform(val_dtm).astype(np.float32)

        test_dtm = vec.transform(test_docs)
        test_lsa = svd.transform(test_dtm).astype(np.float32)

        # Cache
        self._save_npz(train_lsa, paths["train"])
        self._save_npz(val_lsa, paths["val"])
        self._save_npz(test_lsa, paths["test"])
        joblib.dump((vec, svd), paths["model"])

        return train_lsa, val_lsa, test_lsa

    def generate_embeddings(self, train_df, val_df, test_df, load_cached=True):
        """
        Generates SBERT embeddings for:
        1. Request Text (Title + Body) -> (N, 384)
        2. User History (Sequence of Subreddits) -> (N, Max_Len, 384)

        Also computes Alignment Score (Cosine Sim between Request and History Centroid).
        """
        # Paths for Request Embeddings
        req_paths = self._get_paths("req_embed")
        # Paths for History Embeddings
        hist_paths = self._get_paths("hist_embed")
        # Paths for Alignment Scores
        align_paths = self._get_paths("alignment")

        all_paths = (
            list(req_paths.values())[:-1]
            + list(hist_paths.values())[:-1]
            + list(align_paths.values())[:-1]
        )

        if load_cached and all(os.path.exists(p) for p in all_paths):
            print("Loading cached SBERT embeddings and alignment scores...")
            return (
                self._load_npz(req_paths["train"]),
                self._load_npz(hist_paths["train"]),
                self._load_npz(align_paths["train"]),
                self._load_npz(req_paths["val"]),
                self._load_npz(hist_paths["val"]),
                self._load_npz(align_paths["val"]),
                self._load_npz(req_paths["test"]),
                self._load_npz(hist_paths["test"]),
                self._load_npz(align_paths["test"]),
            )

        print("Generating SBERT embeddings (this may take a while)...")
        model = SentenceTransformer(SBERT_MODEL_NAME, device=DEVICE)

        # --- 1. Request Embeddings ---
        def get_text(df):
            return (
                df["request_title"].fillna("")
                + " "
                + df["request_text_edit_aware"].fillna("")
            ).tolist()

        print("Embedding requests...")
        req_train = model.encode(
            get_text(train_df), batch_size=SBERT_BATCH_SIZE, show_progress_bar=False
        )
        req_val = model.encode(
            get_text(val_df), batch_size=SBERT_BATCH_SIZE, show_progress_bar=False
        )
        req_test = model.encode(
            get_text(test_df), batch_size=SBERT_BATCH_SIZE, show_progress_bar=False
        )

        # --- 2. History Embeddings ---
        print("Embedding history...")
        # Strategy: Identify unique subreddits, embed them, then map back to create sequences

        # Collect all subreddits
        all_subs = set()
        for df in [train_df, val_df, test_df]:
            for sub_list in df["requester_subreddits_at_request"]:
                if isinstance(sub_list, list):
                    all_subs.update(sub_list)

        all_subs = list(all_subs)
        if not all_subs:
            # Handle edge case where no subreddits exist
            sub_to_vec = {}
            embedding_dim = SBERT_EMBEDDING_DIM
        else:
            print(f"Unique subreddits to embed: {len(all_subs)}")
            sub_vecs = model.encode(
                all_subs, batch_size=SBERT_BATCH_SIZE, show_progress_bar=False
            )
            sub_to_vec = {sub: vec for sub, vec in zip(all_subs, sub_vecs)}
            embedding_dim = sub_vecs.shape[1]

        def process_history(df):
            n_samples = len(df)
            # Tensor for MLP: (N, Max_Len, Dim)
            hist_tensor = np.zeros(
                (n_samples, MAX_HISTORY_LENGTH, embedding_dim), dtype=np.float32
            )
            # Vector for Alignment: (N, Dim) - Centroid
            centroids = np.zeros((n_samples, embedding_dim), dtype=np.float32)

            for idx, sub_list in enumerate(df["requester_subreddits_at_request"]):
                if not isinstance(sub_list, list) or len(sub_list) == 0:
                    continue

                # Truncate to max length or take recent?
                # Taking most recent (last ones in list usually? Data doesn't specify order, assume random or chronological)
                # We'll take the first N found in list for simplicity/consistency
                current_vecs = []
                for sub in sub_list[:MAX_HISTORY_LENGTH]:
                    if sub in sub_to_vec:
                        current_vecs.append(sub_to_vec[sub])

                if not current_vecs:
                    continue

                current_vecs = np.array(current_vecs)

                # Fill Tensor
                n_items = min(len(current_vecs), MAX_HISTORY_LENGTH)
                hist_tensor[idx, :n_items, :] = current_vecs[:n_items]

                # Compute Centroid
                centroids[idx] = np.mean(current_vecs, axis=0)

            return hist_tensor, centroids

        hist_train, cent_train = process_history(train_df)
        hist_val, cent_val = process_history(val_df)
        hist_test, cent_test = process_history(test_df)

        # --- 3. Alignment Score ---
        # Cosine Similarity between Request and History Centroid
        def calc_align(req, cent):
            # Dot product of normalized vectors
            # SBERT output is usually normalized, but let's ensure
            norm_req = np.linalg.norm(req, axis=1, keepdims=True) + 1e-9
            norm_cent = np.linalg.norm(cent, axis=1, keepdims=True) + 1e-9

            req_n = req / norm_req
            cent_n = cent / norm_cent

            # Element-wise dot product sum
            sim = np.sum(req_n * cent_n, axis=1).reshape(-1, 1)
            return sim.astype(np.float32)

        align_train = calc_align(req_train, cent_train)
        align_val = calc_align(req_val, cent_val)
        align_test = calc_align(req_test, cent_test)

        # Cache everything
        self._save_npz(req_train, req_paths["train"])
        self._save_npz(req_val, req_paths["val"])
        self._save_npz(req_test, req_paths["test"])

        self._save_npz(hist_train, hist_paths["train"])
        self._save_npz(hist_val, hist_paths["val"])
        self._save_npz(hist_test, hist_paths["test"])

        self._save_npz(align_train, align_paths["train"])
        self._save_npz(align_val, align_paths["val"])
        self._save_npz(align_test, align_paths["test"])

        return (
            req_train,
            hist_train,
            align_train,
            req_val,
            hist_val,
            align_val,
            req_test,
            hist_test,
            align_test,
        )

    def process_features(self, train_df, val_df, test_df, load_cached=True):
        """
        Orchestrates the feature generation pipeline.
        Returns:
            data_rf: Dictionary containing X_train, y_train, X_val, y_val, X_test for Random Forest.
            data_mlp: Dictionary containing inputs for MLP (Request Emb, History Emb, Metadata).
        """
        print("Starting Feature Engineering Pipeline...")

        # 1. Metadata
        print("Processing Metadata...")
        meta_rf_train, meta_mlp_train = self.extract_metadata(
            train_df, "train", fit_scaler=True
        )
        meta_rf_val, meta_mlp_val = self.extract_metadata(
            val_df, "val", fit_scaler=False
        )
        meta_rf_test, meta_mlp_test = self.extract_metadata(
            test_df, "test", fit_scaler=False
        )

        # 2. TF-IDF (RF)
        tfidf_train, tfidf_val, tfidf_test = self.generate_tfidf(
            train_df, val_df, test_df, load_cached
        )

        # 3. LSA (RF)
        lsa_train, lsa_val, lsa_test = self.generate_lsa(
            train_df, val_df, test_df, load_cached
        )

        # 4. Embeddings & Alignment (MLP & RF)
        (
            req_train,
            hist_train,
            align_train,
            req_val,
            hist_val,
            align_val,
            req_test,
            hist_test,
            align_test,
        ) = self.generate_embeddings(train_df, val_df, test_df, load_cached)

        # --- Assemble RF Data ---
        # Concatenate: Metadata (DF) + TFIDF (NP) + LSA (NP) + Alignment (NP)
        def assemble_rf(meta, tfidf, lsa, align):
            # Convert meta to numpy
            m_arr = meta.values.astype(np.float32)
            combined = np.hstack([m_arr, tfidf, lsa, align])
            return combined

        X_train_rf = assemble_rf(meta_rf_train, tfidf_train, lsa_train, align_train)
        X_val_rf = assemble_rf(meta_rf_val, tfidf_val, lsa_val, align_val)
        X_test_rf = assemble_rf(meta_rf_test, tfidf_test, lsa_test, align_test)

        # Targets
        y_train = train_df["requester_received_pizza"].astype(int).values
        y_val = val_df["requester_received_pizza"].astype(int).values

        data_rf = {
            "X_train": X_train_rf,
            "y_train": y_train,
            "X_val": X_val_rf,
            "y_val": y_val,
            "X_test": X_test_rf,
        }

        # --- Assemble MLP Data ---
        # MLP expects dictionary of inputs
        data_mlp = {
            "train": {
                "req_emb": req_train,
                "hist_emb": hist_train,
                "meta": meta_mlp_train,
                "y": y_train,
            },
            "val": {
                "req_emb": req_val,
                "hist_emb": hist_val,
                "meta": meta_mlp_val,
                "y": y_val,
            },
            "test": {"req_emb": req_test, "hist_emb": hist_test, "meta": meta_mlp_test},
        }

        print("Feature Engineering Complete.")
        print(f"RF Feature Shape: {X_train_rf.shape}")
        print(f"MLP Request Emb Shape: {req_train.shape}")
        print(f"MLP History Emb Shape: {hist_train.shape}")

        return data_rf, data_mlp
