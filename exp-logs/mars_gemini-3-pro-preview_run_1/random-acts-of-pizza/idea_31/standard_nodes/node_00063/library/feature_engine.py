import os
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sentence_transformers import SentenceTransformer
from library import config, data_loader, neural_net


class FeaturePipeline:
    def __init__(self):
        # Stream A (RF) Tools
        self.tfidf = TfidfVectorizer(
            max_features=config.TFIDF_VOCAB, stop_words="english", dtype=np.float32
        )
        self.imputer_rf = SimpleImputer(strategy="median")

        # Stream B (NN) Tools
        self.scaler_nn = StandardScaler()

        # Shared Tools
        self.sbert = SentenceTransformer(config.SBERT_MODEL)

        # State
        self.top_k_subreddits = []
        self.numeric_cols = []
        self.fitted = False

    def fit(self, df):
        """
        Fits the feature engineering pipeline on the training data.
        """
        print("Fitting FeaturePipeline on training data...")

        # 1. Identify Numeric Columns
        # Exclude targets, IDs, and text
        exclude_cols = [
            "requester_received_pizza",
            "request_id",
            "source_file",
            "request_text",
            "request_title",
            "request_text_edit_aware",
            "requester_subreddits_at_request",
            "giver_username_if_known",
            "requester_username",
            "requester_user_flair",
            "post_was_edited",
        ]
        self.numeric_cols = [
            c
            for c in df.select_dtypes(include=[np.number]).columns
            if c not in exclude_cols and not c.endswith("_at_retrieval")
        ]

        # 2. Top-K Subreddits (Stream A)
        # Flatten all subreddits lists
        all_subs = [
            sub
            for sub_list in df["requester_subreddits_at_request"]
            for sub in sub_list
        ]
        counts = Counter(all_subs)
        self.top_k_subreddits = [
            sub for sub, _ in counts.most_common(config.TOP_K_SUBREDDITS)
        ]

        # 3. TF-IDF (Stream A)
        # Concatenate Title + Body
        text_data = (
            df["request_title"].fillna("")
            + " "
            + df["request_text_edit_aware"].fillna("")
        ).tolist()
        self.tfidf.fit(text_data)

        # 4. Metadata Statistics
        meta_data = df[self.numeric_cols].values

        # RF: Simple Imputation
        self.imputer_rf.fit(meta_data)

        # NN: Arcsinh + Scaling
        # We impute with 0 for the NN pipeline before arcsinh to handle NaNs safely
        meta_filled = df[self.numeric_cols].fillna(0).values
        if config.USE_ARCSINH_TRANSFORM:
            meta_transformed = np.arcsinh(meta_filled)
        else:
            meta_transformed = meta_filled
        self.scaler_nn.fit(meta_transformed)

        self.fitted = True

    def _compute_common_artifacts(self, df):
        """
        Computes expensive artifacts required by both RF and NN streams:
        - SBERT Embeddings (Title, Body, History)
        - Alignment Scalars (Cosine Similarity)
        """
        # Extract raw text lists
        titles = df["request_title"].fillna("").astype(str).tolist()
        bodies = df["request_text_edit_aware"].fillna("").astype(str).tolist()
        histories = df["requester_subreddits_at_request"].tolist()

        # Generate Embeddings using helper from neural_net library (or direct usage)
        # We use the instance self.sbert directly to avoid reloading model
        title_emb = neural_net.get_sbert_embeddings(titles, self.sbert)
        body_emb = neural_net.get_sbert_embeddings(bodies, self.sbert)
        history_emb, history_mask = neural_net.process_history(histories, self.sbert)

        # Compute Dual-View Alignment Scalars
        # 1. Compute Mean History Vector
        # (N, Seq, Emb) * (N, Seq, 1) -> Sum over Seq -> (N, Emb)
        mask_expanded = history_mask[:, :, np.newaxis]
        valid_sum = np.sum(history_emb * mask_expanded, axis=1)
        valid_count = np.sum(mask_expanded, axis=1)
        valid_count = np.maximum(valid_count, 1)  # Avoid div by zero
        mean_history = valid_sum / valid_count

        # 2. Cosine Similarity
        def cosine_sim(a, b):
            norm_a = np.linalg.norm(a, axis=1)
            norm_b = np.linalg.norm(b, axis=1)
            dot = np.sum(a * b, axis=1)
            # Avoid div by zero
            denom = np.maximum(norm_a * norm_b, 1e-9)
            sim = dot / denom
            return sim[:, np.newaxis]  # (N, 1)

        topic_sim = cosine_sim(title_emb, mean_history)
        narrative_sim = cosine_sim(body_emb, mean_history)
        alignment = np.hstack([topic_sim, narrative_sim])

        return {
            "title_emb": title_emb,
            "body_emb": body_emb,
            "history_emb": history_emb,
            "history_mask": history_mask,
            "alignment": alignment,
        }

    def transform_rf(self, df, artifacts):
        """
        Generates features for the Random Forest (Stream A).
        Returns a dense numpy array.
        """
        if not self.fitted:
            raise ValueError("Pipeline must be fitted before transformation.")

        print("Generating Random Forest features...")

        # 1. TF-IDF Features
        text_data = (
            df["request_title"].fillna("")
            + " "
            + df["request_text_edit_aware"].fillna("")
        ).tolist()
        tfidf_feats = self.tfidf.transform(text_data).toarray()

        # 2. Top-K Subreddit Flags
        n_samples = len(df)
        top_k_feats = np.zeros(
            (n_samples, len(self.top_k_subreddits)), dtype=np.float32
        )

        # Optimized lookup
        top_k_set = set(self.top_k_subreddits)
        sub_to_idx = {sub: i for i, sub in enumerate(self.top_k_subreddits)}

        for i, subs in enumerate(df["requester_subreddits_at_request"]):
            # Intersection of user subs and top-k subs
            current_subs = set(subs)
            common = current_subs.intersection(top_k_set)
            for sub in common:
                top_k_feats[i, sub_to_idx[sub]] = 1.0

        # 3. Metadata (Raw + Imputed)
        meta_raw = df[self.numeric_cols].values
        meta_imputed = self.imputer_rf.transform(meta_raw)

        # 4. Alignment Scalars
        alignment = artifacts["alignment"]

        # Concatenate all features
        X = np.hstack([tfidf_feats, top_k_feats, alignment, meta_imputed])
        return X.astype(np.float32)

    def transform_nn(self, df, artifacts):
        """
        Generates features for the Neural Network (Stream B).
        Returns a dictionary of numpy arrays/tensors.
        """
        if not self.fitted:
            raise ValueError("Pipeline must be fitted before transformation.")

        print("Generating Neural Network features...")

        # 1. Metadata (Arcsinh + Scaled)
        meta_raw = df[self.numeric_cols].fillna(0).values
        if config.USE_ARCSINH_TRANSFORM:
            meta_transformed = np.arcsinh(meta_raw)
        else:
            meta_transformed = meta_raw

        meta_scaled = self.scaler_nn.transform(meta_transformed)

        # 2. Labels
        labels = None
        if "requester_received_pizza" in df.columns:
            labels = df["requester_received_pizza"].astype(int).values

        # Construct dictionary
        data_dict = {
            "title_emb": artifacts["title_emb"],
            "body_emb": artifacts["body_emb"],
            "history_emb": artifacts["history_emb"],
            "history_mask": artifacts["history_mask"],
            "meta": meta_scaled.astype(np.float32),
            "alignment": artifacts["alignment"].astype(np.float32),
        }

        if labels is not None:
            data_dict["labels"] = labels

        return data_dict


def run_feature_pipeline(load_cached_data=True):
    """
    Orchestrates the feature generation process.
    Checks cache, otherwise loads raw data, fits pipeline, transforms, and saves.

    Returns:
        (X_rf_train, data_nn_train), (X_rf_val, data_nn_val), (X_rf_test, data_nn_test)
    """
    cache_dir = config.WORKING_DIR
    splits = ["train", "val", "test"]

    # Check cache existence
    all_cached = True
    for split in splits:
        rf_path = os.path.join(cache_dir, f"{split}_rf_features.npz")
        nn_path = os.path.join(cache_dir, f"{split}_nn_features.npz")
        if not (os.path.exists(rf_path) and os.path.exists(nn_path)):
            all_cached = False
            break

    if load_cached_data and all_cached:
        print("Loading processed features from cache...")
        results = {}
        for split in splits:
            rf_path = os.path.join(cache_dir, f"{split}_rf_features.npz")
            nn_path = os.path.join(cache_dir, f"{split}_nn_features.npz")

            # Load RF
            with np.load(rf_path) as f:
                X_rf = f["data"]

            # Load NN
            with np.load(nn_path) as f:
                data_nn = {k: f[k] for k in f.files}
                # Handle potential 0-d array for labels if saved weirdly, though np.savez handles kwargs fine
                # If 'labels' key is missing (e.g. test set), it won't be in dict, which is correct.

            results[split] = (X_rf, data_nn)

        return results["train"], results["val"], results["test"]

    # If not cached, compute from scratch
    print("Computing features from scratch...")

    # 1. Load Raw Data
    df_train, df_val = data_loader.get_stratified_split(load_cached_data=True)
    df_test = data_loader.load_dataset("test", load_cached_data=True)

    # 2. Fit Pipeline
    pipeline = FeaturePipeline()
    pipeline.fit(df_train)

    # 3. Transform and Cache
    results = {}
    dfs = {"train": df_train, "val": df_val, "test": df_test}

    for split, df in dfs.items():
        print(f"Processing {split} set...")

        # Compute shared expensive artifacts
        artifacts = pipeline._compute_common_artifacts(df)

        # Transform
        X_rf = pipeline.transform_rf(df, artifacts)
        data_nn = pipeline.transform_nn(df, artifacts)

        # Save
        rf_path = os.path.join(cache_dir, f"{split}_rf_features.npz")
        nn_path = os.path.join(cache_dir, f"{split}_nn_features.npz")

        np.savez(rf_path, data=X_rf)
        np.savez(nn_path, **data_nn)

        results[split] = (X_rf, data_nn)

    return results["train"], results["val"], results["test"]
