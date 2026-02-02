import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sentence_transformers import SentenceTransformer

# Import config
from library.config import (
    WORKING_DIR,
    TFIDF_PARAMS,
    MPNET_MODEL_NAME,
    TEXT_EDIT_AWARE_COL,
    SUBREDDIT_COL,
    TARGET_COL,
    RANDOM_SEED,
)


class FeatureFactory:
    def __init__(self):
        self.working_dir = WORKING_DIR
        os.makedirs(self.working_dir, exist_ok=True)
        self.seed = RANDOM_SEED

    def _get_file_paths(self):
        """Define paths for all cached artifacts."""
        splits = ["train", "val", "test"]

        paths = {}

        # Sparse Matrices (.npz)
        for split in splits:
            paths[f"X_{split}_lexical"] = os.path.join(
                self.working_dir, f"X_{split}_lexical.npz"
            )
            paths[f"X_{split}_behavioral"] = os.path.join(
                self.working_dir, f"X_{split}_behavioral.npz"
            )

        # Dense Arrays (.npy)
        dense_types = ["text_emb", "sub_emb", "metadata"]
        for split in splits:
            for dt in dense_types:
                paths[f"X_{split}_{dt}"] = os.path.join(
                    self.working_dir, f"X_{split}_{dt}.npy"
                )

            # Targets (only for train/val)
            if split != "test":
                paths[f"y_{split}"] = os.path.join(self.working_dir, f"y_{split}.npy")

        return paths

    def _compute_cosine_similarity(self, emb1, emb2):
        """
        Computes row-wise cosine similarity between two embedding matrices.
        Returns shape (N, 1).
        """
        # Normalize rows
        norm1 = np.linalg.norm(emb1, axis=1, keepdims=True)
        norm2 = np.linalg.norm(emb2, axis=1, keepdims=True)

        # Avoid division by zero
        norm1[norm1 == 0] = 1e-9
        norm2[norm2 == 0] = 1e-9

        # Dot product
        dot = np.sum(emb1 * emb2, axis=1, keepdims=True)

        # Cosine similarity
        sim = dot / (norm1 * norm2)
        return sim

    def generate_embeddings(self, texts, model_name=MPNET_MODEL_NAME):
        """Generates dense embeddings using SentenceTransformer."""
        print(f"Generating embeddings with {model_name}...")
        model = SentenceTransformer(model_name)
        # Ensure deterministic behavior
        embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return embeddings

    def process_metadata(self, train_df, val_df, test_df):
        """
        Process numerical metadata:
        1. Convert bools to ints.
        2. Select columns (exclude leakage).
        3. Impute.
        4. Scale.
        """
        print("Processing tabular metadata...")

        # Convert boolean columns to int to ensure they are picked up as numeric
        for df in [train_df, val_df, test_df]:
            for col in df.columns:
                if df[col].dtype == "bool":
                    df[col] = df[col].astype(int)

        # Identify columns
        all_cols = train_df.columns
        exclude_suffixes = (
            "_at_retrieval",
            "request_id",
            "requester_username",
            "source_file",
            "request_text",
            "request_title",
            "request_text_edit_aware",
            "requester_subreddits_at_request",
            "requester_received_pizza",
            "giver_username_if_known",
            "requester_user_flair",
        )

        # Select numerical columns that don't end with excluded suffixes and are not in excluded list
        feature_cols = []
        for col in all_cols:
            if col in exclude_suffixes:
                continue
            if col.endswith("_at_retrieval"):
                continue
            if not pd.api.types.is_numeric_dtype(train_df[col]):
                continue
            # Ensure column exists in test set (Cite debug_lesson_1)
            if col not in test_df.columns:
                continue
            feature_cols.append(col)

        print(f"Selected metadata columns: {feature_cols}")

        # Extract
        X_train = train_df[feature_cols].values.astype(float)
        X_val = val_df[feature_cols].values.astype(float)
        X_test = test_df[feature_cols].values.astype(float)

        # Impute
        imputer = SimpleImputer(strategy="median")
        X_train = imputer.fit_transform(X_train)
        X_val = imputer.transform(X_val)
        X_test = imputer.transform(X_test)

        # Scale
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        return X_train, X_val, X_test

    def process_data(self, train_df, val_df, test_df, load_cached_data=True):
        paths = self._get_file_paths()

        # Check cache
        cache_exists = all(os.path.exists(p) for p in paths.values())

        if load_cached_data and cache_exists:
            print("Loading features from cache...")
            data = {}
            for key, path in paths.items():
                if path.endswith(".npz"):
                    data[key] = sp.load_npz(path)
                else:
                    data[key] = np.load(path)
            return data

        print("Computing features from scratch...")

        # 1. Targets
        y_train = train_df[TARGET_COL].values.astype(int)
        y_val = val_df[TARGET_COL].values.astype(int)

        # 2. Sparse Lexical (Text TFIDF)
        print("Generating Sparse Lexical features...")
        tfidf_text = TfidfVectorizer(**TFIDF_PARAMS)
        X_train_lexical = tfidf_text.fit_transform(
            train_df[TEXT_EDIT_AWARE_COL].fillna("")
        )
        X_val_lexical = tfidf_text.transform(val_df[TEXT_EDIT_AWARE_COL].fillna(""))
        X_test_lexical = tfidf_text.transform(test_df[TEXT_EDIT_AWARE_COL].fillna(""))

        # 3. Sparse Behavioral (Subreddit TFIDF)
        print("Generating Sparse Behavioral features...")
        tfidf_sub = TfidfVectorizer(**TFIDF_PARAMS)
        X_train_behavioral = tfidf_sub.fit_transform(train_df[SUBREDDIT_COL].fillna(""))
        X_val_behavioral = tfidf_sub.transform(val_df[SUBREDDIT_COL].fillna(""))
        X_test_behavioral = tfidf_sub.transform(test_df[SUBREDDIT_COL].fillna(""))

        # 4. Dense Embeddings (MPNet)
        # Text
        X_train_text_emb = self.generate_embeddings(
            train_df[TEXT_EDIT_AWARE_COL].fillna("").tolist()
        )
        X_val_text_emb = self.generate_embeddings(
            val_df[TEXT_EDIT_AWARE_COL].fillna("").tolist()
        )
        X_test_text_emb = self.generate_embeddings(
            test_df[TEXT_EDIT_AWARE_COL].fillna("").tolist()
        )

        # Subreddits
        X_train_sub_emb = self.generate_embeddings(
            train_df[SUBREDDIT_COL].fillna("").tolist()
        )
        X_val_sub_emb = self.generate_embeddings(
            val_df[SUBREDDIT_COL].fillna("").tolist()
        )
        X_test_sub_emb = self.generate_embeddings(
            test_df[SUBREDDIT_COL].fillna("").tolist()
        )

        # 5. Interaction Feature (Cosine Sim)
        print("Computing Interaction features...")
        train_sim = self._compute_cosine_similarity(X_train_text_emb, X_train_sub_emb)
        val_sim = self._compute_cosine_similarity(X_val_text_emb, X_val_sub_emb)
        test_sim = self._compute_cosine_similarity(X_test_text_emb, X_test_sub_emb)

        # 6. Metadata
        X_train_base, X_val_base, X_test_base = self.process_metadata(
            train_df, val_df, test_df
        )

        # Concatenate Metadata + Interaction
        X_train_meta = np.hstack([X_train_base, train_sim])
        X_val_meta = np.hstack([X_val_base, val_sim])
        X_test_meta = np.hstack([X_test_base, test_sim])

        # 7. Save to Cache
        print("Saving features to cache...")
        sp.save_npz(paths["X_train_lexical"], X_train_lexical)
        sp.save_npz(paths["X_val_lexical"], X_val_lexical)
        sp.save_npz(paths["X_test_lexical"], X_test_lexical)

        sp.save_npz(paths["X_train_behavioral"], X_train_behavioral)
        sp.save_npz(paths["X_val_behavioral"], X_val_behavioral)
        sp.save_npz(paths["X_test_behavioral"], X_test_behavioral)

        np.save(paths["X_train_text_emb"], X_train_text_emb)
        np.save(paths["X_val_text_emb"], X_val_text_emb)
        np.save(paths["X_test_text_emb"], X_test_text_emb)

        np.save(paths["X_train_sub_emb"], X_train_sub_emb)
        np.save(paths["X_val_sub_emb"], X_val_sub_emb)
        np.save(paths["X_test_sub_emb"], X_test_sub_emb)

        np.save(paths["X_train_metadata"], X_train_meta)
        np.save(paths["X_val_metadata"], X_val_meta)
        np.save(paths["X_test_metadata"], X_test_meta)

        np.save(paths["y_train"], y_train)
        np.save(paths["y_val"], y_val)

        # Return Data
        return {
            "X_train_lexical": X_train_lexical,
            "X_val_lexical": X_val_lexical,
            "X_test_lexical": X_test_lexical,
            "X_train_behavioral": X_train_behavioral,
            "X_val_behavioral": X_val_behavioral,
            "X_test_behavioral": X_test_behavioral,
            "X_train_text_emb": X_train_text_emb,
            "X_val_text_emb": X_val_text_emb,
            "X_test_text_emb": X_test_text_emb,
            "X_train_sub_emb": X_train_sub_emb,
            "X_val_sub_emb": X_val_sub_emb,
            "X_test_sub_emb": X_test_sub_emb,
            "X_train_metadata": X_train_meta,
            "X_val_metadata": X_val_meta,
            "X_test_metadata": X_test_meta,
            "y_train": y_train,
            "y_val": y_val,
        }
