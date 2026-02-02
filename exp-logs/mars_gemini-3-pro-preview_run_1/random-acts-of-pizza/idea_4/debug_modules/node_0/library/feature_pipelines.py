import os
import numpy as np
import pandas as pd
import scipy.sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
from library import config
from library import data_loader


class StreamA_Pipeline:
    """
    Pipeline for the Lexical-Tabular Learner (Random Forest).
    Generates a sparse matrix concatenating TF-IDF features and raw metadata.
    """

    def __init__(self):
        self.cache_dir = config.CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)

        # Define cache file paths
        self.files = {
            "train_x": os.path.join(self.cache_dir, "stream_a_train_x.npz"),
            "train_y": os.path.join(self.cache_dir, "stream_a_train_y.npy"),
            "val_x": os.path.join(self.cache_dir, "stream_a_val_x.npz"),
            "val_y": os.path.join(self.cache_dir, "stream_a_val_y.npy"),
            "test_x": os.path.join(self.cache_dir, "stream_a_test_x.npz"),
            "test_ids": os.path.join(self.cache_dir, "stream_a_test_ids.npy"),
        }

    def _get_meta_cols(self, df):
        exclude = {config.ID_COL, config.TEXT_COL, config.TARGET_COL}
        return [c for c in df.columns if c not in exclude]

    def run(self, load_cached_data=True):
        # 1. Try Loading from Cache
        if load_cached_data:
            if all(os.path.exists(p) for p in self.files.values()):
                print("Stream A: Loading features from cache...")
                X_train = scipy.sparse.load_npz(self.files["train_x"])
                y_train = np.load(self.files["train_y"])
                X_val = scipy.sparse.load_npz(self.files["val_x"])
                y_val = np.load(self.files["val_y"])
                X_test = scipy.sparse.load_npz(self.files["test_x"])
                ids_test = np.load(self.files["test_ids"], allow_pickle=True)
                return (X_train, y_train), (X_val, y_val), (X_test, ids_test)
            else:
                print("Stream A: Cache miss. Processing...")
        else:
            print("Stream A: Ignoring cache. Processing...")

        # 2. Load Data
        df_train, df_val, df_test = data_loader.load_and_clean_data(
            load_cached_data=load_cached_data
        )

        # 3. Process Text (TF-IDF)
        print("Stream A: Vectorizing text...")
        tfidf = TfidfVectorizer(**config.TFIDF_PARAMS)

        # Fit on train, transform all
        train_text = df_train[config.TEXT_COL].fillna("").astype(str)
        val_text = df_val[config.TEXT_COL].fillna("").astype(str)
        test_text = df_test[config.TEXT_COL].fillna("").astype(str)

        X_text_train = tfidf.fit_transform(train_text)
        X_text_val = tfidf.transform(val_text)
        X_text_test = tfidf.transform(test_text)

        # 4. Process Metadata (Raw)
        print("Stream A: Processing metadata...")
        meta_cols = self._get_meta_cols(df_train)

        # Ensure alignment and type
        X_meta_train = scipy.sparse.csr_matrix(df_train[meta_cols].values.astype(float))
        X_meta_val = scipy.sparse.csr_matrix(df_val[meta_cols].values.astype(float))
        X_meta_test = scipy.sparse.csr_matrix(df_test[meta_cols].values.astype(float))

        # 5. Concatenate
        X_train = scipy.sparse.hstack([X_text_train, X_meta_train])
        X_val = scipy.sparse.hstack([X_text_val, X_meta_val])
        X_test = scipy.sparse.hstack([X_text_test, X_meta_test])

        # 6. Extract Targets and IDs
        y_train = df_train[config.TARGET_COL].values.astype(int)
        y_val = df_val[config.TARGET_COL].values.astype(int)
        ids_test = df_test[config.ID_COL].values

        # 7. Save to Cache
        print("Stream A: Saving to cache...")
        scipy.sparse.save_npz(self.files["train_x"], X_train)
        np.save(self.files["train_y"], y_train)
        scipy.sparse.save_npz(self.files["val_x"], X_val)
        np.save(self.files["val_y"], y_val)
        scipy.sparse.save_npz(self.files["test_x"], X_test)
        np.save(self.files["test_ids"], ids_test)

        return (X_train, y_train), (X_val, y_val), (X_test, ids_test)


class StreamB_Pipeline:
    """
    Pipeline for the Semantic-Tabular Learner (Dual-Branch MLP).
    Generates dense embeddings (Sentence Transformer) and scaled metadata.
    Returns separated inputs for the dual-branch architecture.
    """

    def __init__(self):
        self.cache_dir = config.CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)

        # Define cache file paths
        self.files = {
            "train_sem": os.path.join(self.cache_dir, "stream_b_train_sem.npy"),
            "train_meta": os.path.join(self.cache_dir, "stream_b_train_meta.npy"),
            "train_y": os.path.join(self.cache_dir, "stream_b_train_y.npy"),
            "val_sem": os.path.join(self.cache_dir, "stream_b_val_sem.npy"),
            "val_meta": os.path.join(self.cache_dir, "stream_b_val_meta.npy"),
            "val_y": os.path.join(self.cache_dir, "stream_b_val_y.npy"),
            "test_sem": os.path.join(self.cache_dir, "stream_b_test_sem.npy"),
            "test_meta": os.path.join(self.cache_dir, "stream_b_test_meta.npy"),
            "test_ids": os.path.join(self.cache_dir, "stream_b_test_ids.npy"),
        }

    def _get_meta_cols(self, df):
        exclude = {config.ID_COL, config.TEXT_COL, config.TARGET_COL}
        return [c for c in df.columns if c not in exclude]

    def run(self, load_cached_data=True):
        # 1. Try Loading from Cache
        if load_cached_data:
            if all(os.path.exists(p) for p in self.files.values()):
                print("Stream B: Loading features from cache...")
                # Train
                X_sem_train = np.load(self.files["train_sem"])
                X_meta_train = np.load(self.files["train_meta"])
                y_train = np.load(self.files["train_y"])
                # Val
                X_sem_val = np.load(self.files["val_sem"])
                X_meta_val = np.load(self.files["val_meta"])
                y_val = np.load(self.files["val_y"])
                # Test
                X_sem_test = np.load(self.files["test_sem"])
                X_meta_test = np.load(self.files["test_meta"])
                ids_test = np.load(self.files["test_ids"], allow_pickle=True)

                return (
                    (X_sem_train, X_meta_train, y_train),
                    (X_sem_val, X_meta_val, y_val),
                    (X_sem_test, X_meta_test, ids_test),
                )
            else:
                print("Stream B: Cache miss. Processing...")
        else:
            print("Stream B: Ignoring cache. Processing...")

        # 2. Load Data
        df_train, df_val, df_test = data_loader.load_and_clean_data(
            load_cached_data=load_cached_data
        )

        # 3. Process Text (Sentence Transformer)
        print(
            f"Stream B: Generating embeddings using {config.SENTENCE_TRANSFORMER_MODEL}..."
        )
        model = SentenceTransformer(config.SENTENCE_TRANSFORMER_MODEL)

        train_text = df_train[config.TEXT_COL].fillna("").astype(str).tolist()
        val_text = df_val[config.TEXT_COL].fillna("").astype(str).tolist()
        test_text = df_test[config.TEXT_COL].fillna("").astype(str).tolist()

        # Encode (Dense vectors)
        X_sem_train = model.encode(
            train_text, show_progress_bar=False, convert_to_numpy=True
        )
        X_sem_val = model.encode(
            val_text, show_progress_bar=False, convert_to_numpy=True
        )
        X_sem_test = model.encode(
            test_text, show_progress_bar=False, convert_to_numpy=True
        )

        # 4. Process Metadata (Scaled)
        print("Stream B: Scaling metadata...")
        meta_cols = self._get_meta_cols(df_train)
        scaler = StandardScaler()

        X_meta_train = scaler.fit_transform(df_train[meta_cols].values.astype(float))
        X_meta_val = scaler.transform(df_val[meta_cols].values.astype(float))
        X_meta_test = scaler.transform(df_test[meta_cols].values.astype(float))

        # 5. Extract Targets and IDs
        y_train = df_train[config.TARGET_COL].values.astype(int)
        y_val = df_val[config.TARGET_COL].values.astype(int)
        ids_test = df_test[config.ID_COL].values

        # 6. Save to Cache
        print("Stream B: Saving to cache...")
        np.save(self.files["train_sem"], X_sem_train)
        np.save(self.files["train_meta"], X_meta_train)
        np.save(self.files["train_y"], y_train)

        np.save(self.files["val_sem"], X_sem_val)
        np.save(self.files["val_meta"], X_meta_val)
        np.save(self.files["val_y"], y_val)

        np.save(self.files["test_sem"], X_sem_test)
        np.save(self.files["test_meta"], X_meta_test)
        np.save(self.files["test_ids"], ids_test)

        return (
            (X_sem_train, X_meta_train, y_train),
            (X_sem_val, X_meta_val, y_val),
            (X_sem_test, X_meta_test, ids_test),
        )
