import os
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
from library.config import Config, safe_literal_eval, get_tabular_features


class FeaturePipeline:
    def __init__(self):
        self.config = Config
        self.cache_dir = self.config.CACHE_DIR
        self.rf_cache_path = os.path.join(self.cache_dir, "rf_data.npz")
        self.mlp_cache_path = os.path.join(self.cache_dir, "mlp_data.npz")
        self.meta_cache_path = os.path.join(self.cache_dir, "meta.json")

    def process_data(self, load_cached_data=True):
        # 1. Check Cache
        if load_cached_data and self._check_cache_exists():
            print("Loading cached data...")
            return self._load_cache()

        print("Processing data from scratch...")
        os.makedirs(self.cache_dir, exist_ok=True)

        # 2. Load Raw Data
        df_train = pd.read_csv(self.config.TRAIN_DATA_PATH)
        df_val = pd.read_csv(self.config.VAL_DATA_PATH)
        df_test = pd.read_csv(self.config.TEST_DATA_PATH)

        y_train = df_train["requester_received_pizza"].astype(int).values
        y_val = df_val["requester_received_pizza"].astype(int).values
        test_ids = df_test["request_id"].values

        train_text = df_train["request_text_edit_aware"].fillna("").astype(str).tolist()
        val_text = df_val["request_text_edit_aware"].fillna("").astype(str).tolist()
        test_text = df_test["request_text_edit_aware"].fillna("").astype(str).tolist()

        # 3. Stream A: RF Processing
        # TF-IDF
        X_train_tfidf, X_val_tfidf, X_test_tfidf = self._process_tfidf(
            train_text, val_text, test_text
        )

        # Metadata
        X_train_tab, tab_cols = self._generate_metadata(df_train)
        X_val_tab, _ = self._generate_metadata(df_val, train_cols=tab_cols)
        X_test_tab, _ = self._generate_metadata(df_test, train_cols=tab_cols)

        # Imputation
        X_train_tab_imp, X_val_tab_imp, X_test_tab_imp = self._preprocess_rf(
            X_train_tab, X_val_tab, X_test_tab
        )

        # Concatenate
        X_train_rf = np.hstack([X_train_tfidf, X_train_tab_imp])
        X_val_rf = np.hstack([X_val_tfidf, X_val_tab_imp])
        X_test_rf = np.hstack([X_test_tfidf, X_test_tab_imp])

        # 4. Stream B: MLP Processing
        # Initialize SBERT once
        sbert = SentenceTransformer(self.config.SBERT_MODEL_NAME)

        # Text Embeddings
        X_train_sbert = self._encode_text_sbert(train_text, sbert)
        X_val_sbert = self._encode_text_sbert(val_text, sbert)
        X_test_sbert = self._encode_text_sbert(test_text, sbert)

        # History Embeddings
        X_train_comm = self._encode_history_sbert(
            df_train["requester_subreddits_at_request"], sbert
        )
        X_val_comm = self._encode_history_sbert(
            df_val["requester_subreddits_at_request"], sbert
        )
        X_test_comm = self._encode_history_sbert(
            df_test["requester_subreddits_at_request"], sbert
        )

        # Tabular Preprocessing (Arcsinh + Scale)
        X_train_tab_scaled, X_val_tab_scaled, X_test_tab_scaled = self._preprocess_mlp(
            X_train_tab, X_val_tab, X_test_tab
        )

        # 5. Save Cache
        self._save_cache(
            X_train_rf,
            X_val_rf,
            X_test_rf,
            X_train_sbert,
            X_train_comm,
            X_train_tab_scaled,
            X_val_sbert,
            X_val_comm,
            X_val_tab_scaled,
            X_test_sbert,
            X_test_comm,
            X_test_tab_scaled,
            y_train,
            y_val,
            test_ids,
            tab_cols,
        )

        # 6. Return Data
        return {
            "rf": {"X_train": X_train_rf, "X_val": X_val_rf, "X_test": X_test_rf},
            "mlp": {
                "X_train_text": X_train_sbert,
                "X_train_comm": X_train_comm,
                "X_train_tab": X_train_tab_scaled,
                "X_val_text": X_val_sbert,
                "X_val_comm": X_val_comm,
                "X_val_tab": X_val_tab_scaled,
                "X_test_text": X_test_sbert,
                "X_test_comm": X_test_comm,
                "X_test_tab": X_test_tab_scaled,
            },
            "y_train": y_train,
            "y_val": y_val,
            "test_ids": test_ids,
        }

    def _generate_metadata(self, df, train_cols=None):
        return get_tabular_features(df, train_cols)

    def _process_tfidf(self, train_text, val_text, test_text):
        tfidf = TfidfVectorizer(
            ngram_range=self.config.TFIDF_NGRAM_RANGE,
            max_features=self.config.TFIDF_MAX_FEATURES,
            binary=True,
            stop_words="english",
        )
        X_train = tfidf.fit_transform(train_text).toarray()
        X_val = tfidf.transform(val_text).toarray()
        X_test = tfidf.transform(test_text).toarray()
        return X_train, X_val, X_test

    def _preprocess_rf(self, X_train, X_val, X_test):
        imputer = SimpleImputer(strategy="median")
        X_train_imp = imputer.fit_transform(X_train)
        X_val_imp = imputer.transform(X_val)
        X_test_imp = imputer.transform(X_test)
        return X_train_imp, X_val_imp, X_test_imp

    def _encode_text_sbert(self, texts, sbert_model):
        return sbert_model.encode(texts, show_progress_bar=False)

    def _encode_history_sbert(self, subreddits_series, sbert_model):
        embeddings = []
        for subs_str in subreddits_series:
            subs = safe_literal_eval(subs_str)
            if not subs:
                embeddings.append(np.zeros(384))
            else:
                sub_embs = sbert_model.encode(subs, show_progress_bar=False)
                embeddings.append(np.mean(sub_embs, axis=0))
        return np.array(embeddings)

    def _preprocess_mlp(self, X_train, X_val, X_test):
        X_train_arc = np.arcsinh(np.nan_to_num(X_train))
        X_val_arc = np.arcsinh(np.nan_to_num(X_val))
        X_test_arc = np.arcsinh(np.nan_to_num(X_test))

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_arc)
        X_val_scaled = scaler.transform(X_val_arc)
        X_test_scaled = scaler.transform(X_test_arc)
        return X_train_scaled, X_val_scaled, X_test_scaled

    def _check_cache_exists(self):
        return (
            os.path.exists(self.rf_cache_path)
            and os.path.exists(self.mlp_cache_path)
            and os.path.exists(self.meta_cache_path)
        )

    def _load_cache(self):
        rf_data = np.load(self.rf_cache_path, allow_pickle=True)
        mlp_data = np.load(self.mlp_cache_path, allow_pickle=True)
        return {
            "rf": {
                "X_train": rf_data["X_train"],
                "X_val": rf_data["X_val"],
                "X_test": rf_data["X_test"],
            },
            "mlp": {k: mlp_data[k] for k in mlp_data.files},
            "y_train": rf_data["y_train"],
            "y_val": rf_data["y_val"],
            "test_ids": rf_data["test_ids"],
        }

    def _save_cache(
        self,
        X_train_rf,
        X_val_rf,
        X_test_rf,
        X_train_sbert,
        X_train_comm,
        X_train_tab_scaled,
        X_val_sbert,
        X_val_comm,
        X_val_tab_scaled,
        X_test_sbert,
        X_test_comm,
        X_test_tab_scaled,
        y_train,
        y_val,
        test_ids,
        tab_cols,
    ):
        np.savez(
            self.rf_cache_path,
            X_train=X_train_rf,
            X_val=X_val_rf,
            X_test=X_test_rf,
            y_train=y_train,
            y_val=y_val,
            test_ids=test_ids,
        )
        np.savez(
            self.mlp_cache_path,
            X_train_text=X_train_sbert,
            X_train_comm=X_train_comm,
            X_train_tab=X_train_tab_scaled,
            X_val_text=X_val_sbert,
            X_val_comm=X_val_comm,
            X_val_tab=X_val_tab_scaled,
            X_test_text=X_test_sbert,
            X_test_comm=X_test_comm,
            X_test_tab=X_test_tab_scaled,
        )
        with open(self.meta_cache_path, "w") as f:
            json.dump({"tab_cols": tab_cols}, f)
