import os
import ast
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sentence_transformers import SentenceTransformer

from library.config import Config
from library.data_utils import load_dataset


class FeatureEngineer:
    """
    Handles feature engineering for the Hybrid Ensemble (RF + MLP).
    Generates TF-IDF, SBERT embeddings, Semantic Scores (Centroid/Peak),
    Top-K Community Indicators, and Scaled Metadata.
    """

    def __init__(self):
        self.sbert = None  # Lazy load
        self.tfidf = TfidfVectorizer(
            max_features=Config.TFIDF_MAX_FEATURES,
            ngram_range=Config.TFIDF_NGRAM_RANGE,
            stop_words="english",
            sublinear_tf=True,
        )
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy="median")
        self.top_k_subreddits = []
        # Cache for subreddit embeddings to avoid re-encoding
        self.subreddit_embeddings = {}

    def process_data(self, load_cached_data=True):
        """
        Main pipeline execution. Checks cache, loads data, fits/transforms, and saves cache.
        Returns dictionaries containing features for RF and MLP.
        """
        # Define cache paths
        cache_train = os.path.join(Config.WORKING_DIR, "train_features.npz")
        cache_val = os.path.join(Config.WORKING_DIR, "val_features.npz")
        cache_test = os.path.join(Config.WORKING_DIR, "test_features.npz")

        # Check if cache exists
        if (
            load_cached_data
            and os.path.exists(cache_train)
            and os.path.exists(cache_val)
            and os.path.exists(cache_test)
        ):
            print("Loading features from cache...")
            train_data = self._load_npz(cache_train)
            val_data = self._load_npz(cache_val)
            test_data = self._load_npz(cache_test)
            return train_data, val_data, test_data

        print("Computing features from scratch...")
        # Load raw data
        train_df, val_df, test_df = load_dataset(debug=Config.DEBUG)

        # 1. Fit Steps (TFIDF, Top-K, Scaler) using Train
        print("Fitting feature extractors...")
        self._fit(train_df)

        # 2. Transform Steps
        print("Transforming Train...")
        train_data = self._transform(train_df, is_train=True)
        print("Transforming Val...")
        val_data = self._transform(val_df, is_train=False)
        print("Transforming Test...")
        test_data = self._transform(test_df, is_train=False)

        # 3. Save to Cache
        print("Saving features to cache...")
        self._save_npz(cache_train, train_data)
        self._save_npz(cache_val, val_data)
        self._save_npz(cache_test, test_data)

        return train_data, val_data, test_data

    def _fit(self, df):
        """
        Fits stateful transformers (TF-IDF, Scaler, Imputer) and identifies Top-K subreddits.
        """
        # TF-IDF
        text_data = (
            df["request_title"].fillna("")
            + " "
            + df["request_text_edit_aware"].fillna("")
        ).tolist()
        self.tfidf.fit(text_data)

        # Top-K Subreddits
        all_subreddits = []
        for sub_list_str in df["requester_subreddits_at_request"]:
            try:
                subs = (
                    ast.literal_eval(sub_list_str)
                    if isinstance(sub_list_str, str)
                    else []
                )
                all_subreddits.extend(subs)
            except:
                pass

        if all_subreddits:
            counts = pd.Series(all_subreddits).value_counts()
            self.top_k_subreddits = counts.head(Config.TOP_K_SUBREDDITS).index.tolist()
        else:
            self.top_k_subreddits = []

        # Metadata Scaler
        meta_features = self._extract_raw_metadata(df)
        self.imputer.fit(meta_features)
        self.scaler.fit(self.imputer.transform(meta_features))

    def _transform(self, df, is_train=False):
        """
        Generates all features for a given dataframe.
        """
        # --- 1. Text Features (TF-IDF) ---
        text_data = (
            df["request_title"].fillna("")
            + " "
            + df["request_text_edit_aware"].fillna("")
        ).tolist()
        tfidf_mat = self.tfidf.transform(text_data)  # Sparse

        # --- 2. Metadata Features (Scaled) ---
        raw_meta = self._extract_raw_metadata(df)
        imputed_meta = self.imputer.transform(raw_meta)
        scaled_meta = self.scaler.transform(imputed_meta)

        # --- 3. Top-K Indicators ---
        top_k_mat = self._generate_top_k_indicators(df)

        # --- 4. SBERT Features (Embeddings + Semantic Scores) ---
        # This returns components for both MLP (embeddings) and RF (scores)
        sbert_out = self._generate_sbert_features(df)

        # --- Assemble RF Features ---
        # Concatenate: TFIDF (sparse) + Scaled Meta (dense) + Top-K (dense) + Semantic Scores (dense)
        semantic_scores = sbert_out["semantic_scores"]  # (N, 4)

        # Combine dense features
        dense_features = np.hstack([scaled_meta, top_k_mat, semantic_scores])

        # Final RF Matrix (Sparse CSR)
        X_rf = sp.hstack([tfidf_mat, sp.csr_matrix(dense_features)], format="csr")

        # --- Assemble MLP Features ---
        X_mlp = {
            "title_emb": sbert_out["title_emb"],
            "body_emb": sbert_out["body_emb"],
            "history_emb": sbert_out["history_emb"],  # (N, Max_Len, Dim)
            "metadata": scaled_meta.astype(np.float32),
        }

        # --- Labels ---
        if "requester_received_pizza" in df.columns:
            y = df["requester_received_pizza"].astype(int).values
        else:
            y = np.zeros(len(df))  # Dummy for test

        return {
            "X_rf": X_rf,  # Sparse matrix
            "X_mlp": X_mlp,  # Dict of arrays
            "y": y,
            "ids": df["request_id"].values,
        }

    def _extract_raw_metadata(self, df):
        """
        Extracts numeric columns and applies Arcsinh transformation to skewed counts.
        """
        data = df[Config.NUMERIC_COLS].copy()

        # Apply Arcsinh to count columns (skewed)
        for col in Config.NUMERIC_COLS:
            # Heuristic: apply to count/age/score columns
            if "number_of" in col or "age" in col or "upvotes" in col:
                data[col] = np.arcsinh(data[col])

        return data.values.astype(np.float32)

    def _generate_top_k_indicators(self, df):
        """
        Creates a binary matrix indicating presence of top-K subreddits in user history.
        """
        N = len(df)
        K = len(self.top_k_subreddits)
        mat = np.zeros((N, K), dtype=np.float32)

        sub_map = {sub: i for i, sub in enumerate(self.top_k_subreddits)}

        for idx, row_str in enumerate(df["requester_subreddits_at_request"]):
            try:
                subs = ast.literal_eval(row_str) if isinstance(row_str, str) else []
                for sub in subs:
                    if sub in sub_map:
                        mat[idx, sub_map[sub]] = 1.0
            except:
                pass
        return mat

    def _generate_sbert_features(self, df):
        """
        Generates SBERT embeddings for Title, Body, and User History.
        Computes Semantic Scores (Centroid Similarity, Peak Relevance).
        """
        # Lazy load model
        if self.sbert is None:
            print(f"Loading SBERT model: {Config.SBERT_MODEL_NAME}...")
            self.sbert = SentenceTransformer(
                Config.SBERT_MODEL_NAME, device=Config.DEVICE
            )

        titles = df["request_title"].fillna("").tolist()
        bodies = df["request_text_edit_aware"].fillna("").tolist()

        print("Encoding Titles...")
        title_embs = self.sbert.encode(
            titles, batch_size=64, show_progress_bar=False, convert_to_numpy=True
        )
        print("Encoding Bodies...")
        body_embs = self.sbert.encode(
            bodies, batch_size=64, show_progress_bar=False, convert_to_numpy=True
        )

        # History Processing
        # 1. Identify all unique subreddits in this batch to batch-encode them
        all_subs_in_df = set()
        parsed_histories = []

        for row_str in df["requester_subreddits_at_request"]:
            try:
                subs = ast.literal_eval(row_str) if isinstance(row_str, str) else []
                # Limit history length
                subs = subs[: Config.MAX_HISTORY_LEN]
                parsed_histories.append(subs)
                all_subs_in_df.update(subs)
            except:
                parsed_histories.append([])

        # Encode unique subreddits
        unique_subs = list(all_subs_in_df)
        if unique_subs:
            print(f"Encoding {len(unique_subs)} unique subreddits...")
            sub_embs = self.sbert.encode(
                unique_subs,
                batch_size=128,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            sub_emb_map = {sub: emb for sub, emb in zip(unique_subs, sub_embs)}
        else:
            sub_emb_map = {}

        # Construct History Tensor and Semantic Scores
        N = len(df)
        dim = Config.SBERT_EMBEDDING_DIM
        max_len = Config.MAX_HISTORY_LEN

        history_tensor = np.zeros((N, max_len, dim), dtype=np.float32)

        # Semantic Scores: Global (Centroid) and Peak (Max)
        # Shape: (N, 4) -> [Title-Centroid, Body-Centroid, Title-Peak, Body-Peak]
        semantic_scores = np.zeros((N, 4), dtype=np.float32)

        for i in range(N):
            subs = parsed_histories[i]
            user_sub_embs = []
            for sub in subs:
                if sub in sub_emb_map:
                    user_sub_embs.append(sub_emb_map[sub])

            if len(user_sub_embs) > 0:
                user_sub_embs = np.array(user_sub_embs)  # (L, D)

                # Fill Tensor
                L = min(len(user_sub_embs), max_len)
                history_tensor[i, :L, :] = user_sub_embs[:L]

                # Centroid
                centroid = np.mean(user_sub_embs, axis=0)

                # Cosine Sims
                # Normalize for dot product
                t_norm = title_embs[i] / (np.linalg.norm(title_embs[i]) + 1e-9)
                b_norm = body_embs[i] / (np.linalg.norm(body_embs[i]) + 1e-9)
                c_norm = centroid / (np.linalg.norm(centroid) + 1e-9)

                # History items normalized
                h_norms = user_sub_embs / (
                    np.linalg.norm(user_sub_embs, axis=1, keepdims=True) + 1e-9
                )

                # Global Consistency
                semantic_scores[i, 0] = np.dot(t_norm, c_norm)
                semantic_scores[i, 1] = np.dot(b_norm, c_norm)

                # Peak Relevance
                # Dot product between title/body and all history items -> take max
                t_sims = np.dot(h_norms, t_norm)
                b_sims = np.dot(h_norms, b_norm)

                semantic_scores[i, 2] = np.max(t_sims)
                semantic_scores[i, 3] = np.max(b_sims)
            else:
                # No history: scores remain 0
                pass

        return {
            "title_emb": title_embs,
            "body_emb": body_embs,
            "history_emb": history_tensor,
            "semantic_scores": semantic_scores,
        }

    def _save_npz(self, path, data_dict):
        """
        Saves the composite data structure to a compressed NPZ file.
        Deconstructs sparse matrices into components.
        """
        save_dict = {
            "X_rf_data": data_dict["X_rf"].data,
            "X_rf_indices": data_dict["X_rf"].indices,
            "X_rf_indptr": data_dict["X_rf"].indptr,
            "X_rf_shape": data_dict["X_rf"].shape,
            "mlp_title": data_dict["X_mlp"]["title_emb"],
            "mlp_body": data_dict["X_mlp"]["body_emb"],
            "mlp_history": data_dict["X_mlp"]["history_emb"],
            "mlp_meta": data_dict["X_mlp"]["metadata"],
            "y": data_dict["y"],
            "ids": data_dict["ids"],
        }
        np.savez_compressed(path, **save_dict)

    def _load_npz(self, path):
        """
        Loads the composite data structure from an NPZ file.
        Reconstructs sparse matrices.
        """
        loaded = np.load(path, allow_pickle=True)

        # Reconstruct Sparse Matrix
        X_rf = sp.csr_matrix(
            (loaded["X_rf_data"], loaded["X_rf_indices"], loaded["X_rf_indptr"]),
            shape=loaded["X_rf_shape"],
        )

        X_mlp = {
            "title_emb": loaded["mlp_title"],
            "body_emb": loaded["mlp_body"],
            "history_emb": loaded["mlp_history"],
            "metadata": loaded["mlp_meta"],
        }

        return {
            "X_rf": X_rf,
            "X_mlp": X_mlp,
            "y": loaded["y"],
            "ids": loaded["ids"],
        }
