import os
import ast
import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from collections import Counter

from library.config import Config
from library.utils import print_log, seed_everything
from library.text_encoder import SBERTEncoder, TFIDFEncoder


class FeatureManager:
    def __init__(self):
        self.sbert_encoder = None
        self.tfidf_encoder = None
        self.top_k_subreddits = []
        self.scaler = StandardScaler()
        self.subreddit_embedding_map = {}

        # Ensure working directory exists
        os.makedirs(Config.WORKING_DIR, exist_ok=True)

    def load_raw_data(self):
        """
        Loads raw data from metadata CSVs.
        Parses list columns.
        """
        print_log("Loading raw metadata CSVs...")
        train_df = pd.read_csv(Config.TRAIN_PATH)
        val_df = pd.read_csv(Config.VAL_PATH)
        test_df = pd.read_csv(Config.TEST_PATH)

        # Parse list columns
        for col in Config.LIST_COLS:
            for df in [train_df, val_df, test_df]:
                if col in df.columns:
                    # Handle string representation of lists
                    df[col] = df[col].apply(
                        lambda x: (
                            ast.literal_eval(x)
                            if isinstance(x, str)
                            else (x if isinstance(x, list) else [])
                        )
                    )

        # Subsample for debugging if configured
        if Config.MAX_SAMPLES:
            print_log(
                f"Subsampling data to {Config.MAX_SAMPLES} samples for debugging."
            )
            train_df = train_df.head(Config.MAX_SAMPLES)
            val_df = val_df.head(Config.MAX_SAMPLES)
            test_df = test_df.head(Config.MAX_SAMPLES)

        return train_df, val_df, test_df

    def preprocess_metadata(self, df):
        """
        Generates basic numerical features, ratios, and handles missing values.
        """
        df = df.copy()

        # Fill NaNs in numeric columns with 0 or median
        for col in Config.NUMERIC_COLS:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        # 1. Ratios and Differences
        # Upvote Ratio: up / (up + down)
        # Avoid division by zero
        total_votes = df["requester_upvotes_plus_downvotes_at_request"]
        diff_votes = df["requester_upvotes_minus_downvotes_at_request"]

        # Recover upvotes and downvotes from sum and diff
        # sum = u + d, diff = u - d => u = (sum+diff)/2, d = (sum-diff)/2
        upvotes = (total_votes + diff_votes) / 2
        downvotes = (total_votes - diff_votes) / 2

        df["feature_upvote_ratio"] = np.where(
            total_votes > 0, upvotes / total_votes, 0.5
        )

        # 2. Log Transforms for skewed distributions (counts)
        skewed_cols = [
            "requester_number_of_comments_at_request",
            "requester_number_of_posts_at_request",
            "requester_upvotes_plus_downvotes_at_request",
        ]
        for col in skewed_cols:
            if col in df.columns:
                df[f"log_{col}"] = np.log1p(df[col])

        # 3. Text Meta-Features
        for text_col in Config.TEXT_COLS:
            if text_col in df.columns:
                # Fill NaN text
                df[text_col] = df[text_col].fillna("")
                # Length features
                df[f"len_char_{text_col}"] = df[text_col].apply(len)
                df[f"len_word_{text_col}"] = df[text_col].apply(
                    lambda x: len(str(x).split())
                )

        return df

    def _get_unique_subreddits(self, dfs):
        """Collects all unique subreddits from all datasets."""
        unique_subs = set()
        for df in dfs:
            for sub_list in df["requester_subreddits_at_request"]:
                unique_subs.update(sub_list)
        return list(unique_subs)

    def generate_sbert_features(self, train_df, val_df, test_df):
        """
        Generates SBERT embeddings for Title, Body, and User History.
        Computes Centroids and Consistency Scalars.
        """
        if self.sbert_encoder is None:
            self.sbert_encoder = SBERTEncoder()

        print_log("Generating SBERT embeddings...")

        all_dfs = [train_df, val_df, test_df]

        # 1. Embed Unique Subreddits
        unique_subs = self._get_unique_subreddits(all_dfs)
        print_log(f"Embedding {len(unique_subs)} unique subreddits...")
        # Batch encode subreddits
        sub_embeddings = self.sbert_encoder.encode(unique_subs, show_progress_bar=False)
        self.subreddit_embedding_map = {
            sub: emb for sub, emb in zip(unique_subs, sub_embeddings)
        }

        # 2. Process each dataframe
        results = []
        for df in all_dfs:
            # A. Title & Body Embeddings
            title_embs = self.sbert_encoder.encode(
                df["request_title"].tolist(), show_progress_bar=False
            )
            body_embs = self.sbert_encoder.encode(
                df["request_text_edit_aware"].tolist(), show_progress_bar=False
            )

            # B. History Centroids & Sequences
            centroid_embs = []
            history_seqs = []

            # Determine max history length for padding (or use a fixed logical max)
            # For this implementation, we'll pad to the max length found in the batch or a fixed size
            # Let's use a fixed max length for the sequence to keep memory managed
            MAX_SEQ_LEN = 20
            embedding_dim = sub_embeddings.shape[1]

            for sub_list in df["requester_subreddits_at_request"]:
                if not sub_list:
                    # No history
                    centroid_embs.append(np.zeros(embedding_dim, dtype=np.float32))
                    history_seqs.append(
                        np.zeros((MAX_SEQ_LEN, embedding_dim), dtype=np.float32)
                    )
                else:
                    # Gather embeddings
                    embs = [
                        self.subreddit_embedding_map.get(s, np.zeros(embedding_dim))
                        for s in sub_list
                    ]
                    embs = np.array(embs, dtype=np.float32)

                    # Centroid
                    centroid = np.mean(embs, axis=0)
                    centroid_embs.append(centroid)

                    # Sequence (Truncate or Pad)
                    seq_len = min(len(embs), MAX_SEQ_LEN)
                    padded_seq = np.zeros(
                        (MAX_SEQ_LEN, embedding_dim), dtype=np.float32
                    )
                    padded_seq[:seq_len] = embs[:seq_len]
                    history_seqs.append(padded_seq)

            centroid_embs = np.array(centroid_embs, dtype=np.float32)
            history_seqs = np.array(history_seqs, dtype=np.float32)

            # C. Consistency Scalars
            # Cosine Sim between Title/Body and Centroid
            # Reshape for pairwise calculation (n_samples, dim)
            # We want row-wise cosine similarity: (A . B) / (|A|*|B|)
            # Using sklearn cosine_similarity returns matrix; we want diagonal.
            # Optimized: sum(A*B, axis=1) / (norm(A)*norm(B))

            def cosine_sim_rows(a, b):
                # Add epsilon to avoid div by zero
                norm_a = np.linalg.norm(a, axis=1) + 1e-9
                norm_b = np.linalg.norm(b, axis=1) + 1e-9
                dot = np.sum(a * b, axis=1)
                return dot / (norm_a * norm_b)

            consistency_title = cosine_sim_rows(title_embs, centroid_embs)
            consistency_body = cosine_sim_rows(body_embs, centroid_embs)

            # Add scalars to DF for RF usage
            df["consistency_title"] = consistency_title
            df["consistency_body"] = consistency_body

            results.append(
                {
                    "title_emb": title_embs,
                    "body_emb": body_embs,
                    "centroid_emb": centroid_embs,
                    "history_seq": history_seqs,
                    "df": df,
                }
            )

        return results[0], results[1], results[2]

    def generate_top_k_features(self, train_df, val_df, test_df):
        """
        Identifies top K subreddits from train and adds binary indicators.
        """
        print_log(f"Generating Top-{Config.TOP_K_SUBREDDITS} subreddit indicators...")

        # Count frequencies in train
        all_subs = []
        for sub_list in train_df["requester_subreddits_at_request"]:
            all_subs.extend(sub_list)

        counts = Counter(all_subs)
        self.top_k_subreddits = [
            sub for sub, count in counts.most_common(Config.TOP_K_SUBREDDITS)
        ]

        # Create features
        for df in [train_df, val_df, test_df]:
            for sub in self.top_k_subreddits:
                # Clean column name
                col_name = f"sub_flag_{sub}"
                df[col_name] = df["requester_subreddits_at_request"].apply(
                    lambda x: 1 if sub in x else 0
                )

        return train_df, val_df, test_df

    def generate_interaction_features(self, df):
        """
        Creates cross-product features between Consistency Scalars and Credibility Metrics.
        """
        for metric in Config.INTERACTION_CREDIBILITY_METRICS:
            if metric in df.columns:
                # Interaction with Title Consistency
                df[f"interact_{metric}_x_title_cons"] = (
                    df[metric] * df["consistency_title"]
                )
                # Interaction with Body Consistency
                df[f"interact_{metric}_x_body_cons"] = (
                    df[metric] * df["consistency_body"]
                )
        return df

    def get_rf_dataset(self, load_cached_data=True):
        """
        Orchestrates the creation of the Random Forest dataset.
        Returns X_train, y_train, X_val, y_val, X_test, test_ids
        """
        # Check cache
        if load_cached_data and os.path.exists(Config.CACHE_RF_TRAIN):
            print_log("Loading cached RF data...")
            train_df = pd.read_parquet(Config.CACHE_RF_TRAIN)
            val_df = pd.read_parquet(Config.CACHE_RF_VAL)
            test_df = pd.read_parquet(Config.CACHE_RF_TEST)

            # Separate features and targets
            feature_cols = [
                c
                for c in train_df.columns
                if c not in [Config.ID_COL, Config.TARGET_COL]
            ]

            return (
                train_df[feature_cols],
                train_df[Config.TARGET_COL],
                val_df[feature_cols],
                val_df[Config.TARGET_COL],
                test_df[feature_cols],
                test_df[Config.ID_COL],
            )

        # Compute from scratch
        print_log("Computing RF data from scratch...")
        train_df, val_df, test_df = self.load_raw_data()

        # 1. Preprocess Metadata
        train_df = self.preprocess_metadata(train_df)
        val_df = self.preprocess_metadata(val_df)
        test_df = self.preprocess_metadata(test_df)

        # 2. SBERT Features (Needed for Consistency Scalars)
        # We discard the dense embeddings for RF, keeping only scalars and modified DF
        res_train, res_val, res_test = self.generate_sbert_features(
            train_df, val_df, test_df
        )
        train_df, val_df, test_df = res_train["df"], res_val["df"], res_test["df"]

        # 3. Top-K Features
        train_df, val_df, test_df = self.generate_top_k_features(
            train_df, val_df, test_df
        )

        # 4. Interaction Features
        train_df = self.generate_interaction_features(train_df)
        val_df = self.generate_interaction_features(val_df)
        test_df = self.generate_interaction_features(test_df)

        # 5. TF-IDF Features
        print_log("Generating TF-IDF features...")
        if self.tfidf_encoder is None:
            self.tfidf_encoder = TFIDFEncoder()

        # Combine title and body
        train_text = (
            train_df["request_title"] + " " + train_df["request_text_edit_aware"]
        )
        val_text = val_df["request_title"] + " " + val_df["request_text_edit_aware"]
        test_text = test_df["request_title"] + " " + test_df["request_text_edit_aware"]

        # Fit on train, transform all
        X_tfidf_train = self.tfidf_encoder.fit_transform(train_text)
        X_tfidf_val = self.tfidf_encoder.transform(val_text)
        X_tfidf_test = self.tfidf_encoder.transform(test_text)

        # Convert TF-IDF to DataFrame
        tfidf_cols = self.tfidf_encoder.get_feature_names_out()
        tfidf_cols = [f"tfidf_{c}" for c in tfidf_cols]

        # Helper to merge sparse TFIDF with Dense DF
        def merge_tfidf(df, sparse_matrix, cols):
            dense_tfidf = pd.DataFrame(
                sparse_matrix.toarray(), columns=cols, index=df.index
            )
            return pd.concat([df, dense_tfidf], axis=1)

        train_df = merge_tfidf(train_df, X_tfidf_train, tfidf_cols)
        val_df = merge_tfidf(val_df, X_tfidf_val, tfidf_cols)
        test_df = merge_tfidf(test_df, X_tfidf_test, tfidf_cols)

        # 6. Select Final Columns
        # Exclude raw text, list columns, and ID/Target (keep ID/Target for cache saving)
        exclude_types = ["object"]  # Exclude strings/lists
        numeric_train = train_df.select_dtypes(exclude=exclude_types)

        # Ensure ID and Target are present for saving, but handle their types
        # We actually want to save the full processed DF to parquet, then split on load
        # But we must drop the raw list/text columns that parquet might struggle with or are unnecessary
        drop_cols = Config.TEXT_COLS + Config.LIST_COLS + ["source_file"]

        final_train = train_df.drop(
            columns=[c for c in drop_cols if c in train_df.columns], errors="ignore"
        )
        final_val = val_df.drop(
            columns=[c for c in drop_cols if c in val_df.columns], errors="ignore"
        )
        final_test = test_df.drop(
            columns=[c for c in drop_cols if c in test_df.columns], errors="ignore"
        )

        # Ensure target is present (test set might not have it, but we don't need it there)

        # Save to Cache
        print_log("Saving RF data to cache...")
        final_train.to_parquet(Config.CACHE_RF_TRAIN)
        final_val.to_parquet(Config.CACHE_RF_VAL)
        final_test.to_parquet(Config.CACHE_RF_TEST)

        # Return
        feature_cols = [
            c
            for c in final_train.columns
            if c not in [Config.ID_COL, Config.TARGET_COL]
        ]

        return (
            final_train[feature_cols],
            final_train[Config.TARGET_COL],
            final_val[feature_cols],
            final_val[Config.TARGET_COL],
            final_test[feature_cols],
            final_test[Config.ID_COL],
        )

    def get_mlp_dataset(self, load_cached_data=True):
        """
        Orchestrates the creation of the MLP dataset.
        Returns dictionaries of arrays for Train, Val, Test.
        """
        if load_cached_data and os.path.exists(Config.CACHE_MLP_TRAIN):
            print_log("Loading cached MLP data...")
            train_data = np.load(Config.CACHE_MLP_TRAIN)
            val_data = np.load(Config.CACHE_MLP_VAL)
            test_data = np.load(Config.CACHE_MLP_TEST)
            return train_data, val_data, test_data

        print_log("Computing MLP data from scratch...")
        train_df, val_df, test_df = self.load_raw_data()

        # 1. Preprocess Metadata
        train_df = self.preprocess_metadata(train_df)
        val_df = self.preprocess_metadata(val_df)
        test_df = self.preprocess_metadata(test_df)

        # 2. SBERT Features (Embeddings + History)
        res_train, res_val, res_test = self.generate_sbert_features(
            train_df, val_df, test_df
        )

        # 3. Metadata Scaling
        # Select numeric columns for MLP (Metadata branch)
        # We include the base numeric cols + generated ratios + consistency scalars
        # Exclude Top-K and TF-IDF
        meta_cols = Config.NUMERIC_COLS + [
            "feature_upvote_ratio",
            "consistency_title",
            "consistency_body",
        ]
        # Add log cols
        meta_cols += [
            c
            for c in res_train["df"].columns
            if c.startswith("log_") or c.startswith("len_")
        ]

        X_meta_train = res_train["df"][meta_cols].values.astype(np.float32)
        X_meta_val = res_val["df"][meta_cols].values.astype(np.float32)
        X_meta_test = res_test["df"][meta_cols].values.astype(np.float32)

        # Fit Scaler on Train
        self.scaler.fit(X_meta_train)
        X_meta_train = self.scaler.transform(X_meta_train)
        X_meta_val = self.scaler.transform(X_meta_val)
        X_meta_test = self.scaler.transform(X_meta_test)

        # 4. Targets
        y_train = res_train["df"][Config.TARGET_COL].astype(int).values
        y_val = res_val["df"][Config.TARGET_COL].astype(int).values
        # Test target placeholder
        y_test = np.zeros(len(res_test["df"]))
        test_ids = res_test["df"][Config.ID_COL].values

        # 5. Pack Data
        def pack(res, meta, y, ids=None):
            data = {
                "title_emb": res["title_emb"],
                "body_emb": res["body_emb"],
                "history_seq": res["history_seq"],
                "centroid_emb": res["centroid_emb"],
                "metadata": meta,
                "y": y,
            }
            if ids is not None:
                data["ids"] = ids
            return data

        train_data = pack(res_train, X_meta_train, y_train)
        val_data = pack(res_val, X_meta_val, y_val)
        test_data = pack(res_test, X_meta_test, y_test, test_ids)

        # Save to Cache
        print_log("Saving MLP data to cache...")
        np.savez(Config.CACHE_MLP_TRAIN, **train_data)
        np.savez(Config.CACHE_MLP_VAL, **val_data)
        np.savez(Config.CACHE_MLP_TEST, **test_data)

        return train_data, val_data, test_data
