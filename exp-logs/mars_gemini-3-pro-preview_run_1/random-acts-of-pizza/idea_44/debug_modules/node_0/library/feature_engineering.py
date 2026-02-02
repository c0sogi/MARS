import os
import ast
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from library.config import Config
from library.utils import seed_everything


class FeatureEngineer:
    def __init__(self):
        self.config = Config
        seed_everything(self.config.RANDOM_SEED)

        # Define numerical columns to use (excluding leakage/retrieval columns)
        self.numeric_cols = [
            "requester_account_age_in_days_at_request",
            "requester_days_since_first_post_on_raop_at_request",
            "requester_number_of_comments_at_request",
            "requester_number_of_comments_in_raop_at_request",
            "requester_number_of_posts_at_request",
            "requester_number_of_posts_on_raop_at_request",
            "requester_number_of_subreddits_at_request",
            "requester_upvotes_minus_downvotes_at_request",
            "requester_upvotes_plus_downvotes_at_request",
        ]

    def _load_data(self):
        """Loads raw data from metadata CSVs."""
        print("Loading metadata CSVs...")
        train_df = pd.read_csv(self.config.TRAIN_PATH)
        val_df = pd.read_csv(self.config.VAL_PATH)
        test_df = pd.read_csv(self.config.TEST_PATH)

        # Parse stringified lists
        for df in [train_df, val_df, test_df]:
            df["requester_subreddits_at_request"] = df[
                "requester_subreddits_at_request"
            ].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])
            # Fill missing text
            df["request_text_edit_aware"] = df["request_text_edit_aware"].fillna("")
            df["request_title"] = df["request_title"].fillna("")

            # Create full text column for TF-IDF
            df["full_text"] = df["request_title"] + " " + df["request_text_edit_aware"]

        if self.config.DEBUG:
            print(f"DEBUG MODE: Truncating data to {self.config.DEBUG_SIZE} samples.")
            train_df = train_df.iloc[: self.config.DEBUG_SIZE]
            val_df = val_df.iloc[: self.config.DEBUG_SIZE]
            test_df = test_df.iloc[: self.config.DEBUG_SIZE]

        return train_df, val_df, test_df

    def _process_tfidf(self, train_df, val_df, test_df):
        """Generates TF-IDF features for Random Forest."""
        print("Generating TF-IDF features...")
        vectorizer = TfidfVectorizer(
            max_features=self.config.TFIDF_MAX_FEATURES,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )

        # Fit on train, transform all
        X_train = (
            vectorizer.fit_transform(train_df["full_text"]).toarray().astype(np.float32)
        )
        X_val = vectorizer.transform(val_df["full_text"]).toarray().astype(np.float32)
        X_test = vectorizer.transform(test_df["full_text"]).toarray().astype(np.float32)

        return X_train, X_val, X_test

    def _process_metadata(self, train_df, val_df, test_df):
        """
        Processes numerical metadata and Top-K subreddits.
        Returns:
            meta_scaled_dict: {split: scaled_numeric_matrix} for MLP
            meta_raw_df_dict: {split: dataframe_with_raw_and_topk} for RF
        """
        print("Processing metadata...")

        # 1. Numerical Metadata Processing (Impute -> Arcsinh -> Scale)
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()

        # Prepare raw matrices
        X_train_num = train_df[self.numeric_cols].values
        X_val_num = val_df[self.numeric_cols].values
        X_test_num = test_df[self.numeric_cols].values

        # Impute
        X_train_num = imputer.fit_transform(X_train_num)
        X_val_num = imputer.transform(X_val_num)
        X_test_num = imputer.transform(X_test_num)

        # Arcsinh transform (handle skew)
        X_train_log = np.arcsinh(X_train_num)
        X_val_log = np.arcsinh(X_val_num)
        X_test_log = np.arcsinh(X_test_num)

        # Scale
        X_train_scaled = scaler.fit_transform(X_train_log).astype(np.float32)
        X_val_scaled = scaler.transform(X_val_log).astype(np.float32)
        X_test_scaled = scaler.transform(X_test_log).astype(np.float32)

        meta_scaled = {
            "train": X_train_scaled,
            "val": X_val_scaled,
            "test": X_test_scaled,
        }

        # 2. Top-K Subreddits (Binary Flags)
        # Flatten all subreddits in train to find top K
        all_subs = [
            sub
            for sub_list in train_df["requester_subreddits_at_request"]
            for sub in sub_list
        ]
        top_k_subs = (
            pd.Series(all_subs)
            .value_counts()
            .head(self.config.TOP_K_SUBREDDITS)
            .index.tolist()
        )

        def get_topk_features(df):
            # Create a matrix of shape (N, K)
            features = np.zeros((len(df), len(top_k_subs)), dtype=np.float32)
            for i, row_subs in enumerate(df["requester_subreddits_at_request"]):
                row_subs_set = set(row_subs)
                for j, sub in enumerate(top_k_subs):
                    if sub in row_subs_set:
                        features[i, j] = 1.0
            return features, [f"sub_{s}" for s in top_k_subs]

        X_train_topk, topk_names = get_topk_features(train_df)
        X_val_topk, _ = get_topk_features(val_df)
        X_test_topk, _ = get_topk_features(test_df)

        # Combine Raw Numerical + TopK for RF
        # We use the imputed (but not scaled/arcsinh) numerical values for RF,
        # plus the TopK flags.
        def combine_rf_meta(num_arr, topk_arr):
            return np.hstack([num_arr, topk_arr]).astype(np.float32)

        meta_rf = {
            "train": combine_rf_meta(X_train_num, X_train_topk),
            "val": combine_rf_meta(X_val_num, X_val_topk),
            "test": combine_rf_meta(X_test_num, X_test_topk),
        }

        return meta_scaled, meta_rf, topk_names

    def _process_sbert_and_history(self, train_df, val_df, test_df):
        """
        Generates SBERT embeddings for Title, Body, and History.
        Computes Centroids and Consistency Scores.
        """
        print("Generating SBERT embeddings and History features...")
        model = SentenceTransformer(
            self.config.SBERT_MODEL_NAME, device=self.config.DEVICE
        )

        # 1. Encode Title and Body
        print("  Encoding Titles...")
        title_emb = {
            "train": model.encode(
                train_df["request_title"].tolist(), show_progress_bar=False
            ),
            "val": model.encode(
                val_df["request_title"].tolist(), show_progress_bar=False
            ),
            "test": model.encode(
                test_df["request_title"].tolist(), show_progress_bar=False
            ),
        }

        print("  Encoding Bodies...")
        body_emb = {
            "train": model.encode(
                train_df["request_text_edit_aware"].tolist(), show_progress_bar=False
            ),
            "val": model.encode(
                val_df["request_text_edit_aware"].tolist(), show_progress_bar=False
            ),
            "test": model.encode(
                test_df["request_text_edit_aware"].tolist(), show_progress_bar=False
            ),
        }

        # 2. Process History (Subreddits)
        print("  Processing History...")
        # Collect all unique subreddits
        unique_subs = set()
        for df in [train_df, val_df, test_df]:
            for sub_list in df["requester_subreddits_at_request"]:
                unique_subs.update(sub_list)

        unique_subs_list = list(unique_subs)
        if not unique_subs_list:
            # Handle edge case where no subreddits exist
            sub_to_vec = {}
            embedding_dim = self.config.SBERT_EMBEDDING_DIM
        else:
            print(f"  Encoding {len(unique_subs_list)} unique subreddits...")
            sub_vecs = model.encode(
                unique_subs_list, batch_size=256, show_progress_bar=False
            )
            sub_to_vec = {sub: vec for sub, vec in zip(unique_subs_list, sub_vecs)}
            embedding_dim = sub_vecs.shape[1]

        # Helper to process a single dataframe
        def process_user_history(df, titles, bodies):
            num_samples = len(df)
            max_seq_len = self.config.TOP_K_SUBREDDITS  # Use 50 as max history length

            # Arrays for output
            history_seq = np.zeros(
                (num_samples, max_seq_len, embedding_dim), dtype=np.float32
            )
            centroids = np.zeros((num_samples, embedding_dim), dtype=np.float32)
            consistency_title = np.zeros((num_samples, 1), dtype=np.float32)
            consistency_body = np.zeros((num_samples, 1), dtype=np.float32)

            for i, sub_list in enumerate(df["requester_subreddits_at_request"]):
                if not sub_list:
                    continue

                # Get vectors for this user's subreddits
                # Take last K subreddits if list is long (most recent usually at end or just truncate)
                # Dataset doesn't specify order, assume list order. We take up to max_seq_len.
                current_vecs = []
                for sub in sub_list[:max_seq_len]:
                    if sub in sub_to_vec:
                        current_vecs.append(sub_to_vec[sub])

                if not current_vecs:
                    continue

                current_vecs = np.array(current_vecs)

                # Fill Sequence (Padding is already 0)
                seq_len = min(len(current_vecs), max_seq_len)
                history_seq[i, :seq_len, :] = current_vecs[:seq_len]

                # Compute Centroid
                centroid = np.mean(current_vecs, axis=0)
                centroids[i] = centroid

                # Compute Consistency (Cosine Sim)
                # Reshape for sklearn cosine_similarity (1, D)
                c_vec = centroid.reshape(1, -1)
                t_vec = titles[i].reshape(1, -1)
                b_vec = bodies[i].reshape(1, -1)

                consistency_title[i] = cosine_similarity(t_vec, c_vec)[0, 0]
                consistency_body[i] = cosine_similarity(b_vec, c_vec)[0, 0]

            return history_seq, centroids, consistency_title, consistency_body

        history_data = {}
        for split, df in zip(["train", "val", "test"], [train_df, val_df, test_df]):
            hist, cent, cons_t, cons_b = process_user_history(
                df, title_emb[split], body_emb[split]
            )
            history_data[split] = {
                "history_seq": hist,
                "centroid": cent,
                "consistency_title": cons_t,
                "consistency_body": cons_b,
            }

        return title_emb, body_emb, history_data

    def _process_interactions(self, meta_rf, history_data):
        """
        Generates explicit interaction features for Random Forest.
        Interaction = Consistency * log(1 + Metadata)
        """
        print("Generating Interaction features...")
        interaction_data = {}

        for split in ["train", "val", "test"]:
            # Extract components
            # meta_rf[split] contains [Numerical (9 cols) | TopK (50 cols)]
            # We only want to interact with the Numerical columns
            num_cols_count = len(self.numeric_cols)
            raw_meta = meta_rf[split][:, :num_cols_count]

            cons_t = history_data[split]["consistency_title"]  # (N, 1)
            cons_b = history_data[split]["consistency_body"]  # (N, 1)

            # Log transform metadata for interaction (log(1+x) to handle zeros)
            # Clip negative values to 0 before log to avoid errors (though input shouldn't be neg for counts)
            # Some cols like upvotes-downvotes can be negative.
            # We'll use absolute value or offset for log.
            # Strategy: log(1 + abs(x)) * sign(x) is safer, or just interact with raw.
            # Prompt suggests: log(1 + Metadata). Let's assume non-negative or shift.
            # For simplicity and robustness: log(1 + abs(x))
            log_meta = np.log1p(np.abs(raw_meta))

            # Interactions:
            # 1. Title_Consistency * Metadata
            # 2. Body_Consistency * Metadata

            int_t = log_meta * cons_t
            int_b = log_meta * cons_b

            # Concatenate
            interactions = np.hstack([int_t, int_b]).astype(np.float32)
            interaction_data[split] = interactions

        return interaction_data

    def run(self, load_cached_data=True):
        """
        Main execution method.
        Checks cache, loads if available, else computes and saves.
        """
        # Ensure cache directory exists
        os.makedirs(self.config.CACHE_DIR, exist_ok=True)

        # Define cache file paths
        cache_files = {
            "tfidf": self.config.CACHE_TFIDF_FEATURES,
            "meta": self.config.CACHE_METADATA_FEATURES,
            "sbert": self.config.CACHE_SBERT_EMBEDDINGS,
            "persona": self.config.CACHE_PERSONA_FEATURES,
            "interaction": self.config.CACHE_INTERACTION_FEATURES,
        }

        # Check if all exist
        all_exist = all(os.path.exists(p) for p in cache_files.values())

        if load_cached_data and all_exist:
            print("Loading features from cache...")
            tfidf_data = np.load(cache_files["tfidf"])
            meta_data = pd.read_parquet(
                cache_files["meta"]
            )  # Storing dicts in parquet is tricky, using npz for arrays
            # Correction: Parquet is for tables. Let's use npz for everything to be safe and consistent with arrays.
            # Re-defining load logic based on file extensions in Config
            # Config defines:
            # CACHE_SBERT_EMBEDDINGS = ...npz
            # CACHE_TFIDF_FEATURES = ...npz
            # CACHE_TOPK_FEATURES = ...parquet (This was in config, but I will bundle TopK into meta npz for simplicity)
            # CACHE_METADATA_FEATURES = ...parquet

            # To strictly follow config paths:
            # I will assume the user wants me to use those paths.
            # However, npz is much better for storing multiple arrays (train/val/test).
            # I will load what I saved.

            # Let's just implement the loading logic based on what I WILL save below.
            # I will save everything as .npz for array data.

            try:
                # Load TFIDF
                f_tfidf = np.load(self.config.CACHE_TFIDF_FEATURES)
                X_tfidf = {k: f_tfidf[k] for k in ["train", "val", "test"]}

                # Load Meta (Scaled for MLP, RF combined)
                # I'll save these in one npz
                f_meta = np.load(
                    self.config.CACHE_METADATA_FEATURES.replace(".parquet", ".npz")
                )
                meta_scaled = {
                    k: f_meta[f"scaled_{k}"] for k in ["train", "val", "test"]
                }
                meta_rf = {k: f_meta[f"rf_{k}"] for k in ["train", "val", "test"]}

                # Load SBERT
                f_sbert = np.load(self.config.CACHE_SBERT_EMBEDDINGS)
                title_emb = {k: f_sbert[f"title_{k}"] for k in ["train", "val", "test"]}
                body_emb = {k: f_sbert[f"body_{k}"] for k in ["train", "val", "test"]}

                # Load Persona/History
                f_persona = np.load(self.config.CACHE_PERSONA_FEATURES)
                history_data = {}
                for split in ["train", "val", "test"]:
                    history_data[split] = {
                        "history_seq": f_persona[f"seq_{split}"],
                        "centroid": f_persona[f"cent_{split}"],
                        "consistency_title": f_persona[f"const_{split}"],
                        "consistency_body": f_persona[f"consb_{split}"],
                    }

                # Load Interactions
                f_inter = np.load(
                    self.config.CACHE_INTERACTION_FEATURES.replace(".parquet", ".npz")
                )
                interaction_data = {k: f_inter[k] for k in ["train", "val", "test"]}

                # Load Targets (fast enough to reload from CSV)
                train_df, val_df, test_df = self._load_data()
                y_train = train_df["requester_received_pizza"].astype(int).values
                y_val = val_df["requester_received_pizza"].astype(int).values

                print("Cache loaded successfully.")

            except Exception as e:
                print(f"Failed to load cache: {e}. Recomputing...")
                return self.run(load_cached_data=False)

        else:
            print("Computing features from scratch...")
            train_df, val_df, test_df = self._load_data()

            # Compute
            X_tfidf_train, X_tfidf_val, X_tfidf_test = self._process_tfidf(
                train_df, val_df, test_df
            )
            X_tfidf = {"train": X_tfidf_train, "val": X_tfidf_val, "test": X_tfidf_test}

            meta_scaled, meta_rf, _ = self._process_metadata(train_df, val_df, test_df)

            title_emb, body_emb, history_data = self._process_sbert_and_history(
                train_df, val_df, test_df
            )

            interaction_data = self._process_interactions(meta_rf, history_data)

            # Save to Cache
            print("Saving to cache...")
            np.savez(self.config.CACHE_TFIDF_FEATURES, **X_tfidf)

            # Save Meta (modify path to .npz for array storage)
            np.savez(
                self.config.CACHE_METADATA_FEATURES.replace(".parquet", ".npz"),
                scaled_train=meta_scaled["train"],
                scaled_val=meta_scaled["val"],
                scaled_test=meta_scaled["test"],
                rf_train=meta_rf["train"],
                rf_val=meta_rf["val"],
                rf_test=meta_rf["test"],
            )

            # Save SBERT
            sbert_dict = {}
            for k in ["train", "val", "test"]:
                sbert_dict[f"title_{k}"] = title_emb[k]
                sbert_dict[f"body_{k}"] = body_emb[k]
            np.savez(self.config.CACHE_SBERT_EMBEDDINGS, **sbert_dict)

            # Save Persona
            persona_dict = {}
            for k in ["train", "val", "test"]:
                persona_dict[f"seq_{k}"] = history_data[k]["history_seq"]
                persona_dict[f"cent_{k}"] = history_data[k]["centroid"]
                persona_dict[f"const_{k}"] = history_data[k]["consistency_title"]
                persona_dict[f"consb_{k}"] = history_data[k]["consistency_body"]
            np.savez(self.config.CACHE_PERSONA_FEATURES, **persona_dict)

            # Save Interactions
            np.savez(
                self.config.CACHE_INTERACTION_FEATURES.replace(".parquet", ".npz"),
                **interaction_data,
            )

            y_train = train_df["requester_received_pizza"].astype(int).values
            y_val = val_df["requester_received_pizza"].astype(int).values

        # Assemble Output
        output = {}
        for split in ["train", "val", "test"]:
            # Stream A: Random Forest Inputs
            # RF Input = TFIDF + Meta(Raw+TopK) + Interactions
            X_rf = np.hstack(
                [X_tfidf[split], meta_rf[split], interaction_data[split]]
            ).astype(np.float32)

            # Stream B: MLP Inputs
            # MLP Needs: Title, Body, HistorySeq, Centroid, Meta(Scaled)
            # We return them separately for the Dataset class to handle

            split_data = {
                "X_rf": X_rf,
                "X_mlp_title": title_emb[split],
                "X_mlp_body": body_emb[split],
                "X_mlp_history": history_data[split]["history_seq"],
                "X_mlp_centroid": history_data[split]["centroid"],
                "X_mlp_meta": meta_scaled[split],
                "X_mlp_consistency": np.hstack(
                    [
                        history_data[split]["consistency_title"],
                        history_data[split]["consistency_body"],
                    ]
                ),
            }

            if split != "test":
                # y is same for both streams
                target = y_train if split == "train" else y_val
                split_data["y"] = target

            output[split] = split_data

        return output
