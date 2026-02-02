import os
import ast
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from library import config, data_loader, utils


class FeatureProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.tfidf = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words="english",
            min_df=5,
            max_df=0.9,
        )
        # SBERT model is loaded lazily or just initialized here
        self.sbert_model_name = "all-MiniLM-L6-v2"
        self.feature_cols = []

    def _remove_leakage(self, df, is_test=False):
        """
        Removes columns that contain leakage or future information.
        """
        drop_cols = config.DROP_COLS.copy()

        # Identify columns matching leakage keywords
        for col in df.columns:
            for keyword in config.LEAKAGE_KEYWORDS:
                if keyword in col:
                    drop_cols.append(col)
                    break

        # Drop identified columns
        # We keep the target column separate, so we can drop it from X later
        cols_to_drop = [c for c in drop_cols if c in df.columns]
        df_clean = df.drop(columns=cols_to_drop)

        return df_clean

    def _parse_subreddits(self, sub_str):
        """Parses stringified list of subreddits."""
        try:
            if pd.isna(sub_str) or sub_str == "":
                return []
            return ast.literal_eval(sub_str)
        except:
            return []

    def _generate_ratios(self, df):
        """
        Generates ratio-centric behavioral features and text meta-features.
        """
        # Ensure we work on a copy to avoid SettingWithCopy warnings
        X = df.copy()

        # --- 1. Behavioral Ratios ---
        # Recover raw upvotes/downvotes from Sum and Diff
        # Sum = Up + Down, Diff = Up - Down
        # Up = (Sum + Diff) / 2
        # Down = (Sum - Diff) / 2

        # Fill NaNs with 0 for calculation safety
        up_plus_down = X.get("requester_upvotes_plus_downvotes_at_request", 0).fillna(0)
        up_minus_down = X.get("requester_upvotes_minus_downvotes_at_request", 0).fillna(
            0
        )

        upvotes = (up_plus_down + up_minus_down) / 2
        downvotes = (up_plus_down - up_minus_down) / 2

        # Upvote Ratio (avoid div by zero)
        X["feat_upvote_ratio"] = upvotes / (up_plus_down + 1e-5)

        # Interaction Ratio: Comments / Posts
        n_comments = X.get("requester_number_of_comments_at_request", 0).fillna(0)
        n_posts = X.get("requester_number_of_posts_at_request", 0).fillna(0)
        X["feat_interaction_ratio"] = n_comments / (n_posts + 1e-5)

        # Activity Age Ratio: Total Activity / Account Age
        acc_age = X.get("requester_account_age_in_days_at_request", 0).fillna(0)
        X["feat_activity_age_ratio"] = (n_comments + n_posts) / (acc_age + 1.0)

        # --- 2. Community Ratios ---
        # Define altruism/need keywords for subreddits
        altruism_keywords = {
            "assistance",
            "food",
            "loan",
            "borrow",
            "random",
            "pantry",
            "help",
        }

        def calculate_community_score(row):
            subs = self._parse_subreddits(
                row.get("requester_subreddits_at_request", "[]")
            )
            if not subs:
                return 0.0
            count = sum(
                1 for s in subs if any(k in s.lower() for k in altruism_keywords)
            )
            return count / len(subs)

        X["feat_community_altruism_ratio"] = X.apply(calculate_community_score, axis=1)
        X["feat_num_subreddits"] = X.get(
            "requester_number_of_subreddits_at_request", 0
        ).fillna(0)

        # --- 3. Text Meta-Features ---
        text_col = config.TEXT_COL
        # Ensure text column is string
        texts = X[text_col].fillna("").astype(str)

        X["feat_text_len_char"] = texts.apply(len)
        X["feat_text_len_word"] = texts.apply(lambda x: len(x.split()))
        X["feat_text_caps_ratio"] = texts.apply(
            lambda x: sum(1 for c in x if c.isupper()) / (len(x) + 1e-5)
        )
        X["feat_text_shout_ratio"] = texts.apply(
            lambda x: x.count("!") / (len(x) + 1e-5)
        )

        # Select generated numeric feature columns
        feat_cols = [c for c in X.columns if c.startswith("feat_")]

        # Identify Raw Numeric Columns (Cite solution_lesson_node_00021)
        # We must preserve raw magnitude features alongside ratios.
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

        raw_cols = []
        for col in numeric_cols:
            # Skip target, leakage, drop cols, and already selected feat_ cols
            if col == config.TARGET_COL:
                continue
            if any(k in col for k in config.LEAKAGE_KEYWORDS):
                continue
            if col in config.DROP_COLS:
                continue
            if col in feat_cols:
                continue
            raw_cols.append(col)

        # Combine raw and engineered features
        final_cols = raw_cols + feat_cols

        # Handle any remaining NaNs/Infs in features
        X_feats = X[final_cols].replace([np.inf, -np.inf], 0).fillna(0)

        return X_feats, final_cols

    def _process_text_features(self, train_df, val_df, test_df):
        """
        Generates TF-IDF and SBERT embeddings.
        """
        text_col = config.TEXT_COL
        train_text = train_df[text_col].fillna("").astype(str).tolist()
        val_text = val_df[text_col].fillna("").astype(str).tolist()
        test_text = test_df[text_col].fillna("").astype(str).tolist()

        # 1. TF-IDF (Sparse)
        print("Fitting TF-IDF...")
        self.tfidf.fit(train_text)
        tfidf_train = self.tfidf.transform(train_text)
        tfidf_val = self.tfidf.transform(val_text)
        tfidf_test = self.tfidf.transform(test_text)

        # 2. SBERT (Dense)
        # Check if embeddings are already cached on disk by config paths to save time
        # However, this method is called inside process_data which handles the main cache logic.
        # We will compute them here.
        print("Generating SBERT embeddings...")
        model = SentenceTransformer(
            self.sbert_model_name, device=config.MLP_CONFIG["device"]
        )

        emb_train = model.encode(train_text, batch_size=64, show_progress_bar=False)
        emb_val = model.encode(val_text, batch_size=64, show_progress_bar=False)
        emb_test = model.encode(test_text, batch_size=64, show_progress_bar=False)

        return (tfidf_train, tfidf_val, tfidf_test), (emb_train, emb_val, emb_test)

    def process_data(self, load_cached_data=True):
        """
        Main orchestration method.
        Checks for cached numpy/npz files. If missing, processes from scratch.
        """
        # Define cache paths for the processed matrices
        cache_dir = config.WORKING_DIR
        paths = {
            "train_dense": os.path.join(cache_dir, "X_dense_train.npy"),
            "val_dense": os.path.join(cache_dir, "X_dense_val.npy"),
            "test_dense": os.path.join(cache_dir, "X_dense_test.npy"),
            "train_tfidf": os.path.join(cache_dir, "X_tfidf_train.npz"),
            "val_tfidf": os.path.join(cache_dir, "X_tfidf_val.npz"),
            "test_tfidf": os.path.join(cache_dir, "X_tfidf_test.npz"),
            "train_emb": os.path.join(cache_dir, "X_emb_train.npy"),
            "val_emb": os.path.join(cache_dir, "X_emb_val.npy"),
            "test_emb": os.path.join(cache_dir, "X_emb_test.npy"),
            "train_y": os.path.join(cache_dir, "y_train.npy"),
            "val_y": os.path.join(cache_dir, "y_val.npy"),
            "test_ids": os.path.join(cache_dir, "test_ids.npy"),
        }

        all_exist = all(os.path.exists(p) for p in paths.values())

        if load_cached_data and all_exist:
            print("Loading processed features from cache...")
            data = {
                "train": {
                    "dense": np.load(paths["train_dense"]),
                    "tfidf": sparse.load_npz(paths["train_tfidf"]),
                    "embedding": np.load(paths["train_emb"]),
                    "y": np.load(paths["train_y"]),
                },
                "val": {
                    "dense": np.load(paths["val_dense"]),
                    "tfidf": sparse.load_npz(paths["val_tfidf"]),
                    "embedding": np.load(paths["val_emb"]),
                    "y": np.load(paths["val_y"]),
                },
                "test": {
                    "dense": np.load(paths["test_dense"]),
                    "tfidf": sparse.load_npz(paths["test_tfidf"]),
                    "embedding": np.load(paths["test_emb"]),
                    "ids": np.load(
                        paths["test_ids"], allow_pickle=True
                    ),  # IDs are strings
                },
            }
            return data

        print("Processing data from scratch...")

        # 1. Load Metadata
        train_df, val_df, test_df = data_loader.load_metadata_splits(
            load_cached_data=load_cached_data
        )

        # 2. Extract Targets and IDs
        y_train = train_df[config.TARGET_COL].values
        y_val = val_df[config.TARGET_COL].values
        test_ids = test_df[config.ID_COL].values

        # 3. Generate Dense Ratio Features
        print("Generating dense ratio features...")
        X_dense_train_df, feat_cols = self._generate_ratios(train_df)
        X_dense_val_df, _ = self._generate_ratios(val_df)
        X_dense_test_df, _ = self._generate_ratios(test_df)

        # Align columns (ensure test/val have same columns as train)
        # (Though _generate_ratios should be deterministic based on logic)
        X_dense_val_df = X_dense_val_df[feat_cols]
        X_dense_test_df = X_dense_test_df[feat_cols]

        # Scale Dense Features
        print("Scaling dense features...")
        self.scaler.fit(X_dense_train_df)
        X_dense_train = self.scaler.transform(X_dense_train_df)
        X_dense_val = self.scaler.transform(X_dense_val_df)
        X_dense_test = self.scaler.transform(X_dense_test_df)

        # 4. Generate Text Features (TF-IDF and SBERT)
        (X_tfidf_train, X_tfidf_val, X_tfidf_test), (
            X_emb_train,
            X_emb_val,
            X_emb_test,
        ) = self._process_text_features(train_df, val_df, test_df)

        # 5. Save to Cache
        print(f"Saving features to {cache_dir}...")
        np.save(paths["train_dense"], X_dense_train)
        np.save(paths["val_dense"], X_dense_val)
        np.save(paths["test_dense"], X_dense_test)

        sparse.save_npz(paths["train_tfidf"], X_tfidf_train)
        sparse.save_npz(paths["val_tfidf"], X_tfidf_val)
        sparse.save_npz(paths["test_tfidf"], X_tfidf_test)

        np.save(paths["train_emb"], X_emb_train)
        np.save(paths["val_emb"], X_emb_val)
        np.save(paths["test_emb"], X_emb_test)

        np.save(paths["train_y"], y_train)
        np.save(paths["val_y"], y_val)
        np.save(paths["test_ids"], test_ids)

        # 6. Return Data Structure
        data = {
            "train": {
                "dense": X_dense_train,
                "tfidf": X_tfidf_train,
                "embedding": X_emb_train,
                "y": y_train,
            },
            "val": {
                "dense": X_dense_val,
                "tfidf": X_tfidf_val,
                "embedding": X_emb_val,
                "y": y_val,
            },
            "test": {
                "dense": X_dense_test,
                "tfidf": X_tfidf_test,
                "embedding": X_emb_test,
                "ids": test_ids,
            },
        }
        return data
