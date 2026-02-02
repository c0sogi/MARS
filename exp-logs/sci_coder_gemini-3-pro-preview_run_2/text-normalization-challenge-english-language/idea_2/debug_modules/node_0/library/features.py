import os
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

from library.config import Config
from library.utils import load_dataset, safe_hash, set_seed


class FeaturePipeline:
    """
    Manages the feature engineering pipeline for the Text Normalization task.
    Handles data loading, processing, feature extraction, and caching.
    """

    def __init__(self):
        self.working_dir = Config.WORKING_DIR
        self.vocab_path = os.path.join(self.working_dir, "vectorizer_vocab.json")
        self.encoder_path = Config.LABEL_ENCODER_PATH

        # Ensure working directory exists
        os.makedirs(self.working_dir, exist_ok=True)

    def _get_context_features(self, df):
        """
        Generates context features for the previous and next tokens using vectorized operations.
        """
        window = Config.CONTEXT_WINDOW

        # We will collect new feature columns in a dictionary to concat at once (more efficient)
        new_cols = {}

        # Pre-calculate hashes for the current tokens to shift them later
        # Using map with safe_hash is reasonably fast
        token_hashes = (
            df["before"].astype(str).map(lambda x: safe_hash(x, num_buckets=10000))
        )

        for i in range(1, window + 1):
            # --- Previous i-th token ---
            # Shift data
            prev_token = df["before"].shift(i).fillna("")
            prev_sent = df["sentence_id"].shift(i)
            prev_hash = token_hashes.shift(i).fillna(0).astype(int)

            # Mask where sentence boundary is crossed
            mask = prev_sent == df["sentence_id"]

            # Apply mask (if different sentence, context is empty/padding)
            # We use specific values for padding
            p_token = prev_token.where(mask, "")
            p_hash = prev_hash.where(mask, 0)

            # Extract features
            new_cols[f"prev_{i}_is_digit"] = p_token.str.isdigit().astype(int)
            new_cols[f"prev_{i}_is_title"] = p_token.str.istitle().astype(int)
            new_cols[f"prev_{i}_len"] = p_token.str.len()
            new_cols[f"prev_{i}_hash"] = p_hash

            # --- Next i-th token ---
            next_token = df["before"].shift(-i).fillna("")
            next_sent = df["sentence_id"].shift(-i)
            next_hash = token_hashes.shift(-i).fillna(0).astype(int)

            mask_next = next_sent == df["sentence_id"]

            n_token = next_token.where(mask_next, "")
            n_hash = next_hash.where(mask_next, 0)

            new_cols[f"next_{i}_is_digit"] = n_token.str.isdigit().astype(int)
            new_cols[f"next_{i}_is_title"] = n_token.str.istitle().astype(int)
            new_cols[f"next_{i}_len"] = n_token.str.len()
            new_cols[f"next_{i}_hash"] = n_hash

        return pd.DataFrame(new_cols, index=df.index)

    def _extract_orthographic_features(self, df):
        """
        Extracts basic orthographic features from the target token.
        """
        s = df["before"].astype(str)

        feats = pd.DataFrame(index=df.index)
        feats["len"] = s.str.len()
        feats["is_digit"] = s.str.isdigit().astype(int)
        feats["is_alpha"] = s.str.isalpha().astype(int)
        feats["is_upper"] = s.str.isupper().astype(int)
        feats["is_title"] = s.str.istitle().astype(int)
        feats["count_digits"] = s.str.count(r"\d")
        feats["has_digit"] = (feats["count_digits"] > 0).astype(int)
        feats["has_punct"] = s.str.contains(r"[^\w\s]").astype(int)

        # Safe hash of the current token
        feats["token_hash"] = s.map(lambda x: safe_hash(x, num_buckets=10000))

        return feats

    def _get_ngram_features(self, df, fit=False):
        """
        Generates character n-gram features using CountVectorizer.
        """
        text_data = df["before"].astype(str).fillna("")

        if fit:
            # Initialize and fit vectorizer
            vectorizer = CountVectorizer(
                analyzer="char",
                ngram_range=Config.CHAR_NGRAM_RANGE,
                max_features=Config.MAX_TEXT_FEATURES,
                binary=True,  # Presence is usually more robust than count for short tokens
            )
            print("Fitting CountVectorizer on training data...")
            ngram_sparse = vectorizer.fit_transform(text_data)

            # Save vocabulary
            vocab = vectorizer.get_feature_names_out().tolist()
            with open(self.vocab_path, "w") as f:
                json.dump(vocab, f)
            print(f"Vocabulary saved to {self.vocab_path}")

        else:
            # Load vocabulary
            if not os.path.exists(self.vocab_path):
                raise FileNotFoundError(
                    f"Vocabulary not found at {self.vocab_path}. Process train set first."
                )

            with open(self.vocab_path, "r") as f:
                vocab = json.load(f)

            # Reconstruct vectorizer with fixed vocabulary
            vectorizer = CountVectorizer(
                analyzer="char",
                ngram_range=Config.CHAR_NGRAM_RANGE,
                vocabulary=vocab,
                binary=True,
            )
            ngram_sparse = vectorizer.transform(text_data)

        # Convert to dense dataframe (memory safe due to MAX_TEXT_FEATURES limit)
        # Using float32 to save memory
        feature_names = [f"ngram_{i}" for i in range(ngram_sparse.shape[1])]
        ngram_df = pd.DataFrame(
            ngram_sparse.toarray(),
            index=df.index,
            columns=feature_names,
            dtype=np.float32,
        )

        return ngram_df

    def _process_dataframe(self, df, is_train=False):
        """
        Internal method to drive the feature generation for a given dataframe.
        """
        print(f"Generating orthographic features for {len(df)} rows...")
        ortho_feats = self._extract_orthographic_features(df)

        print("Generating context features...")
        context_feats = self._get_context_features(df)

        print("Generating n-gram features...")
        ngram_feats = self._get_ngram_features(df, fit=is_train)

        # Concatenate all features
        X = pd.concat([ortho_feats, context_feats, ngram_feats], axis=1)

        # Handle Target (y) if class exists
        y = None
        if "class" in df.columns:
            if is_train:
                print("Encoding labels...")
                le = LabelEncoder()
                y = le.fit_transform(df["class"])
                # Save classes
                np.save(self.encoder_path, le.classes_)
            else:
                # Load classes
                if os.path.exists(self.encoder_path):
                    classes = np.load(self.encoder_path, allow_pickle=True)
                    le = LabelEncoder()
                    le.classes_ = classes
                    # Handle unseen labels in validation (though unlikely given split)
                    # We map them to a safe default or error.
                    # For this task, we assume validation classes are subset of train.
                    y = le.transform(df["class"])
                else:
                    raise FileNotFoundError(
                        "Label encoder classes not found. Process train set first."
                    )

        return X, y

    def get_train_data(self, load_cached_data=True):
        """
        Returns processed training features (X) and labels (y).
        Applies downsampling to the raw data before processing.
        """
        output_path = Config.TRAIN_FEATURES_PATH

        # 1. Try Loading Cache
        if load_cached_data and os.path.exists(output_path):
            print(f"Loading cached training data from {output_path}...")
            df_full = pd.read_parquet(output_path)
            # Assume last column is target 'label_class' if we saved it together
            if "label_class" in df_full.columns:
                y = df_full["label_class"].values
                X = df_full.drop(columns=["label_class"])
                return X, y

        # 2. Compute from Scratch
        print("Processing training data from scratch...")
        df = load_dataset(Config.TRAIN_DATA_PATH)

        # Debug limit
        if Config.DEBUG_ROW_LIMIT:
            print(f"DEBUG: Limiting to {Config.DEBUG_ROW_LIMIT} rows.")
            df = df.head(Config.DEBUG_ROW_LIMIT)

        # Downsample PLAIN class
        print("Downsampling PLAIN class...")
        plain_mask = df["class"] == "PLAIN"
        df_plain = df[plain_mask].sample(
            frac=Config.PLAIN_DOWNSAMPLE_RATIO, random_state=Config.SEED
        )
        df_others = df[~plain_mask]
        df_balanced = (
            pd.concat([df_plain, df_others])
            .sample(frac=1.0, random_state=Config.SEED)
            .reset_index(drop=True)
        )

        print(f"Training set size after balancing: {len(df_balanced)}")

        X, y = self._process_dataframe(df_balanced, is_train=True)

        # Save to cache
        # Combine X and y for single file parquet save
        df_save = X.copy()
        df_save["label_class"] = y
        df_save.to_parquet(output_path, index=False)
        print(f"Saved processed training data to {output_path}")

        return X, y

    def get_val_data(self, load_cached_data=True):
        """
        Returns processed validation features (X) and labels (y).
        """
        output_path = Config.VAL_FEATURES_PATH

        if load_cached_data and os.path.exists(output_path):
            print(f"Loading cached validation data from {output_path}...")
            df_full = pd.read_parquet(output_path)
            if "label_class" in df_full.columns:
                y = df_full["label_class"].values
                X = df_full.drop(columns=["label_class"])
                return X, y

        print("Processing validation data from scratch...")
        df = load_dataset(Config.VAL_DATA_PATH)

        if Config.DEBUG_ROW_LIMIT:
            df = df.head(Config.DEBUG_ROW_LIMIT)

        X, y = self._process_dataframe(df, is_train=False)

        df_save = X.copy()
        df_save["label_class"] = y
        df_save.to_parquet(output_path, index=False)
        print(f"Saved processed validation data to {output_path}")

        return X, y

    def get_test_data(self, load_cached_data=True):
        """
        Returns processed test features (X) and the original ID column for submission.
        """
        output_path = Config.TEST_FEATURES_PATH

        # For test, we might also want to return the raw 'before' text or IDs for submission mapping
        # But usually we just need X for prediction and IDs for the file.
        # We will return X and a DataFrame containing 'id' and 'before' (for dictionary lookup).

        if load_cached_data and os.path.exists(output_path):
            print(f"Loading cached test data from {output_path}...")
            # We need to separate features from metadata
            # We assume we saved them together but distinct columns
            df_full = pd.read_parquet(output_path)

            meta_cols = ["id_ref", "before_ref"]
            meta_df = df_full[meta_cols].rename(
                columns={"id_ref": "id", "before_ref": "before"}
            )
            X = df_full.drop(columns=meta_cols)
            return X, meta_df

        print("Processing test data from scratch...")
        df = load_dataset(Config.TEST_DATA_PATH)

        if Config.DEBUG_ROW_LIMIT:
            df = df.head(Config.DEBUG_ROW_LIMIT)

        X, _ = self._process_dataframe(df, is_train=False)

        # Save cache
        # We attach ID and Before text to the parquet so we can reconstruct submission
        df_save = X.copy()
        df_save["id_ref"] = df["id"].values
        df_save["before_ref"] = df["before"].values

        df_save.to_parquet(output_path, index=False)
        print(f"Saved processed test data to {output_path}")

        return X, df[["id", "before"]]
