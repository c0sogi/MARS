import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer, normalize
from sentence_transformers import SentenceTransformer
from library.config import (
    TRANSFORMER_MODEL,
    NORMALIZE_EMBEDDINGS,
    TABULAR_SCALER,
    SEED,
    WORKING_DIR,
)
from library.data_loader import load_dataset_with_metadata

# Define numerical features to use (excluding retrieval-time features and leakage)
NUMERIC_FEATURES = [
    "requester_account_age_in_days_at_request",
    "requester_days_since_first_post_on_raop_at_request",
    "requester_number_of_comments_at_request",
    "requester_number_of_comments_in_raop_at_request",
    "requester_number_of_posts_at_request",
    "requester_number_of_posts_on_raop_at_request",
    "requester_number_of_subreddits_at_request",
    "requester_upvotes_minus_downvotes_at_request",
    "requester_upvotes_plus_downvotes_at_request",
    "unix_timestamp_of_request_utc",
]


class FeatureEngineer:
    def __init__(self):
        self.model_name = TRANSFORMER_MODEL
        self.normalize_embeddings = NORMALIZE_EMBEDDINGS
        self.numeric_features = NUMERIC_FEATURES
        self.seed = SEED
        self.working_dir = WORKING_DIR

        # Ensure working directory exists
        os.makedirs(self.working_dir, exist_ok=True)

    def _get_cache_paths(self, split):
        x_path = os.path.join(self.working_dir, f"X_{split}.npy")
        y_path = os.path.join(self.working_dir, f"y_{split}.npy")
        return x_path, y_path

    def _preprocess_text(self, df):
        """Combines title and text content."""
        # Use edit_aware text if available, else standard text
        text_col = (
            "request_text_edit_aware"
            if "request_text_edit_aware" in df.columns
            else "request_text"
        )

        # Fill NaNs
        titles = df["request_title"].fillna("").astype(str)
        texts = df[text_col].fillna("").astype(str)

        # Combine: Title + " " + Text
        combined = titles + " " + texts
        return combined.tolist()

    def _encode_text(self, text_list):
        """Encodes text using SentenceTransformer and optionally normalizes."""
        print(f"Encoding {len(text_list)} text samples with {self.model_name}...")
        model = SentenceTransformer(self.model_name)
        # Encode
        embeddings = model.encode(
            text_list, batch_size=32, show_progress_bar=False, convert_to_numpy=True
        )

        if self.normalize_embeddings:
            print("Normalizing embeddings (L2)...")
            embeddings = normalize(embeddings, norm="l2")

        return embeddings

    def _process_numerics(self, df, is_train=False, train_df=None):
        """
        Processes numerical features using RankGauss (QuantileTransformer).
        If is_train=True, fits the transformer on df.
        If is_train=False, fits the transformer on train_df, then transforms df.
        """
        # Extract features
        X_num = df[self.numeric_features].copy()

        # Simple imputation for safety (fill NaN with 0)
        X_num = X_num.fillna(0).values

        scaler = QuantileTransformer(
            output_distribution="normal", random_state=self.seed
        )

        if is_train:
            print("Fitting QuantileTransformer on training data...")
            X_transformed = scaler.fit_transform(X_num)
        else:
            if train_df is None:
                raise ValueError(
                    "train_df must be provided when processing validation/test data to fit the scaler."
                )

            print(
                "Fitting QuantileTransformer on provided training data (for consistency)..."
            )
            X_train_num = train_df[self.numeric_features].fillna(0).values
            scaler.fit(X_train_num)

            print("Transforming data...")
            X_transformed = scaler.transform(X_num)

        return X_transformed

    def create_features(self, split, load_cached_data=True):
        """
        Main pipeline to create features for a specific split.
        Handles caching, text encoding, and numerical scaling.
        """
        x_cache, y_cache = self._get_cache_paths(split)

        # 1. Check Cache
        if load_cached_data and os.path.exists(x_cache):
            # Check if y cache exists (only for train/val)
            y_exists = os.path.exists(y_cache)
            if split == "test" or y_exists:
                print(f"Loading {split} features from cache...")
                X = np.load(x_cache)
                y = np.load(y_cache) if y_exists else None
                return X, y

        print(f"Generating features for {split} from scratch...")

        # 2. Load Data
        df = load_dataset_with_metadata(split, load_cached_data=load_cached_data)

        # 3. Text Features
        text_list = self._preprocess_text(df)
        X_text = self._encode_text(text_list)

        # 4. Numerical Features
        if split == "train":
            X_num = self._process_numerics(df, is_train=True)
        else:
            # We need raw train data to fit the scaler
            print("Loading train data to fit numerical scaler...")
            df_train = load_dataset_with_metadata(
                "train", load_cached_data=load_cached_data
            )
            X_num = self._process_numerics(df, is_train=False, train_df=df_train)

        # 5. Concatenate
        print("Concatenating text and numerical features...")
        X = np.hstack([X_text, X_num])

        # 6. Extract Labels
        if "requester_received_pizza" in df.columns:
            y = df["requester_received_pizza"].values.astype(int)
        else:
            y = None

        # 7. Save to Cache
        print(f"Saving {split} features to cache...")
        np.save(x_cache, X)
        if y is not None:
            np.save(y_cache, y)

        return X, y


def get_features(split, load_cached_data=True):
    """Wrapper function to be called by other modules."""
    engineer = FeatureEngineer()
    return engineer.create_features(split, load_cached_data=load_cached_data)
