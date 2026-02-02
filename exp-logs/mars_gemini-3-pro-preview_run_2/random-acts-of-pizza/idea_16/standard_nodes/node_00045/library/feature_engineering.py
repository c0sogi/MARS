import os
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import QuantileTransformer, normalize
from sklearn.impute import SimpleImputer
from library.config import Config
from library.utils import save_numpy, load_numpy, ensure_directory, set_seed


class TextEmbedder:
    """
    Handles generation and caching of text embeddings using SentenceTransformers.
    Applies L2 normalization to the embeddings to project them onto the hypersphere.
    """

    def __init__(self, model_name=Config.PRETRAINED_MODEL_NAME):
        self.model_name = model_name
        self.model = None

    def _load_model(self):
        if self.model is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = SentenceTransformer(self.model_name, device=device)

    def encode(self, texts):
        """
        Encodes a list of texts into embeddings and applies L2 normalization.
        """
        self._load_model()
        # Encode with sentence-transformers
        # normalize_embeddings=False because we apply explicit L2 normalization later
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )

        # Apply L2 Normalization
        embeddings = normalize(embeddings, norm="l2", axis=1)
        return embeddings

    def process_and_cache(self, df, cache_path, load_from_cache=True):
        """
        Generates embeddings for the dataframe or loads from cache.
        Checks if cached data matches the dataframe length to ensure consistency.
        """
        # Prepare text data: Concatenate title and edit-aware text
        title = df["request_title"].fillna("").astype(str)
        text = df["request_text_edit_aware"].fillna("").astype(str)
        full_text = (title + " " + text).tolist()
        expected_len = len(full_text)

        # Try loading from cache
        if load_from_cache and os.path.exists(cache_path):
            try:
                embeddings = load_numpy(cache_path)
                if len(embeddings) == expected_len:
                    return embeddings
                # If length mismatch, fall through to recompute
            except Exception:
                # If load fails, fall through to recompute
                pass

        # Compute embeddings
        embeddings = self.encode(full_text)

        # Save to cache
        save_numpy(embeddings, cache_path)

        return embeddings


class TabularPreprocessor:
    """
    Handles preprocessing of tabular features:
    - Selection of specific numerical features based on analysis
    - Imputation (Median)
    - Transformation (QuantileTransformer / RankGauss)
    """

    def __init__(self):
        # Features selected based on analysis and idea description
        self.numeric_features = [
            "requester_account_age_in_days_at_request",
            "requester_days_since_first_post_on_raop_at_request",
            "requester_number_of_comments_at_request",
            "requester_number_of_comments_in_raop_at_request",
            "requester_number_of_posts_at_request",
            "requester_number_of_posts_on_raop_at_request",
            "requester_number_of_subreddits_at_request",
            "requester_upvotes_minus_downvotes_at_request",
            "requester_upvotes_plus_downvotes_at_request",
            "unix_timestamp_of_request",
        ]

        self.imputer = SimpleImputer(strategy="median")
        # RankGauss transformation to normalize distributions
        self.scaler = QuantileTransformer(
            output_distribution="normal", random_state=Config.SEED
        )

    def fit_transform(self, df):
        # Extract features
        X = df[self.numeric_features].copy()

        # Impute
        X = self.imputer.fit_transform(X)

        # Transform
        X = self.scaler.fit_transform(X)

        return X.astype(np.float32)

    def transform(self, df):
        # Extract features
        X = df[self.numeric_features].copy()

        # Impute
        X = self.imputer.transform(X)

        # Transform
        X = self.scaler.transform(X)

        return X.astype(np.float32)


def get_processed_data(df_train, df_test, load_from_cache=True):
    """
    Main feature engineering function.

    Args:
        df_train (pd.DataFrame): Training data (including validation).
        df_test (pd.DataFrame): Test data.
        load_from_cache (bool): Whether to use cached embeddings.

    Returns:
        X_train (np.ndarray): Processed training features (Text + Tabular).
        y_train (np.ndarray): Training labels.
        X_test (np.ndarray): Processed test features (Text + Tabular).
    """
    set_seed(Config.SEED)

    # --- 1. Text Embeddings ---
    embedder = TextEmbedder()

    # Process Train (includes Val)
    train_emb = embedder.process_and_cache(
        df_train, Config.TRAIN_EMBEDDINGS_PATH, load_from_cache
    )

    # Process Test
    test_emb = embedder.process_and_cache(
        df_test, Config.TEST_EMBEDDINGS_PATH, load_from_cache
    )

    # --- 2. Tabular Features ---
    tab_preprocessor = TabularPreprocessor()

    # Fit on train, transform train and test
    train_tab = tab_preprocessor.fit_transform(df_train)
    test_tab = tab_preprocessor.transform(df_test)

    # --- 3. Feature Fusion ---
    # Concatenate [Embeddings, Tabular]
    X_train = np.hstack([train_emb, train_tab])
    X_test = np.hstack([test_emb, test_tab])

    # --- 4. Targets ---
    y_train = df_train["requester_received_pizza"].values.astype(int)

    return X_train, y_train, X_test
