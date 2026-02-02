import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer
from library.utils import extract_text_embeddings
from library.data_loader import load_dataset


class TextEmbedder:
    def __init__(
        self, model_name="sentence-transformers/all-MiniLM-L6-v2", batch_size=32
    ):
        self.model_name = model_name
        self.batch_size = batch_size

    def _get_texts(self, df):
        # Prefer edit aware text, fallback to standard request text
        if "request_text_edit_aware" in df.columns:
            return df["request_text_edit_aware"].fillna("").astype(str).tolist()
        return df["request_text"].fillna("").astype(str).tolist()

    def transform(self, df):
        texts = self._get_texts(df)
        return extract_text_embeddings(
            texts, model_name=self.model_name, batch_size=self.batch_size
        )


class MetadataExtractor:
    def __init__(self, numeric_cols=None):
        if numeric_cols is None:
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
        else:
            self.numeric_cols = numeric_cols

        # RankGauss scaler as per requirements
        self.scaler = QuantileTransformer(output_distribution="normal", random_state=42)

    def _extract_raw(self, df):
        # Select columns and fill NaNs with 0
        return df[self.numeric_cols].fillna(0).values

    def fit_transform(self, df):
        X = self._extract_raw(df)
        return self.scaler.fit_transform(X)

    def transform(self, df):
        X = self._extract_raw(df)
        return self.scaler.transform(X)


def run_feature_extraction(load_cached_data=True, debug_sample_size=None):
    """
    Orchestrates the feature extraction process with caching.

    Returns:
        Dictionary containing:
        - X_text_train, X_text_val, X_text_test
        - X_meta_train, X_meta_val, X_meta_test
        - y_train, y_val
        - test_ids
    """
    cache_dir = "./working/idea_9/"
    os.makedirs(cache_dir, exist_ok=True)

    # Define cache file paths
    cache_files = {
        "X_text_train": os.path.join(cache_dir, "X_text_train.npy"),
        "X_text_val": os.path.join(cache_dir, "X_text_val.npy"),
        "X_text_test": os.path.join(cache_dir, "X_text_test.npy"),
        "X_meta_train": os.path.join(cache_dir, "X_meta_train.npy"),
        "X_meta_val": os.path.join(cache_dir, "X_meta_val.npy"),
        "X_meta_test": os.path.join(cache_dir, "X_meta_test.npy"),
        "y_train": os.path.join(cache_dir, "y_train.npy"),
        "y_val": os.path.join(cache_dir, "y_val.npy"),
        "test_ids": os.path.join(cache_dir, "test_ids.npy"),
    }

    # 1. Try Loading from Cache
    if load_cached_data:
        all_exist = all(os.path.exists(p) for p in cache_files.values())
        if all_exist:
            print("Loading features from cache...")
            data = {}
            for key, path in cache_files.items():
                data[key] = np.load(path, allow_pickle=(key == "test_ids"))
            return data
        else:
            print("Cache incomplete or missing. Recomputing features...")
    else:
        print("Force recomputing features...")

    # 2. Load Raw Data
    df_train, df_val, df_test = load_dataset(
        load_cached_data=load_cached_data, debug_sample_size=debug_sample_size
    )

    # Extract Labels and IDs
    y_train = df_train["requester_received_pizza"].astype(int).values
    y_val = df_val["requester_received_pizza"].astype(int).values
    test_ids = df_test["request_id"].values

    # 3. Extract Text Embeddings
    print("Extracting text embeddings...")
    text_embedder = TextEmbedder()
    X_text_train = text_embedder.transform(df_train)
    X_text_val = text_embedder.transform(df_val)
    X_text_test = text_embedder.transform(df_test)

    # 4. Extract and Scale Metadata
    print("Extracting and scaling metadata...")
    meta_extractor = MetadataExtractor()
    X_meta_train = meta_extractor.fit_transform(df_train)
    X_meta_val = meta_extractor.transform(df_val)
    X_meta_test = meta_extractor.transform(df_test)

    # 5. Save to Cache
    print("Saving features to cache...")
    np.save(cache_files["X_text_train"], X_text_train)
    np.save(cache_files["X_text_val"], X_text_val)
    np.save(cache_files["X_text_test"], X_text_test)
    np.save(cache_files["X_meta_train"], X_meta_train)
    np.save(cache_files["X_meta_val"], X_meta_val)
    np.save(cache_files["X_meta_test"], X_meta_test)
    np.save(cache_files["y_train"], y_train)
    np.save(cache_files["y_val"], y_val)
    np.save(cache_files["test_ids"], test_ids)

    return {
        "X_text_train": X_text_train,
        "X_text_val": X_text_val,
        "X_text_test": X_text_test,
        "X_meta_train": X_meta_train,
        "X_meta_val": X_meta_val,
        "X_meta_test": X_meta_test,
        "y_train": y_train,
        "y_val": y_val,
        "test_ids": test_ids,
    }
