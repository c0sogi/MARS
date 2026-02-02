import os
import ast
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sentence_transformers import SentenceTransformer

from library.config import (
    TRAIN_PATH,
    VAL_PATH,
    TEST_PATH,
    CACHE_DIR,
    SBERT_MODEL_NAME,
    MAX_HISTORY_LEN,
    TFIDF_PARAMS,
    RANDOM_SEED,
)
from library.utils import get_common_features, set_seed

# Ensure VADER lexicon is available
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon", quiet=True)

# Set seed for reproducibility
set_seed(RANDOM_SEED)


class PizzaDataset(Dataset):
    """
    PyTorch Dataset for the Alignment-Injected Dual-Query MLP.
    """

    def __init__(self, title_emb, body_emb, history_emb, metadata, labels=None):
        self.title_emb = torch.tensor(title_emb, dtype=torch.float32)
        self.body_emb = torch.tensor(body_emb, dtype=torch.float32)
        self.history_emb = torch.tensor(history_emb, dtype=torch.float32)
        self.metadata = torch.tensor(metadata, dtype=torch.float32)

        if labels is not None:
            self.labels = torch.tensor(labels, dtype=torch.float32)
        else:
            self.labels = None

    def __len__(self):
        return len(self.title_emb)

    def __getitem__(self, idx):
        item = {
            "title_emb": self.title_emb[idx],
            "body_emb": self.body_emb[idx],
            "history_emb": self.history_emb[idx],
            "metadata": self.metadata[idx],
        }
        if self.labels is not None:
            item["label"] = self.labels[idx]
        return item


class FeatureEngineer:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.numeric_cols = None

    def extract_sentiment(self, df):
        """
        Extracts VADER sentiment scores for Title and Body.
        """
        print("Extracting sentiment features...")
        titles = df["request_title"].fillna("").astype(str).tolist()
        bodies = df["request_text_edit_aware"].fillna("").astype(str).tolist()

        features = []
        for t, b in zip(titles, bodies):
            t_scores = self.sia.polarity_scores(t)
            b_scores = self.sia.polarity_scores(b)
            features.append(
                {
                    "title_compound": t_scores["compound"],
                    "title_pos": t_scores["pos"],
                    "title_neg": t_scores["neg"],
                    "title_neu": t_scores["neu"],
                    "body_compound": b_scores["compound"],
                    "body_pos": b_scores["pos"],
                    "body_neg": b_scores["neg"],
                    "body_neu": b_scores["neu"],
                }
            )

        return pd.DataFrame(features)

    def fit_metadata(self, train_df, val_df, test_df):
        """
        Identifies common numeric columns and fits imputer/scaler.
        """
        # Exclude text, ID, and leakage columns
        exclude = [
            "request_id",
            "request_text",
            "request_title",
            "request_text_edit_aware",
            "requester_subreddits_at_request",
            "requester_received_pizza",
            "giver_username_if_known",
            "source_file",
            "post_was_edited",
            "requester_username",
            "requester_user_flair",
        ]

        # Get intersection of columns
        common_cols = get_common_features(train_df, test_df, exclude_cols=exclude)

        # Filter for numeric types only
        self.numeric_cols = [
            c for c in common_cols if pd.api.types.is_numeric_dtype(train_df[c])
        ]

        print(f"Selected {len(self.numeric_cols)} numeric metadata columns.")

        # Fit imputer on training data
        self.imputer.fit(train_df[self.numeric_cols])

        # For NN, we also fit a scaler (after log/arcsinh transform)
        # We'll do the transform first then fit scaler in process_metadata_nn

        return self.numeric_cols

    def process_metadata_rf(self, df):
        """
        Prepares metadata for Random Forest (Imputation + Ratios).
        """
        data = df[self.numeric_cols].copy()

        # Impute
        data_imputed = pd.DataFrame(
            self.imputer.transform(data), columns=self.numeric_cols, index=df.index
        )

        # Feature Engineering: Ratios
        # Avoid division by zero
        eps = 1e-6
        data_imputed["upvote_ratio"] = data_imputed[
            "number_of_upvotes_of_request_at_retrieval"
        ] / (
            data_imputed["number_of_upvotes_of_request_at_retrieval"]
            + data_imputed["number_of_downvotes_of_request_at_retrieval"]
            + eps
        )

        return data_imputed

    def process_metadata_nn(self, df, is_train=False):
        """
        Prepares metadata for Neural Network (Arcsinh + Scaling).
        """
        data = df[self.numeric_cols].copy()

        # Impute
        data_vals = self.imputer.transform(data)

        # Apply Arcsinh (handles zeros and negative values better than log)
        data_trans = np.arcsinh(data_vals)

        # Scale
        if is_train:
            self.scaler.fit(data_trans)

        data_scaled = self.scaler.transform(data_trans)

        return data_scaled


class TextProcessor:
    def __init__(self):
        self.tfidf = TfidfVectorizer(**TFIDF_PARAMS)
        self.sbert = None  # Lazy load
        self.subreddit_embeddings = {}

    def fit_transform_tfidf(self, train_df, val_df, test_df):
        """
        Fits TF-IDF on Train and transforms all splits.
        """
        print("Fitting TF-IDF...")

        # Concatenate Title + Body
        def get_text(df):
            return (
                df["request_title"].fillna("")
                + " "
                + df["request_text_edit_aware"].fillna("")
            ).astype(str)

        train_text = get_text(train_df)
        val_text = get_text(val_df)
        test_text = get_text(test_df)

        # Fit on Train
        X_train = self.tfidf.fit_transform(train_text)
        X_val = self.tfidf.transform(val_text)
        X_test = self.tfidf.transform(test_text)

        # Convert to DataFrame to keep it clean for concatenation
        feature_names = [f"tfidf_{i}" for i in range(X_train.shape[1])]

        return (
            pd.DataFrame.sparse.from_spmatrix(
                X_train, columns=feature_names, index=train_df.index
            ),
            pd.DataFrame.sparse.from_spmatrix(
                X_val, columns=feature_names, index=val_df.index
            ),
            pd.DataFrame.sparse.from_spmatrix(
                X_test, columns=feature_names, index=test_df.index
            ),
        )

    def load_sbert(self):
        if self.sbert is None:
            print(f"Loading SBERT model: {SBERT_MODEL_NAME}")
            self.sbert = SentenceTransformer(SBERT_MODEL_NAME)

    def encode_text_sbert(self, df):
        self.load_sbert()
        titles = df["request_title"].fillna("").astype(str).tolist()
        bodies = df["request_text_edit_aware"].fillna("").astype(str).tolist()

        print("Encoding titles and bodies...")
        title_emb = self.sbert.encode(titles, batch_size=32, show_progress_bar=False)
        body_emb = self.sbert.encode(bodies, batch_size=32, show_progress_bar=False)

        return title_emb, body_emb

    def encode_history_sbert(self, train_df, val_df, test_df):
        """
        Encodes subreddit history sequences.
        Optimized by encoding unique subreddits first.
        """
        self.load_sbert()
        print("Encoding history...")

        # Gather all unique subreddits
        all_subs = set()
        for df in [train_df, val_df, test_df]:
            for sub_list in df["requester_subreddits_at_request"]:
                all_subs.update(sub_list)

        unique_subs = sorted(list(all_subs))
        print(f"Unique subreddits to encode: {len(unique_subs)}")

        # Encode unique subreddits
        if unique_subs:
            sub_embs = self.sbert.encode(
                unique_subs, batch_size=64, show_progress_bar=False
            )
            sub_map = {sub: emb for sub, emb in zip(unique_subs, sub_embs)}
        else:
            sub_map = {}

        # Helper to create padded tensor for a dataframe
        def create_history_tensor(df):
            N = len(df)
            dim = self.sbert.get_sentence_embedding_dimension()
            tensor = np.zeros((N, MAX_HISTORY_LEN, dim), dtype=np.float32)

            for i, sub_list in enumerate(df["requester_subreddits_at_request"]):
                # Truncate to max len
                subs = sub_list[:MAX_HISTORY_LEN]
                for j, sub in enumerate(subs):
                    if sub in sub_map:
                        tensor[i, j, :] = sub_map[sub]
            return tensor

        return (
            create_history_tensor(train_df),
            create_history_tensor(val_df),
            create_history_tensor(test_df),
        )


def load_raw_data():
    """
    Loads raw CSVs and parses list columns.
    """
    print("Loading raw metadata CSVs...")
    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)
    test_df = pd.read_csv(TEST_PATH)

    # Parse list columns
    for df in [train_df, val_df, test_df]:
        if "requester_subreddits_at_request" in df.columns:
            df["requester_subreddits_at_request"] = df[
                "requester_subreddits_at_request"
            ].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])

    return train_df, val_df, test_df


def get_rf_dataset(split="train", load_cached_data=True):
    """
    Returns (X, y) for Random Forest.
    Features: Metadata + Sentiment + TF-IDF.
    Strictly excludes History features (Orthogonal Scoping).
    """
    cache_file = os.path.join(CACHE_DIR, f"rf_data_{split}.parquet")

    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading cached RF data for {split}...")
        df = pd.read_parquet(cache_file)
        if split != "test":
            y = df["target"]
            X = df.drop(columns=["target"])
            return X, y
        else:
            return df, None

    # If not cached, compute everything
    train_df, val_df, test_df = load_raw_data()

    # Initialize Processors
    fe = FeatureEngineer()
    tp = TextProcessor()

    # 1. Fit Metadata and TF-IDF on Train (and transform all)
    fe.fit_metadata(train_df, val_df, test_df)
    tfidf_train, tfidf_val, tfidf_test = tp.fit_transform_tfidf(
        train_df, val_df, test_df
    )

    # 2. Process Sentiment
    sent_train = fe.extract_sentiment(train_df)
    sent_val = fe.extract_sentiment(val_df)
    sent_test = fe.extract_sentiment(test_df)

    # 3. Process Metadata
    meta_train = fe.process_metadata_rf(train_df)
    meta_val = fe.process_metadata_rf(val_df)
    meta_test = fe.process_metadata_rf(test_df)

    # 4. Combine
    def combine(meta, sent, tfidf, raw_df, has_target=True):
        # Reset indices to ensure alignment
        meta = meta.reset_index(drop=True)
        sent = sent.reset_index(drop=True)
        tfidf = tfidf.reset_index(drop=True)

        combined = pd.concat([meta, sent, tfidf], axis=1)

        if has_target:
            combined["target"] = raw_df["requester_received_pizza"].astype(int).values

        return combined

    train_final = combine(meta_train, sent_train, tfidf_train, train_df, True)
    val_final = combine(meta_val, sent_val, tfidf_val, val_df, True)
    test_final = combine(meta_test, sent_test, tfidf_test, test_df, False)

    # Save to cache
    print("Caching RF datasets...")
    train_final.to_parquet(os.path.join(CACHE_DIR, "rf_data_train.parquet"))
    val_final.to_parquet(os.path.join(CACHE_DIR, "rf_data_val.parquet"))
    test_final.to_parquet(os.path.join(CACHE_DIR, "rf_data_test.parquet"))

    # Return requested split
    if split == "train":
        return train_final.drop(columns=["target"]), train_final["target"]
    elif split == "val":
        return val_final.drop(columns=["target"]), val_final["target"]
    elif split == "test":
        return test_final, None
    else:
        raise ValueError(f"Unknown split: {split}")


def get_nn_dataset(split="train", load_cached_data=True):
    """
    Returns PizzaDataset for Neural Network.
    Features: Title Emb, Body Emb, History Emb, Metadata (Arcsinh).
    """
    cache_file = os.path.join(CACHE_DIR, f"nn_data_{split}.npz")

    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading cached NN data for {split}...")
        data = np.load(cache_file)
        labels = data["labels"] if "labels" in data else None
        return PizzaDataset(
            data["title_emb"],
            data["body_emb"],
            data["history_emb"],
            data["metadata"],
            labels,
        )

    # Compute from scratch
    train_df, val_df, test_df = load_raw_data()

    fe = FeatureEngineer()
    tp = TextProcessor()

    # 1. Metadata
    fe.fit_metadata(train_df, val_df, test_df)
    meta_train = fe.process_metadata_nn(train_df, is_train=True)
    meta_val = fe.process_metadata_nn(val_df, is_train=False)
    meta_test = fe.process_metadata_nn(test_df, is_train=False)

    # 2. Text Embeddings
    t_train, b_train = tp.encode_text_sbert(train_df)
    t_val, b_val = tp.encode_text_sbert(val_df)
    t_test, b_test = tp.encode_text_sbert(test_df)

    # 3. History Embeddings
    h_train, h_val, h_test = tp.encode_history_sbert(train_df, val_df, test_df)

    # 4. Labels
    y_train = train_df["requester_received_pizza"].astype(int).values
    y_val = val_df["requester_received_pizza"].astype(int).values

    # Save to cache
    print("Caching NN datasets...")
    np.savez(
        os.path.join(CACHE_DIR, "nn_data_train.npz"),
        title_emb=t_train,
        body_emb=b_train,
        history_emb=h_train,
        metadata=meta_train,
        labels=y_train,
    )
    np.savez(
        os.path.join(CACHE_DIR, "nn_data_val.npz"),
        title_emb=t_val,
        body_emb=b_val,
        history_emb=h_val,
        metadata=meta_val,
        labels=y_val,
    )
    np.savez(
        os.path.join(CACHE_DIR, "nn_data_test.npz"),
        title_emb=t_test,
        body_emb=b_test,
        history_emb=h_test,
        metadata=meta_test,
    )

    # Return requested split
    if split == "train":
        return PizzaDataset(t_train, b_train, h_train, meta_train, y_train)
    elif split == "val":
        return PizzaDataset(t_val, b_val, h_val, meta_val, y_val)
    elif split == "test":
        return PizzaDataset(t_test, b_test, h_test, meta_test, None)
    else:
        raise ValueError(f"Unknown split: {split}")
