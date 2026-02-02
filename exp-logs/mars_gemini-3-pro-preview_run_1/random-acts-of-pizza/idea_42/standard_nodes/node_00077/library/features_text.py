import os
import ast
import numpy as np
import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from library import config, utils

# Ensure VADER lexicon is available
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon", quiet=True)

from nltk.sentiment.vader import SentimentIntensityAnalyzer


class SBERTEmbedder:
    def __init__(self, model_name=config.SBERT_MODEL_NAME):
        self.model_name = model_name
        self.model = None

    def _load_model(self):
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)
            # SentenceTransformer automatically uses GPU if available

    def transform(self, texts):
        """
        Encodes a list of texts into dense embeddings.
        """
        self._load_model()
        # Handle NaNs and ensure strings
        cleaned_texts = [str(t) if pd.notnull(t) else "" for t in texts]
        embeddings = self.model.encode(
            cleaned_texts, show_progress_bar=False, convert_to_numpy=True
        )
        return embeddings

    def compute_history_centroids(self, subreddit_lists):
        """
        Computes the centroid (mean) embedding of a list of subreddits for each user.
        Optimized to embed unique subreddits only once.
        """
        self._load_model()

        # 1. Flatten and find unique subreddits
        all_subs = set()
        for sub_list in subreddit_lists:
            if isinstance(sub_list, list):
                all_subs.update(sub_list)
            elif isinstance(sub_list, str):
                # Attempt to parse if it's a stringified list
                try:
                    parsed = ast.literal_eval(sub_list)
                    if isinstance(parsed, list):
                        all_subs.update(parsed)
                except:
                    pass

        unique_subs = sorted(list(all_subs))
        if not unique_subs:
            return np.zeros(
                (len(subreddit_lists), config.EMBEDDING_DIM), dtype=np.float32
            )

        # 2. Embed unique subreddits
        sub_embeddings = self.model.encode(
            unique_subs, show_progress_bar=False, convert_to_numpy=True
        )
        sub_map = {sub: emb for sub, emb in zip(unique_subs, sub_embeddings)}

        # 3. Compute centroids
        centroids = []
        for sub_list in subreddit_lists:
            # Ensure list format
            if isinstance(sub_list, str):
                try:
                    sub_list = ast.literal_eval(sub_list)
                except:
                    sub_list = []
            if not isinstance(sub_list, list):
                sub_list = []

            valid_embs = [sub_map[s] for s in sub_list if s in sub_map]

            if valid_embs:
                centroid = np.mean(valid_embs, axis=0)
            else:
                centroid = np.zeros(config.EMBEDDING_DIM, dtype=np.float32)
            centroids.append(centroid)

        return np.array(centroids, dtype=np.float32)


class TFIDFGenerator:
    def __init__(
        self, vocab_size=config.TFIDF_VOCAB_SIZE, ngram_range=config.TFIDF_NGRAM_RANGE
    ):
        self.vocab_size = vocab_size
        self.ngram_range = ngram_range
        self.vectorizer = None

    def fit(self, texts):
        cleaned_texts = [str(t) if pd.notnull(t) else "" for t in texts]
        self.vectorizer = TfidfVectorizer(
            max_features=self.vocab_size,
            ngram_range=self.ngram_range,
            stop_words="english",
            dtype=np.float32,
        )
        self.vectorizer.fit(cleaned_texts)
        return self

    def transform(self, texts):
        if self.vectorizer is None:
            raise ValueError("TFIDFGenerator must be fitted before transform.")
        cleaned_texts = [str(t) if pd.notnull(t) else "" for t in texts]
        # Return dense array for consistency with other features
        return self.vectorizer.transform(cleaned_texts).toarray()


class SentimentAnalyzer:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()

    def transform(self, texts):
        """
        Returns a numpy array of shape (N, 4) containing [neg, neu, pos, compound] scores.
        """
        scores_list = []
        for t in texts:
            text_str = str(t) if pd.notnull(t) else ""
            scores = self.sia.polarity_scores(text_str)
            scores_list.append(
                [scores["neg"], scores["neu"], scores["pos"], scores["compound"]]
            )
        return np.array(scores_list, dtype=np.float32)


def generate_text_features(train_df, val_df, test_df, load_cached_data=True):
    """
    Generates or loads text-based features for Train, Val, and Test sets.
    Features include: SBERT embeddings (Title, Body, History Centroid), TF-IDF (Title+Body), and Sentiment.

    Args:
        train_df (pd.DataFrame): Training data.
        val_df (pd.DataFrame): Validation data (can be None).
        test_df (pd.DataFrame): Test data.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        dict: Dictionary containing numpy arrays for all features. Keys follow pattern '{split}_{feature}'.
    """
    cache_file = os.path.join(config.CACHE_DIR, "text_features.npz")

    # 1. Try Loading Cache
    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading text features from cache: {cache_file}")
        try:
            loaded = np.load(cache_file)
            # Convert NpzFile to dict to ensure it's accessible after closing
            return {k: v for k, v in loaded.items()}
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing features.")

    print("Computing text features from scratch...")

    # Initialize Extractors
    sbert = SBERTEmbedder()
    tfidf = TFIDFGenerator()
    sentiment = SentimentAnalyzer()

    # Helper to combine title and body
    def get_full_text(df):
        return (
            df["request_title"].fillna("")
            + " "
            + df["request_text_edit_aware"].fillna("")
        ).tolist()

    # --- 1. SBERT Embeddings ---
    print("Generating SBERT Embeddings (Title)...")
    train_title_emb = sbert.transform(train_df["request_title"])
    test_title_emb = sbert.transform(test_df["request_title"])
    val_title_emb = (
        sbert.transform(val_df["request_title"]) if val_df is not None else None
    )

    print("Generating SBERT Embeddings (Body)...")
    train_body_emb = sbert.transform(train_df["request_text_edit_aware"])
    test_body_emb = sbert.transform(test_df["request_text_edit_aware"])
    val_body_emb = (
        sbert.transform(val_df["request_text_edit_aware"])
        if val_df is not None
        else None
    )

    print("Generating SBERT Embeddings (History Centroids)...")
    # We combine all subreddit lists to ensure global embedding consistency if needed,
    # but compute_history_centroids handles lists independently.
    train_hist_emb = sbert.compute_history_centroids(
        train_df["requester_subreddits_at_request"]
    )
    test_hist_emb = sbert.compute_history_centroids(
        test_df["requester_subreddits_at_request"]
    )
    val_hist_emb = (
        sbert.compute_history_centroids(val_df["requester_subreddits_at_request"])
        if val_df is not None
        else None
    )

    # --- 2. TF-IDF ---
    print("Generating TF-IDF Features...")
    train_full = get_full_text(train_df)
    tfidf.fit(train_full)

    train_tfidf = tfidf.transform(train_full)
    test_tfidf = tfidf.transform(get_full_text(test_df))
    val_tfidf = tfidf.transform(get_full_text(val_df)) if val_df is not None else None

    # --- 3. Sentiment ---
    print("Generating Sentiment Features...")
    # Title Sentiment
    train_title_sent = sentiment.transform(train_df["request_title"])
    test_title_sent = sentiment.transform(test_df["request_title"])
    val_title_sent = (
        sentiment.transform(val_df["request_title"]) if val_df is not None else None
    )

    # Body Sentiment
    train_body_sent = sentiment.transform(train_df["request_text_edit_aware"])
    test_body_sent = sentiment.transform(test_df["request_text_edit_aware"])
    val_body_sent = (
        sentiment.transform(val_df["request_text_edit_aware"])
        if val_df is not None
        else None
    )

    # --- Pack Results ---
    results = {
        "train_title_emb": train_title_emb,
        "train_body_emb": train_body_emb,
        "train_hist_centroid": train_hist_emb,
        "train_tfidf": train_tfidf,
        "train_title_sentiment": train_title_sent,
        "train_body_sentiment": train_body_sent,
        "test_title_emb": test_title_emb,
        "test_body_emb": test_body_emb,
        "test_hist_centroid": test_hist_emb,
        "test_tfidf": test_tfidf,
        "test_title_sentiment": test_title_sent,
        "test_body_sentiment": test_body_sent,
    }

    if val_df is not None:
        results.update(
            {
                "val_title_emb": val_title_emb,
                "val_body_emb": val_body_emb,
                "val_hist_centroid": val_hist_emb,
                "val_tfidf": val_tfidf,
                "val_title_sentiment": val_title_sent,
                "val_body_sentiment": val_body_sent,
            }
        )

    # --- Save to Cache ---
    print(f"Saving text features to {cache_file}...")
    np.savez(cache_file, **results)

    return results
