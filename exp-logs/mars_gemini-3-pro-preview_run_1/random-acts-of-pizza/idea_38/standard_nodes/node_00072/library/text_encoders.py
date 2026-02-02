import os
import numpy as np
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import scipy.sparse
from library.config import Config

# Ensure VADER lexicon is downloaded
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon", quiet=True)


class SBERTEmbedder:
    """
    Handles generation of dense vector embeddings using Sentence-BERT.
    Supports caching of embeddings to disk.
    """

    def __init__(
        self,
        model_name=Config.SBERT_MODEL_NAME,
        device=Config.MLP_PARAMS["device"],
        cache_dir=Config.WORKING_DIR,
    ):
        self.model_name = model_name
        self.device = device if isinstance(device, str) else "cpu"
        self.cache_dir = cache_dir
        self.model = None

    def _load_model(self):
        if self.model is None:
            # Suppress loading info if possible
            self.model = SentenceTransformer(self.model_name, device=self.device)

    def encode_text(self, df, col_name, split_name, load_cached_data=True):
        """
        Encodes a single text column into embeddings.

        Args:
            df (pd.DataFrame): Dataframe containing the column.
            col_name (str): Name of the text column.
            split_name (str): 'train', 'val', or 'test' for cache naming.
            load_cached_data (bool): Whether to use cache.

        Returns:
            np.ndarray: Embeddings of shape (N, D).
        """
        os.makedirs(self.cache_dir, exist_ok=True)
        cache_path = os.path.join(self.cache_dir, f"sbert_{col_name}_{split_name}.npy")

        if load_cached_data and os.path.exists(cache_path):
            return np.load(cache_path)

        self._load_model()

        # Handle NaNs
        texts = df[col_name].fillna("").astype(str).tolist()

        # Encode
        embeddings = self.model.encode(
            texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True
        )

        # Save
        np.save(cache_path, embeddings)

        return embeddings

    def encode_history(
        self, df, col_name, split_name, load_cached_data=True, max_len=None
    ):
        """
        Encodes a column containing lists of strings (e.g. subreddits) into a sequence of embeddings.
        Optimized by encoding unique strings only.

        Args:
            df (pd.DataFrame): Dataframe containing the list column.
            col_name (str): Name of the column (list of strings).
            split_name (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to use cache.
            max_len (int): Optional max sequence length. If None, uses max in data.

        Returns:
            tuple: (embeddings_tensor, mask)
                embeddings_tensor: (N, T, D)
                mask: (N, T) boolean mask (True where data exists, False for padding)
        """
        os.makedirs(self.cache_dir, exist_ok=True)
        emb_cache_path = os.path.join(
            self.cache_dir, f"sbert_hist_emb_{col_name}_{split_name}.npy"
        )
        mask_cache_path = os.path.join(
            self.cache_dir, f"sbert_hist_mask_{col_name}_{split_name}.npy"
        )

        if (
            load_cached_data
            and os.path.exists(emb_cache_path)
            and os.path.exists(mask_cache_path)
        ):
            return np.load(emb_cache_path), np.load(mask_cache_path)

        self._load_model()

        # Extract all lists
        # Assuming data is loaded correctly as lists.
        raw_sequences = df[col_name].tolist()

        # Flatten to find unique subreddits
        unique_items = set()
        for seq in raw_sequences:
            if isinstance(seq, (list, np.ndarray)):
                unique_items.update(seq)

        unique_items_list = sorted(list(unique_items))

        # Handle case with no items
        if not unique_items_list:
            embedding_dim = self.model.get_sentence_embedding_dimension()
            return np.zeros((len(df), 1, embedding_dim), dtype=np.float32), np.zeros(
                (len(df), 1), dtype=bool
            )

        # Encode unique items
        item_embeddings = self.model.encode(
            unique_items_list,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        item_to_idx = {item: i for i, item in enumerate(unique_items_list)}

        # Determine dimensions
        if max_len is None:
            max_len = max(
                (
                    len(seq)
                    for seq in raw_sequences
                    if isinstance(seq, (list, np.ndarray))
                ),
                default=0,
            )
            max_len = max(1, max_len)  # Ensure at least 1

        embedding_dim = item_embeddings.shape[1]
        N = len(df)

        # Construct tensor
        output_tensor = np.zeros((N, max_len, embedding_dim), dtype=np.float32)
        output_mask = np.zeros((N, max_len), dtype=bool)

        for i, seq in enumerate(raw_sequences):
            if isinstance(seq, (list, np.ndarray)):
                # Truncate if necessary
                seq = seq[:max_len]
                for t, item in enumerate(seq):
                    if item in item_to_idx:
                        idx = item_to_idx[item]
                        output_tensor[i, t, :] = item_embeddings[idx]
                        output_mask[i, t] = True

        # Save
        np.save(emb_cache_path, output_tensor)
        np.save(mask_cache_path, output_mask)

        return output_tensor, output_mask


class TFIDFHandler:
    """
    Handles TF-IDF vectorization of text data.
    """

    def __init__(
        self, vocab_size=Config.TFIDF_VOCAB_SIZE, cache_dir=Config.WORKING_DIR
    ):
        self.vocab_size = vocab_size
        self.cache_dir = cache_dir
        self.vectorizer = None

    def process(self, train_df, val_df, test_df, load_cached_data=True):
        """
        Fits on train, transforms train/val/test.
        Combines title and body text.

        Returns:
            tuple: (train_tfidf, val_tfidf, test_tfidf) sparse matrices.
        """
        os.makedirs(self.cache_dir, exist_ok=True)
        train_path = os.path.join(self.cache_dir, "tfidf_train.npz")
        val_path = os.path.join(self.cache_dir, "tfidf_val.npz")
        test_path = os.path.join(self.cache_dir, "tfidf_test.npz")

        if (
            load_cached_data
            and os.path.exists(train_path)
            and os.path.exists(val_path)
            and os.path.exists(test_path)
        ):
            return (
                scipy.sparse.load_npz(train_path),
                scipy.sparse.load_npz(val_path),
                scipy.sparse.load_npz(test_path),
            )

        # Prepare text
        def get_text(df):
            t = df[Config.TEXT_COL_TITLE].fillna("").astype(str)
            b = df[Config.TEXT_COL_BODY].fillna("").astype(str)
            return t + " " + b

        train_text = get_text(train_df)
        val_text = get_text(val_df)
        test_text = get_text(test_df)

        # Fit and Transform
        self.vectorizer = TfidfVectorizer(
            max_features=self.vocab_size, stop_words="english", sublinear_tf=True
        )
        train_tfidf = self.vectorizer.fit_transform(train_text)
        val_tfidf = self.vectorizer.transform(val_text)
        test_tfidf = self.vectorizer.transform(test_text)

        # Save
        scipy.sparse.save_npz(train_path, train_tfidf)
        scipy.sparse.save_npz(val_path, val_tfidf)
        scipy.sparse.save_npz(test_path, test_tfidf)

        return train_tfidf, val_tfidf, test_tfidf


class SentimentAnalyzer:
    """
    Extracts VADER sentiment scores.
    """

    def __init__(self, cache_dir=Config.WORKING_DIR):
        self.cache_dir = cache_dir
        self.analyzer = SentimentIntensityAnalyzer()

    def process(self, df, split_name, load_cached_data=True):
        """
        Calculates sentiment scores for title and body.

        Returns:
            np.ndarray: Array of shape (N, 8) [Title(neg,neu,pos,compound), Body(neg,neu,pos,compound)]
        """
        os.makedirs(self.cache_dir, exist_ok=True)
        cache_path = os.path.join(self.cache_dir, f"sentiment_{split_name}.npy")

        if load_cached_data and os.path.exists(cache_path):
            return np.load(cache_path)

        titles = df[Config.TEXT_COL_TITLE].fillna("").astype(str).tolist()
        bodies = df[Config.TEXT_COL_BODY].fillna("").astype(str).tolist()

        features = []
        for t, b in zip(titles, bodies):
            s_t = self.analyzer.polarity_scores(t)
            s_b = self.analyzer.polarity_scores(b)
            features.append(
                [
                    s_t["neg"],
                    s_t["neu"],
                    s_t["pos"],
                    s_t["compound"],
                    s_b["neg"],
                    s_b["neu"],
                    s_b["pos"],
                    s_b["compound"],
                ]
            )

        features_arr = np.array(features, dtype=np.float32)
        np.save(cache_path, features_arr)

        return features_arr
