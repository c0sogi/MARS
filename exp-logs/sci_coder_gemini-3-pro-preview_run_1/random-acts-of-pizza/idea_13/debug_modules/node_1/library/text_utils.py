import os
import ast
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from library.config import Config


class SBERTEmbedder:
    """
    Generates dense embeddings for requests and user history using SentenceTransformer.
    Implements caching to .npy files to speed up repeated runs.
    """

    def __init__(self, model_name=Config.SBERT_MODEL_NAME, device=Config.DEVICE):
        self.model_name = model_name
        self.device = device
        # Initialize model only when needed to save resources if loading from cache
        self.model = None

    def _load_model(self):
        if self.model is None:
            self.model = SentenceTransformer(self.model_name, device=self.device)

    def _get_cache_path(self, filename):
        # Ensure working directory exists
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        return os.path.join(Config.WORKING_DIR, filename)

    def encode_requests(self, df, split_name, load_cached_data=True):
        """
        Encodes the concatenated title and body of requests.

        Args:
            df (pd.DataFrame): Dataframe containing request text.
            split_name (str): Name of the split (train/val/test) for caching.
            load_cached_data (bool): Whether to load from cache.

        Returns:
            np.ndarray: Embeddings of shape (N, embedding_dim).
        """
        cache_file = f"sbert_requests_{split_name}.npy"
        cache_path = self._get_cache_path(cache_file)

        # 1. Try to load from cache
        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading cached request embeddings from {cache_path}")
            return np.load(cache_path)

        # 2. Compute from scratch
        print(f"Encoding requests for {split_name}...")
        self._load_model()

        # Prepare text: Title + " " + Body
        titles = df[Config.TEXT_COL_TITLE].fillna("").astype(str)
        bodies = df[Config.TEXT_COL_BODY].fillna("").astype(str)
        texts = (titles + " " + bodies).tolist()

        # Encode
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        # 3. Save to cache
        print(f"Saving request embeddings to {cache_path}")
        np.save(cache_path, embeddings)
        return embeddings

    def encode_history(self, df, split_name, load_cached_data=True):
        """
        Encodes the subreddit history for each user.
        Generates a 3D tensor for attention mechanisms.

        Args:
            df (pd.DataFrame): Dataframe containing subreddit history column.
            split_name (str): Name of the split.
            load_cached_data (bool): Whether to load from cache.

        Returns:
            np.ndarray: Tensor of shape (N, max_history_len, embedding_dim).
        """
        cache_file = f"sbert_history_{split_name}.npy"
        cache_path = self._get_cache_path(cache_file)

        # 1. Try to load from cache
        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading cached history embeddings from {cache_path}")
            return np.load(cache_path)

        # 2. Compute from scratch
        print(f"Encoding history for {split_name}...")
        self._load_model()

        # Parse history column (strings like "['sub1', 'sub2']" -> lists)
        history_col = df[Config.SUBREDDIT_LIST_COL]
        parsed_history = []
        for x in history_col:
            if isinstance(x, str):
                try:
                    parsed_history.append(ast.literal_eval(x))
                except (ValueError, SyntaxError):
                    parsed_history.append([])
            elif isinstance(x, list):
                parsed_history.append(x)
            elif isinstance(x, np.ndarray):
                parsed_history.append(x.tolist())
            else:
                parsed_history.append([])

        # Identify unique subreddits across this split to encode efficiently
        all_subs = [sub for user_list in parsed_history for sub in user_list]
        unique_subs = sorted(list(set(all_subs)))

        sub_map = {}
        embedding_dim = Config.EMBEDDING_DIM

        if unique_subs:
            # Encode unique subreddits
            sub_embeddings = self.model.encode(
                unique_subs,
                batch_size=64,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            sub_map = {sub: emb for sub, emb in zip(unique_subs, sub_embeddings)}
            embedding_dim = sub_embeddings.shape[1]

        # Construct the 3D tensor (N, Max_Len, Dim)
        N = len(df)
        max_len = Config.MAX_HISTORY_LENGTH

        history_tensor = np.zeros((N, max_len, embedding_dim), dtype=np.float32)

        for i, user_subs in enumerate(parsed_history):
            # Take up to max_len subreddits
            # We take the first K provided in the list
            subs_to_process = user_subs[:max_len]

            for j, sub in enumerate(subs_to_process):
                if sub in sub_map:
                    history_tensor[i, j, :] = sub_map[sub]
                # Remaining slots stay 0 (padding)

        # 3. Save to cache
        print(f"Saving history embeddings to {cache_path}")
        np.save(cache_path, history_tensor)
        return history_tensor


class DualTFIDFVectorizer:
    """
    Manages separate TF-IDF vectorizers for Request Title and Request Body.
    Used primarily for the Random Forest stream of the ensemble.
    """

    def __init__(self, max_features=5000):
        self.title_vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self.body_vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self.is_fitted = False

    def fit(self, df):
        """
        Fits vectorizers on the training data.
        """
        titles = df[Config.TEXT_COL_TITLE].fillna("").astype(str)
        bodies = df[Config.TEXT_COL_BODY].fillna("").astype(str)

        self.title_vectorizer.fit(titles)
        self.body_vectorizer.fit(bodies)
        self.is_fitted = True
        return self

    def transform(self, df):
        """
        Transforms data into TF-IDF features.

        Returns:
            tuple: (title_sparse_matrix, body_sparse_matrix)
        """
        if not self.is_fitted:
            raise RuntimeError("DualTFIDFVectorizer must be fitted before transform.")

        titles = df[Config.TEXT_COL_TITLE].fillna("").astype(str)
        bodies = df[Config.TEXT_COL_BODY].fillna("").astype(str)

        title_tfidf = self.title_vectorizer.transform(titles)
        body_tfidf = self.body_vectorizer.transform(bodies)

        return title_tfidf, body_tfidf

    def fit_transform(self, df):
        """
        Fits and transforms in one step.
        """
        self.fit(df)
        return self.transform(df)
