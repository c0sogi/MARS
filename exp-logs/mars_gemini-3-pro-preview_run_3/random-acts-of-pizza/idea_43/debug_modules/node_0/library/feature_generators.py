import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.impute import SimpleImputer
from sentence_transformers import SentenceTransformer
from library.config import Config
from library.data_loader import get_metadata


class FeatureCacheMixin:
    """
    Mixin class to handle caching logic for deterministic feature generation.
    """

    def _get_cache_path(self, name, ext=".npy"):
        return os.path.join(Config.CACHE_DIR, f"{name}{ext}")

    def _load_cache(self, name, ext=".npy"):
        path = self._get_cache_path(name, ext)
        if os.path.exists(path):
            if ext == ".npy":
                return np.load(path)
            elif ext == ".parquet":
                return pd.read_parquet(path)
        return None

    def _save_cache(self, data, name, ext=".npy"):
        Config.ensure_directories()
        path = self._get_cache_path(name, ext)
        if ext == ".npy":
            np.save(path, data)
        elif ext == ".parquet":
            data.to_parquet(path, index=False)


class TextProcessor:
    """
    Handles text extraction and concatenation.
    """

    def __init__(self):
        self.title_col = "request_title"
        self.text_col = "request_text_edit_aware"

    def process(self, df):
        """
        Concatenates title and body text.
        Returns a pandas Series of strings.
        """
        # Fill NaNs with empty strings to ensure string operations work
        title = df[self.title_col].fillna("").astype(str)
        body = df[self.text_col].fillna("").astype(str)

        # Concatenate with a space separator
        combined_text = title + " " + body
        return combined_text


class SentenceEmbedder(FeatureCacheMixin):
    """
    Generates dense embeddings using SentenceTransformers.
    """

    def __init__(self):
        self.model_name = Config.EMBEDDING_MODEL
        self.batch_size = 32

    def transform(self, text_series, cache_key=None, load_cached_data=True):
        """
        Generates embeddings for the given text series.

        Args:
            text_series (pd.Series): Series containing text to embed.
            cache_key (str): Unique identifier for caching (e.g., 'train_embeddings').
            load_cached_data (bool): Whether to attempt loading from cache.

        Returns:
            np.ndarray: Dense embeddings (N_samples, 384).
        """
        # Try loading from cache
        if load_cached_data and cache_key:
            cached_data = self._load_cache(cache_key, ext=".npy")
            if cached_data is not None:
                # Verify shape matches
                if len(cached_data) == len(text_series):
                    return cached_data
                else:
                    # If shape mismatch (e.g. different dataset size in debug mode), recompute
                    pass

        # Compute embeddings
        model = SentenceTransformer(self.model_name)

        # Ensure input is a list of strings
        texts = text_series.tolist()

        # Encode
        embeddings = model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        # Save to cache if key provided
        if cache_key:
            self._save_cache(embeddings, cache_key, ext=".npy")

        return embeddings


class LatentCommunityInjector:
    """
    Implements the NMF-based Latent Community Injection.
    Transforms sparse subreddit history into dense topic features.
    """

    def __init__(self):
        self.n_components = Config.NMF_COMPONENTS
        self.max_vocab = Config.MAX_SUBREDDITS_VOCAB
        self.tfidf = None
        self.nmf = None
        self.col_name = "requester_subreddits_at_request"

    def _prepare_text(self, df):
        """
        Converts the list of subreddits into a space-separated string.
        """
        if self.col_name not in df.columns:
            # Fallback if column missing, though data_loader should handle it
            return pd.Series(["" for _ in range(len(df))])

        return df[self.col_name].apply(
            lambda x: " ".join(x) if isinstance(x, (list, np.ndarray)) else ""
        )

    def fit(self, df):
        """
        Fits the TF-IDF Vectorizer and NMF model on the provided dataframe.
        """
        documents = self._prepare_text(df)

        # TF-IDF Vectorizer
        # We use a token pattern that captures subreddit names (alphanumeric + underscores)
        self.tfidf = TfidfVectorizer(
            max_features=self.max_vocab,
            token_pattern=r"(?u)\b\w+\b",
            stop_words="english",  # Basic stop words to remove common noise if any
            lowercase=True,
        )
        tfidf_matrix = self.tfidf.fit_transform(documents)

        # NMF Model
        self.nmf = NMF(
            n_components=self.n_components,
            init="nndsvd",
            random_state=Config.SEED,
            max_iter=500,
        )
        self.nmf.fit(tfidf_matrix)
        return self

    def transform(self, df):
        """
        Transforms the dataframe into NMF topic features.
        Returns a numpy array of shape (n_samples, n_components).
        """
        if self.tfidf is None or self.nmf is None:
            raise ValueError("LatentCommunityInjector must be fit before transform.")

        documents = self._prepare_text(df)
        tfidf_matrix = self.tfidf.transform(documents)
        nmf_features = self.nmf.transform(tfidf_matrix)

        return nmf_features


class MetadataAugmenter:
    """
    Extracts numerical metadata, imputes missing values, and augments with NMF topics.
    """

    def __init__(self):
        self.imputer = None
        self.feature_names = None

    def fit(self, df):
        """
        Fits the imputer on the safe metadata features.
        """
        meta_df = get_metadata(df)

        # Drop non-numerical columns that might have slipped in (like lists)
        # get_metadata returns 'requester_subreddits_at_request' which is a list.
        # We must exclude it from the numerical imputer.
        numeric_df = meta_df.select_dtypes(include=[np.number])

        self.feature_names = numeric_df.columns.tolist()

        self.imputer = SimpleImputer(strategy="median")
        self.imputer.fit(numeric_df)
        return self

    def transform(self, df, nmf_features=None):
        """
        Extracts metadata, imputes, and concatenates with NMF features.

        Args:
            df (pd.DataFrame): Input dataframe.
            nmf_features (np.ndarray, optional): Dense NMF features to augment.

        Returns:
            np.ndarray: Augmented metadata matrix.
        """
        if self.imputer is None:
            raise ValueError("MetadataAugmenter must be fit before transform.")

        meta_df = get_metadata(df)

        # Ensure we select the same columns as fit
        numeric_df = meta_df[self.feature_names]

        # Impute
        imputed_meta = self.imputer.transform(numeric_df)

        # Augment with NMF if provided
        if nmf_features is not None:
            if len(nmf_features) != len(imputed_meta):
                raise ValueError(
                    f"Shape mismatch: Metadata {len(imputed_meta)} vs NMF {len(nmf_features)}"
                )

            # Concatenate along columns
            final_features = np.hstack([imputed_meta, nmf_features])
        else:
            final_features = imputed_meta

        return final_features
