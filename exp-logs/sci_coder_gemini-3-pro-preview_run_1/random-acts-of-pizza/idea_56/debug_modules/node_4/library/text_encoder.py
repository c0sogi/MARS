import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from library.config import Config
from library.utils import print_log, get_device


class SBERTEncoder:
    """
    Wrapper for Sentence-BERT embeddings to generate dense semantic vectors.
    Used for the MLP stream (Title, Body, History, Subreddits).
    """

    def __init__(self, model_name=Config.SBERT_MODEL_NAME):
        """
        Initialize the SBERT model.

        Args:
            model_name (str): Name of the pre-trained model to load.
        """
        self.device = get_device()
        print_log(
            f"Initializing SBERTEncoder with model: {model_name} on {self.device}"
        )
        self.model = SentenceTransformer(model_name, device=str(self.device))
        self.model.eval()

    def encode(
        self, texts, batch_size=32, show_progress_bar=False, normalize_embeddings=False
    ):
        """
        Encodes a list of text strings into dense vectors.

        Args:
            texts (list of str): List of strings to encode.
            batch_size (int): Batch size for encoding.
            show_progress_bar (bool): Whether to show progress.
            normalize_embeddings (bool): Whether to normalize vectors to unit length.

        Returns:
            np.ndarray: Array of shape (n_samples, embedding_dim).
        """
        # Handle empty or None inputs gracefully by replacing them with a single space
        # This prevents crashes in the underlying library
        cleaned_texts = [
            str(t) if t is not None and str(t).strip() != "" else " " for t in texts
        ]

        embeddings = self.model.encode(
            cleaned_texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            device=str(self.device),
            convert_to_numpy=True,
            normalize_embeddings=normalize_embeddings,
        )
        return embeddings


class TFIDFEncoder:
    """
    Wrapper for TF-IDF Vectorization intended for the Random Forest stream.
    Captures explicit keyword signals with n-grams.
    """

    def __init__(self, max_features=Config.TFIDF_VOCAB_SIZE, ngram_range=(1, 2)):
        """
        Initialize the TF-IDF Vectorizer.

        Args:
            max_features (int): Maximum vocabulary size.
            ngram_range (tuple): The lower and upper boundary of the range of n-values for different n-grams.
        """
        print_log(
            f"Initializing TFIDFEncoder with max_features={max_features}, ngram_range={ngram_range}"
        )
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words="english",
            ngram_range=ngram_range,
            sublinear_tf=True,  # Apply sublinear tf scaling: 1 + log(tf)
            dtype=np.float32,
        )

    def fit(self, texts):
        """
        Fits the TF-IDF vectorizer on the provided texts.

        Args:
            texts (list): List of text documents.
        """
        cleaned_texts = [str(t) if t is not None else "" for t in texts]
        self.vectorizer.fit(cleaned_texts)
        return self

    def transform(self, texts):
        """
        Transforms texts into a sparse TF-IDF matrix.

        Args:
            texts (list): List of text documents.

        Returns:
            scipy.sparse.csr_matrix: Sparse TF-IDF matrix.
        """
        cleaned_texts = [str(t) if t is not None else "" for t in texts]
        return self.vectorizer.transform(cleaned_texts)

    def fit_transform(self, texts):
        """
        Fits and transforms in one step.

        Args:
            texts (list): List of text documents.

        Returns:
            scipy.sparse.csr_matrix: Sparse TF-IDF matrix.
        """
        cleaned_texts = [str(t) if t is not None else "" for t in texts]
        return self.vectorizer.fit_transform(cleaned_texts)

    def get_feature_names_out(self):
        """
        Returns feature names from the fitted vectorizer.
        """
        return self.vectorizer.get_feature_names_out()
