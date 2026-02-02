import os
import numpy as np
import pandas as pd
import scipy.sparse
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from library.config import Config
from library.utils import save_artifact, load_artifact


class TextVectorizer:
    """
    Wraps CountVectorizer to convert text to sparse Bag-of-Words matrices.
    """

    def __init__(self):
        self.vectorizer = CountVectorizer(
            max_features=Config.VOCAB_SIZE,
            ngram_range=Config.NGRAM_RANGE,
            min_df=Config.MIN_DF,
            dtype=np.float32,
            token_pattern=r"(?u)\b\w+\b",  # Simple token pattern
        )
        self.is_fitted = False

    def fit(self, texts):
        """
        Fits the vectorizer on the provided texts.
        """
        print(f"Fitting TextVectorizer on {len(texts)} documents...")
        self.vectorizer.fit(texts)
        self.is_fitted = True
        return self

    def transform(self, texts):
        """
        Transforms texts to a sparse matrix.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "TextVectorizer must be fitted before calling transform."
            )

        print(f"Transforming {len(texts)} documents to BoW features...")
        return self.vectorizer.transform(texts)

    def save(self, filepath):
        """
        Saves the fitted vectorizer object.
        """
        save_artifact(self.vectorizer, filepath)

    @classmethod
    def load(cls, filepath):
        """
        Loads a fitted vectorizer object.
        """
        instance = cls()
        instance.vectorizer = load_artifact(filepath)
        instance.is_fitted = True
        return instance


class TagEncoder:
    """
    Encodes lists of tags into binary sparse matrices (One-vs-Rest format)
    and decodes binary matrices back to tag lists.
    """

    def __init__(self):
        self.tag_to_idx = {}
        self.idx_to_tag = {}
        self.top_k = Config.TOP_K_TAGS
        self.is_fitted = False

    def fit(self, tags_series):
        """
        Identifies the top K most frequent tags and builds the vocabulary.
        Args:
            tags_series: A pandas Series or list where each element is a list of tags.
        """
        print("Fitting TagEncoder...")
        # Flatten the list of lists
        all_tags = [tag for tags in tags_series for tag in tags]

        # Count frequencies
        counts = Counter(all_tags)

        # Select top K tags
        most_common = counts.most_common(self.top_k)

        # Build mappings
        self.tag_to_idx = {tag: i for i, (tag, _) in enumerate(most_common)}
        self.idx_to_tag = {i: tag for tag, i in self.tag_to_idx.items()}

        self.is_fitted = True
        print(f"TagEncoder fitted. Vocab size: {len(self.tag_to_idx)}")
        return self

    def transform(self, tags_series):
        """
        Converts a series of tag lists into a binary sparse matrix.
        """
        if not self.is_fitted:
            raise RuntimeError("TagEncoder must be fitted before calling transform.")

        print(f"Encoding targets for {len(tags_series)} samples...")

        n_samples = len(tags_series)
        n_classes = len(self.tag_to_idx)

        # Use lil_matrix for efficient incremental construction
        matrix = scipy.sparse.lil_matrix((n_samples, n_classes), dtype=np.int8)

        for i, tags in enumerate(tags_series):
            for tag in tags:
                if tag in self.tag_to_idx:
                    idx = self.tag_to_idx[tag]
                    matrix[i, idx] = 1

        return matrix.tocsr()

    def inverse_transform(self, binary_matrix):
        """
        Converts a binary matrix (sparse or dense) back to a list of space-delimited tag strings.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "TagEncoder must be fitted before calling inverse_transform."
            )

        # Ensure input is CSR for efficient row slicing
        if not scipy.sparse.isspmatrix_csr(binary_matrix):
            binary_matrix = scipy.sparse.csr_matrix(binary_matrix)

        n_samples = binary_matrix.shape[0]
        result = []

        # Get indices of non-zero elements
        rows, cols = binary_matrix.nonzero()

        # Group columns by row index
        # Since nonzero() returns arrays sorted by row index, we can iterate efficiently
        current_row = 0
        current_tags = []

        # Helper to process row-wise.
        # Alternatively, we can iterate row by row using the sparse structure directly.
        for i in range(n_samples):
            # Get column indices for this row
            row_cols = binary_matrix.indices[
                binary_matrix.indptr[i] : binary_matrix.indptr[i + 1]
            ]
            tags = [self.idx_to_tag[idx] for idx in row_cols]
            result.append(" ".join(tags))

        return result

    def save(self, filepath):
        save_artifact(self, filepath)

    @staticmethod
    def load(filepath):
        return load_artifact(filepath)


def get_text_features(df, split, vectorizer=None, load_cached_data=True):
    """
    Manages caching and computation of text features.

    Args:
        df: DataFrame containing the 'text' column.
        split: 'train', 'val', or 'test'.
        vectorizer: Fitted TextVectorizer instance (required for val/test).
        load_cached_data: Whether to try loading from cache.

    Returns:
        tuple: (features_matrix, vectorizer)
    """
    os.makedirs(Config.WORK_DIR, exist_ok=True)

    feature_path = os.path.join(Config.WORK_DIR, f"{split}_bow_features.npz")
    vectorizer_path = os.path.join(Config.WORK_DIR, "text_vectorizer.pkl")

    # 1. Try loading from cache
    if load_cached_data and os.path.exists(feature_path):
        print(f"Loading {split} text features from cache: {feature_path}")
        features = load_artifact(feature_path)

        # If vectorizer is not provided (e.g. during inference loading), try loading it too
        if vectorizer is None and os.path.exists(vectorizer_path):
            vectorizer = TextVectorizer.load(vectorizer_path)

        return features, vectorizer

    # 2. Compute from scratch
    print(f"Computing {split} text features from scratch...")

    if split == "train":
        if vectorizer is None:
            vectorizer = TextVectorizer()
        vectorizer.fit(df["text"])
        # Save the vectorizer
        vectorizer.save(vectorizer_path)
    else:
        if vectorizer is None:
            # Try to load if not passed
            if os.path.exists(vectorizer_path):
                vectorizer = TextVectorizer.load(vectorizer_path)
            else:
                raise ValueError(
                    "Vectorizer must be provided or cached for validation/test splits."
                )

    features = vectorizer.transform(df["text"])

    # 3. Save to cache
    print(f"Saving {split} text features to cache: {feature_path}")
    save_artifact(features, feature_path)

    return features, vectorizer


def get_target_matrix(df, split, encoder=None, load_cached_data=True):
    """
    Manages caching and computation of target matrices.

    Args:
        df: DataFrame containing the 'tags_list' column.
        split: 'train', 'val', or 'test' (test usually doesn't have targets).
        encoder: Fitted TagEncoder instance.
        load_cached_data: Whether to try loading from cache.

    Returns:
        tuple: (target_matrix, encoder)
    """
    os.makedirs(Config.WORK_DIR, exist_ok=True)

    target_path = os.path.join(Config.WORK_DIR, f"{split}_target_matrix.npz")
    encoder_path = os.path.join(Config.WORK_DIR, "tag_encoder.pkl")

    # 1. Try loading from cache
    if load_cached_data and os.path.exists(target_path):
        print(f"Loading {split} target matrix from cache: {target_path}")
        targets = load_artifact(target_path)

        if encoder is None and os.path.exists(encoder_path):
            encoder = TagEncoder.load(encoder_path)

        return targets, encoder

    # 2. Compute from scratch
    print(f"Computing {split} target matrix from scratch...")

    if split == "train":
        if encoder is None:
            encoder = TagEncoder()
        encoder.fit(df["tags_list"])
        # Save the encoder
        encoder.save(encoder_path)
    else:
        if encoder is None:
            if os.path.exists(encoder_path):
                encoder = TagEncoder.load(encoder_path)
            else:
                raise ValueError(
                    "Encoder must be provided or cached for validation split."
                )

    targets = encoder.transform(df["tags_list"])

    # 3. Save to cache
    print(f"Saving {split} target matrix to cache: {target_path}")
    save_artifact(targets, target_path)

    return targets, encoder
