import os
import numpy as np
import pandas as pd
import joblib
from collections import Counter
from scipy import sparse


class TagEncoder:
    """
    Encodes and decodes tags into a sparse binary matrix format.
    Restricts the vocabulary to the top_k most frequent tags.
    """

    def __init__(self, top_k=5000):
        self.top_k = top_k
        self.classes_ = None
        self.tag_to_idx_ = None
        self.idx_to_tag_ = None

    def fit(self, tags_series):
        """
        Fits the encoder to the provided tags series.
        Identifies the top_k most frequent tags.

        Args:
            tags_series (pd.Series or list): Series of space-delimited tag strings.
        """
        # Count all tags
        tag_counts = Counter()

        # Iterate and count
        # We assume input is iterable of strings
        for tags in tags_series:
            if isinstance(tags, str):
                tag_counts.update(tags.split())

        # Select top K
        most_common = tag_counts.most_common(self.top_k)

        # Build vocabulary
        self.classes_ = [tag for tag, count in most_common]
        self.tag_to_idx_ = {tag: i for i, tag in enumerate(self.classes_)}
        self.idx_to_tag_ = {i: tag for i, tag in enumerate(self.classes_)}

        return self

    def transform(self, tags_series):
        """
        Transforms a series of tag strings into a sparse binary matrix.

        Args:
            tags_series (pd.Series or list): Series of space-delimited tag strings.

        Returns:
            scipy.sparse.csr_matrix: Binary matrix of shape (n_samples, n_classes).
        """
        if self.tag_to_idx_ is None:
            raise ValueError("TagEncoder must be fitted before calling transform.")

        # Convert to list if series to ensure consistent indexing
        tags_list = (
            tags_series.tolist()
            if isinstance(tags_series, pd.Series)
            else list(tags_series)
        )
        n_samples = len(tags_list)
        n_classes = len(self.classes_)

        rows = []
        cols = []

        for i, tags in enumerate(tags_list):
            if isinstance(tags, str):
                for tag in tags.split():
                    if tag in self.tag_to_idx_:
                        rows.append(i)
                        cols.append(self.tag_to_idx_[tag])

        # Create sparse matrix (binary)
        # We use int8 to save memory
        data = np.ones(len(rows), dtype=np.int8)
        matrix = sparse.csr_matrix((data, (rows, cols)), shape=(n_samples, n_classes))

        return matrix

    def inverse_transform(self, matrix_or_indices):
        """
        Converts a binary matrix or list of indices back to space-delimited tag strings.

        Args:
            matrix_or_indices: sparse matrix, dense array, or list of lists of indices.

        Returns:
            List of strings.
        """
        if self.idx_to_tag_ is None:
            raise ValueError(
                "TagEncoder must be fitted before calling inverse_transform."
            )

        result = []

        # Helper to process a list of indices
        def indices_to_string(indices):
            tags = [self.idx_to_tag_[idx] for idx in indices if idx in self.idx_to_tag_]
            return " ".join(tags)

        # Handle Sparse Matrix
        if sparse.issparse(matrix_or_indices):
            matrix = matrix_or_indices.tocsr()
            for i in range(matrix.shape[0]):
                indices = matrix[i].indices
                result.append(indices_to_string(indices))

        # Handle Dense Array
        elif isinstance(matrix_or_indices, np.ndarray):
            for i in range(matrix_or_indices.shape[0]):
                indices = np.where(matrix_or_indices[i])[0]
                result.append(indices_to_string(indices))

        # Handle List of Lists
        elif isinstance(matrix_or_indices, list):
            for indices in matrix_or_indices:
                result.append(indices_to_string(indices))

        else:
            raise TypeError(
                "Input must be a sparse matrix, numpy array, or list of lists."
            )

        return result

    def save(self, path):
        """Saves the encoder object."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path):
        """Loads the encoder object."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"TagEncoder file not found: {path}")
        return joblib.load(path)


def get_target_matrix(
    df: pd.DataFrame,
    split: str,
    encoder: TagEncoder = None,
    load_cached_data: bool = True,
    cache_dir: str = "./working/idea_2",
):
    """
    Retrieves the target sparse matrix for a given split.
    Handles caching using .npz format.

    Args:
        df (pd.DataFrame): Dataframe containing 'Tags' column.
        split (str): The dataset split ('train', 'val').
        encoder (TagEncoder): The encoder instance.
        load_cached_data (bool): Whether to attempt loading from cache.
        cache_dir (str): Directory to store cached files.

    Returns:
        scipy.sparse.csr_matrix: The target binary matrix.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{split}_target_matrix.npz")

    # 1. Attempt to load from cache
    if load_cached_data:
        if os.path.exists(cache_path):
            print(f"Loading cached target matrix for '{split}' from {cache_path}...")
            try:
                return sparse.load_npz(cache_path)
            except Exception as e:
                print(f"Failed to load cache: {e}. Proceeding to re-process.")
        else:
            print(f"No cache found for '{split}' at {cache_path}.")

    # 2. Compute from scratch
    if encoder is None:
        raise ValueError("Encoder instance must be provided to compute targets.")

    if "Tags" not in df.columns:
        raise ValueError(
            f"DataFrame for split '{split}' does not contain 'Tags' column."
        )

    print(f"Computing target matrix for '{split}'...")
    tags_series = df["Tags"].fillna("").astype(str)

    # If training split, fit the encoder if not already fitted
    if split == "train" and encoder.classes_ is None:
        print("Fitting encoder on training data...")
        encoder.fit(tags_series)

    matrix = encoder.transform(tags_series)

    # 3. Save to cache
    print(f"Saving target matrix to {cache_path}...")
    sparse.save_npz(cache_path, matrix)

    return matrix
