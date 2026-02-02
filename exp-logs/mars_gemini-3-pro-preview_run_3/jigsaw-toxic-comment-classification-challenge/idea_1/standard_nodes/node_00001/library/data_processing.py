import os
import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.exceptions import NotFittedError
from library.utils import get_logger

# Configuration
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
CACHE_DIR = "./working/idea_1"
LABEL_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

logger = get_logger("data_processing")


def save_csr(matrix, filename):
    """
    Saves a CSR matrix to .npz using numpy only (no pickle).
    Stores data, indices, indptr, and shape as separate arrays in the archive.
    """
    np.savez(
        filename,
        data=matrix.data,
        indices=matrix.indices,
        indptr=matrix.indptr,
        shape=matrix.shape,
    )


def load_csr(filename):
    """
    Loads a CSR matrix from .npz using numpy only.
    Reconstructs the matrix from the stored arrays.
    """
    loader = np.load(filename)
    return sparse.csr_matrix(
        (loader["data"], loader["indices"], loader["indptr"]), shape=loader["shape"]
    )


def load_data(split="train", load_cached_data=True):
    """
    Loads data for a specific split (train, val, test).
    Uses metadata to align features and labels from the source CSVs.

    Args:
        split (str): One of 'train', 'val', 'test'.
        load_cached_data (bool): Whether to attempt loading from parquet cache.

    Returns:
        pd.DataFrame: The loaded dataframe with text and labels.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"{split}_merged.parquet")

    # 1. Try Cache
    if load_cached_data and os.path.exists(cache_path):
        logger.info(f"Loading {split} data from cache: {cache_path}")
        try:
            return pd.read_parquet(cache_path)
        except Exception as e:
            logger.warning(f"Failed to load cache {cache_path}: {e}. Recomputing...")

    logger.info(f"Processing {split} data from source...")

    # 2. Load Metadata
    meta_path = os.path.join(METADATA_DIR, f"{split}_metadata.csv")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    meta_df = pd.read_csv(meta_path)

    # 3. Load Source Data and Merge
    # We identify which source files are needed (usually just one)
    source_files = meta_df["source_file"].unique()
    merged_dfs = []

    for src_file in source_files:
        src_path = os.path.join(INPUT_DIR, src_file)
        if not os.path.exists(src_path):
            raise FileNotFoundError(f"Source file not found: {src_path}")

        # Read full source CSV
        full_source = pd.read_csv(src_path)

        # Filter metadata for this specific source file
        subset_meta = meta_df[meta_df["source_file"] == src_file]

        # Extract rows using integer position
        indices = subset_meta["source_row_index"].values
        extracted = full_source.iloc[indices].copy()

        # Drop potential label columns from source to avoid conflicts
        # (We trust metadata labels)
        cols_to_drop = [c for c in extracted.columns if c in LABEL_COLS]
        extracted = extracted.drop(columns=cols_to_drop, errors="ignore")

        # Add metadata columns to extracted dataframe to allow merging
        extracted["source_file"] = src_file
        extracted["source_row_index"] = extracted.index

        # Merge with metadata to attach correct labels and IDs
        # We merge on ID and source info to be precise
        extracted = extracted.merge(
            subset_meta, on=["id", "source_file", "source_row_index"], how="inner"
        )

        merged_dfs.append(extracted)

    final_df = pd.concat(merged_dfs, ignore_index=True)

    # 4. Preprocessing
    if "comment_text" in final_df.columns:
        final_df["comment_text"] = final_df["comment_text"].fillna("").astype(str)

    # 5. Save Cache
    logger.info(f"Saving {split} data to cache: {cache_path}")
    final_df.to_parquet(cache_path, index=False)

    return final_df


class FeatureEngineer:
    def __init__(self, max_features_word=None, max_features_char=None):
        """
        Initializes the TF-IDF vectorizers.
        """
        self.word_vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=3,
            max_df=0.9,
            strip_accents="unicode",
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=True,
            max_features=max_features_word,
        )
        self.char_vectorizer = TfidfVectorizer(
            ngram_range=(2, 6),
            min_df=3,
            max_df=0.9,
            strip_accents="unicode",
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=True,
            analyzer="char",
            max_features=max_features_char,
        )

    def fit_transform(self, text_series, load_cached_data=True, cache_suffix="train"):
        """
        Fits the vectorizers on the text and returns the transformed sparse matrix.
        Caches the result to disk.
        """
        os.makedirs(CACHE_DIR, exist_ok=True)
        cache_path = os.path.join(CACHE_DIR, f"features_{cache_suffix}.npz")

        if load_cached_data and os.path.exists(cache_path):
            logger.info(f"Loading features from cache: {cache_path}")
            # Note: This returns the matrix but does NOT restore vectorizer state.
            # If you need to transform test data later, you must re-fit or not use cache here.
            return load_csr(cache_path)

        logger.info("Fitting vectorizers and transforming text...")
        word_features = self.word_vectorizer.fit_transform(text_series)
        char_features = self.char_vectorizer.fit_transform(text_series)

        features = sparse.hstack([word_features, char_features]).tocsr()

        logger.info(f"Saving features to cache: {cache_path}")
        save_csr(features, cache_path)

        return features

    def transform(self, text_series, load_cached_data=True, cache_suffix="test"):
        """
        Transforms the text using the already fitted vectorizers.
        """
        os.makedirs(CACHE_DIR, exist_ok=True)
        cache_path = os.path.join(CACHE_DIR, f"features_{cache_suffix}.npz")

        if load_cached_data and os.path.exists(cache_path):
            logger.info(f"Loading features from cache: {cache_path}")
            return load_csr(cache_path)

        logger.info("Transforming text...")
        try:
            word_features = self.word_vectorizer.transform(text_series)
            char_features = self.char_vectorizer.transform(text_series)
        except NotFittedError:
            raise RuntimeError(
                "Vectorizers are not fitted. If you loaded training features from cache, "
                "the vectorizers were not updated. Disable cache loading for training "
                "if you intend to run inference in the same process."
            )

        features = sparse.hstack([word_features, char_features]).tocsr()

        logger.info(f"Saving features to cache: {cache_path}")
        save_csr(features, cache_path)

        return features
