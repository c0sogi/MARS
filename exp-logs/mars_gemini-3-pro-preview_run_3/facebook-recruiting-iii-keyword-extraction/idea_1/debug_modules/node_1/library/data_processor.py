import os
import re
import gc
import joblib
import numpy as np
import pandas as pd
import scipy.sparse
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer

from library.config import (
    INPUT_DIR,
    METADATA_DIR,
    WORKING_DIR,
    TRAIN_META_PATH,
    VAL_META_PATH,
    TEST_META_PATH,
    TFIDF_VECTORIZER_PATH,
    MLB_PATH,
    TRAIN_FEATURES_PATH,
    VAL_FEATURES_PATH,
    TEST_FEATURES_PATH,
    TRAIN_LABELS_PATH,
    VAL_LABELS_PATH,
    MAX_FEATURES,
    NGRAM_RANGE,
    MIN_DF,
    TOP_K_TAGS,
    set_seed,
)


class TextPreprocessor:
    """
    Handles cleaning and preprocessing of text data.
    """

    @staticmethod
    def clean_text(text):
        """
        Applies cleaning steps: lowercase, remove HTML, remove non-alphanumeric.
        """
        if not isinstance(text, str):
            return ""

        # Lowercase
        text = text.lower()

        # Remove HTML tags
        text = re.sub(r"<[^>]+>", " ", text)

        # Remove non-alphanumeric characters (keeping spaces)
        # We replace them with space to avoid merging words like "end.Start" -> "endStart"
        text = re.sub(r"[^a-z0-9\s]", " ", text)

        # Collapse multiple spaces
        text = re.sub(r"\s+", " ", text).strip()

        return text

    @staticmethod
    def preprocess_dataframe(df):
        """
        Combines Title and Body, then cleans the text.
        Returns a Series of cleaned text.
        """
        # Fill NaNs
        title = df["Title"].fillna("")
        body = df["Body"].fillna("")

        # Combine
        full_text = title + " " + body

        # Clean
        # Using apply is slower than vectorized string ops but allows regex flexibility
        # Given the dataset size, we'll use a compiled regex in a loop or map if needed,
        # but pandas str.replace with regex is reasonably optimized.
        # However, for complex custom logic, map is often used.
        return full_text.map(TextPreprocessor.clean_text)


class FeatureEngineer:
    """
    Manages TF-IDF Vectorization and Multi-Label Binarization.
    """

    def __init__(self):
        self.tfidf = TfidfVectorizer(
            max_features=MAX_FEATURES,
            ngram_range=NGRAM_RANGE,
            min_df=MIN_DF,
            stop_words="english",
            dtype=np.float32,
        )
        self.mlb = None  # Initialized in fit_labels

    def fit_text(self, text_series):
        """Fits the TF-IDF vectorizer."""
        print("Fitting TF-IDF Vectorizer...")
        self.tfidf.fit(text_series)

    def transform_text(self, text_series):
        """Transforms text to TF-IDF sparse matrix."""
        print("Transforming text to TF-IDF features...")
        return self.tfidf.transform(text_series)

    def fit_labels(self, tags_series):
        """
        Determines top K tags and fits MultiLabelBinarizer.
        tags_series: pd.Series of space-delimited tag strings.
        """
        print(f"Fitting Label Binarizer (Top {TOP_K_TAGS} tags)...")

        # Split all tags
        all_tags = []
        for tags in tags_series:
            if isinstance(tags, str):
                all_tags.extend(tags.split())

        # Count frequencies
        counts = Counter(all_tags)

        # Select Top K
        top_tags = [tag for tag, count in counts.most_common(TOP_K_TAGS)]

        # Initialize MLB with specific classes
        self.mlb = MultiLabelBinarizer(classes=sorted(top_tags), sparse_output=True)

        # We don't strictly need to 'fit' MLB again if we provided classes,
        # but calling fit on a dummy list ensures internal state is set if needed.
        # However, passing classes to __init__ is sufficient for transform.
        # To be safe and follow sklearn pattern:
        self.mlb.fit([[]])

    def transform_labels(self, tags_series):
        """
        Transforms space-delimited tags to binary sparse matrix.
        """
        print("Transforming labels...")
        # Convert space-delimited strings to lists of strings
        tags_list = tags_series.fillna("").str.split().tolist()
        return self.mlb.transform(tags_list)

    def inverse_transform_labels(self, binary_matrix):
        """
        Converts binary matrix predictions back to space-delimited strings.
        """
        # mlb.inverse_transform returns a list of tuples of tags
        tags_tuples = self.mlb.inverse_transform(binary_matrix)
        return [" ".join(tags) for tags in tags_tuples]

    def save(self):
        """Saves the vectorizer and binarizer to disk."""
        print(f"Saving FeatureEngineer artifacts to {WORKING_DIR}...")
        joblib.dump(self.tfidf, TFIDF_VECTORIZER_PATH)
        joblib.dump(self.mlb, MLB_PATH)

    def load(self):
        """Loads the vectorizer and binarizer from disk."""
        print(f"Loading FeatureEngineer artifacts from {WORKING_DIR}...")
        if os.path.exists(TFIDF_VECTORIZER_PATH) and os.path.exists(MLB_PATH):
            self.tfidf = joblib.load(TFIDF_VECTORIZER_PATH)
            self.mlb = joblib.load(MLB_PATH)
            return True
        return False


def load_raw_data_split(meta_path, raw_file_name, use_tags=True):
    """
    Loads raw data based on metadata file.
    """
    print(f"Loading metadata from {meta_path}...")
    df_meta = pd.read_csv(meta_path)

    print(f"Loading raw data from {os.path.join(INPUT_DIR, raw_file_name)}...")
    cols = ["Id", "Title", "Body"]

    # Only load Tags from raw data if requested AND not already present in metadata
    if use_tags and "Tags" not in df_meta.columns:
        cols.append("Tags")

    # Load raw data
    # Note: raw files are large, but we filter by ID via merge
    # To optimize, we read the whole raw file (assuming memory allows) and merge.
    # Given 220GB RAM, reading 6M rows is fine.
    df_raw = pd.read_csv(os.path.join(INPUT_DIR, raw_file_name), usecols=cols)

    # Merge to get the specific split
    df = pd.merge(df_meta, df_raw, on="Id", how="inner")

    return df


def process_data(load_cached_data=True):
    """
    Main data processing pipeline.
    Returns: X_train, y_train, X_val, y_val, X_test, feature_engineer
    """
    set_seed()
    os.makedirs(WORKING_DIR, exist_ok=True)

    # 1. Check Cache
    if load_cached_data:
        print("Checking for cached data...")
        files_exist = all(
            os.path.exists(p)
            for p in [
                TRAIN_FEATURES_PATH,
                VAL_FEATURES_PATH,
                TEST_FEATURES_PATH,
                TRAIN_LABELS_PATH,
                VAL_LABELS_PATH,
                TFIDF_VECTORIZER_PATH,
                MLB_PATH,
            ]
        )

        if files_exist:
            print("Loading cached features and labels...")
            X_train = scipy.sparse.load_npz(TRAIN_FEATURES_PATH)
            y_train = scipy.sparse.load_npz(TRAIN_LABELS_PATH)
            X_val = scipy.sparse.load_npz(VAL_FEATURES_PATH)
            y_val = scipy.sparse.load_npz(VAL_LABELS_PATH)
            X_test = scipy.sparse.load_npz(TEST_FEATURES_PATH)

            fe = FeatureEngineer()
            fe.load()

            return X_train, y_train, X_val, y_val, X_test, fe
        else:
            print(
                "Cached data not found or incomplete. Starting processing from scratch..."
            )

    # 2. Load and Process Training Data
    print("\n--- Processing Training Data ---")
    df_train = load_raw_data_split(TRAIN_META_PATH, "train.csv", use_tags=True)
    train_text = TextPreprocessor.preprocess_dataframe(df_train)
    train_tags = df_train["Tags"]

    # Initialize and Fit Feature Engineer
    fe = FeatureEngineer()
    fe.fit_text(train_text)
    fe.fit_labels(train_tags)

    # Transform Training Data
    X_train = fe.transform_text(train_text)
    y_train = fe.transform_labels(train_tags)

    # Clean up memory
    del df_train, train_text, train_tags
    gc.collect()

    # 3. Load and Process Validation Data
    print("\n--- Processing Validation Data ---")
    df_val = load_raw_data_split(VAL_META_PATH, "train.csv", use_tags=True)
    val_text = TextPreprocessor.preprocess_dataframe(df_val)
    val_tags = df_val["Tags"]

    X_val = fe.transform_text(val_text)
    y_val = fe.transform_labels(val_tags)

    del df_val, val_text, val_tags
    gc.collect()

    # 4. Load and Process Test Data
    print("\n--- Processing Test Data ---")
    df_test = load_raw_data_split(TEST_META_PATH, "test.csv", use_tags=False)
    test_text = TextPreprocessor.preprocess_dataframe(df_test)

    X_test = fe.transform_text(test_text)

    del df_test, test_text
    gc.collect()

    # 5. Save Artifacts
    print("\n--- Saving Artifacts ---")
    scipy.sparse.save_npz(TRAIN_FEATURES_PATH, X_train)
    scipy.sparse.save_npz(TRAIN_LABELS_PATH, y_train)
    scipy.sparse.save_npz(VAL_FEATURES_PATH, X_val)
    scipy.sparse.save_npz(VAL_LABELS_PATH, y_val)
    scipy.sparse.save_npz(TEST_FEATURES_PATH, X_test)
    fe.save()

    return X_train, y_train, X_val, y_val, X_test, fe
