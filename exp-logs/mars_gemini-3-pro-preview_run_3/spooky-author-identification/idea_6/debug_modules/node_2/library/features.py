import os
import numpy as np
import pandas as pd
import scipy.sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from library.config import Config


class StylometricExtractor:
    """
    Extracts explicit stylometric features from text data.
    Features:
    - Sentence Length (Character and Word counts)
    - Punctuation Density
    - Lexical Diversity (Type-Token Ratio)
    - Average Word Length
    """

    def __init__(self):
        self.punct_set = [";", ":", ",", "?", "!", "-"]

    def _extract_single(self, text):
        text = str(text)
        char_len = len(text)
        words = text.split()
        word_len = len(words)

        # Avoid zero division
        safe_word_len = word_len if word_len > 0 else 1

        # Lexical diversity (Type-Token Ratio)
        unique_words = len(set(words))
        ttr = unique_words / safe_word_len

        # Average word length
        avg_word_len = char_len / safe_word_len

        # Punctuation density (count per character to normalize for length)
        # Using max(char_len, 1) to avoid division by zero
        safe_char_len = max(char_len, 1)
        punct_counts = [text.count(p) for p in self.punct_set]
        punct_density = [c / safe_char_len for c in punct_counts]

        return [char_len, word_len, avg_word_len, ttr] + punct_density

    def transform(self, texts):
        """
        Transforms a list/series of texts into a dense feature matrix.
        """
        features = [self._extract_single(t) for t in texts]
        return np.array(features, dtype=np.float32)


def save_sparse_npy(path_prefix, matrix):
    """
    Saves a CSR matrix as separate .npy files to avoid using pickle.
    """
    if not isinstance(matrix, scipy.sparse.csr_matrix):
        matrix = matrix.tocsr()

    np.save(f"{path_prefix}_data.npy", matrix.data)
    np.save(f"{path_prefix}_indices.npy", matrix.indices)
    np.save(f"{path_prefix}_indptr.npy", matrix.indptr)
    np.save(f"{path_prefix}_shape.npy", np.array(matrix.shape))


def load_sparse_npy(path_prefix):
    """
    Loads a CSR matrix from separate .npy files.
    """
    data = np.load(f"{path_prefix}_data.npy")
    indices = np.load(f"{path_prefix}_indices.npy")
    indptr = np.load(f"{path_prefix}_indptr.npy")
    shape = np.load(f"{path_prefix}_shape.npy")

    return scipy.sparse.csr_matrix((data, indices, indptr), shape=tuple(shape))


def get_data(load_cached_data=True, debug=Config.DEBUG):
    """
    Main function to load data, extract features, and return processed matrices.

    Args:
        load_cached_data (bool): If True, attempts to load from disk.
        debug (bool): If True, uses a subset of data and appends '_debug' to cache filenames.

    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, test_ids)
    """
    # Define cache paths
    cache_dir = Config.FEATURES_DIR
    suffix = "_debug" if debug else ""

    # File prefixes
    train_prefix = os.path.join(cache_dir, f"X_train_combined{suffix}")
    val_prefix = os.path.join(cache_dir, f"X_val_combined{suffix}")
    test_prefix = os.path.join(cache_dir, f"X_test_combined{suffix}")

    y_train_path = os.path.join(cache_dir, f"y_train{suffix}.npy")
    y_val_path = os.path.join(cache_dir, f"y_val{suffix}.npy")
    test_ids_path = os.path.join(cache_dir, f"test_ids{suffix}.npy")

    # 1. Try Loading from Cache
    if load_cached_data:
        try:
            print(f"Attempting to load features from {cache_dir}...")
            X_train = load_sparse_npy(train_prefix)
            X_val = load_sparse_npy(val_prefix)
            X_test = load_sparse_npy(test_prefix)

            y_train = np.load(y_train_path)
            y_val = np.load(y_val_path)
            test_ids = np.load(test_ids_path, allow_pickle=True)  # IDs are strings

            print("Successfully loaded features from cache.")
            return X_train, y_train, X_val, y_val, X_test, test_ids
        except (FileNotFoundError, IOError) as e:
            print(f"Cache miss or load error ({e}). Computing features from scratch...")
    else:
        print("Force recompute requested. Computing features from scratch...")

    # 2. Load Raw Data
    print("Loading raw metadata...")
    df_train = pd.read_csv(Config.TRAIN_DATA_PATH)
    df_val = pd.read_csv(Config.VAL_DATA_PATH)
    df_test = pd.read_csv(Config.TEST_DATA_PATH)

    # Handle Debug Mode
    if debug:
        print(f"DEBUG mode: Sampling {Config.DEBUG_SAMPLE_SIZE} rows.")
        df_train = df_train.head(Config.DEBUG_SAMPLE_SIZE)
        df_val = df_val.head(Config.DEBUG_SAMPLE_SIZE)
        df_test = df_test.head(Config.DEBUG_SAMPLE_SIZE)

    # Extract Text and Labels
    train_text = df_train["text"].fillna("").astype(str)
    val_text = df_val["text"].fillna("").astype(str)
    test_text = df_test["text"].fillna("").astype(str)

    y_train = df_train["author"].map(Config.LABEL_MAP).values
    y_val = df_val["author"].map(Config.LABEL_MAP).values
    test_ids = df_test["id"].values

    # 3. TF-IDF Features (Sparse)
    print("Generating TF-IDF features...")

    # Word N-grams
    word_vectorizer = TfidfVectorizer(
        ngram_range=Config.TFIDF_PARAMS["ngram_range_word"],
        max_features=Config.TFIDF_PARAMS["max_features_word"],
        sublinear_tf=Config.TFIDF_PARAMS["sublinear_tf"],
        analyzer="word",
        token_pattern=r"\w{1,}",
    )

    # Char N-grams
    char_vectorizer = TfidfVectorizer(
        ngram_range=Config.TFIDF_PARAMS["ngram_range_char"],
        max_features=Config.TFIDF_PARAMS["max_features_char"],
        sublinear_tf=Config.TFIDF_PARAMS["sublinear_tf"],
        analyzer="char",
    )

    # Fit on Train, Transform All
    print("Fitting Word Vectorizer...")
    X_train_word = word_vectorizer.fit_transform(train_text)
    X_val_word = word_vectorizer.transform(val_text)
    X_test_word = word_vectorizer.transform(test_text)

    print("Fitting Char Vectorizer...")
    X_train_char = char_vectorizer.fit_transform(train_text)
    X_val_char = char_vectorizer.transform(val_text)
    X_test_char = char_vectorizer.transform(test_text)

    # 4. Stylometric Features (Dense)
    print("Generating Stylometric features...")
    stylo_extractor = StylometricExtractor()

    X_train_stylo = stylo_extractor.transform(train_text)
    X_val_stylo = stylo_extractor.transform(val_text)
    X_test_stylo = stylo_extractor.transform(test_text)

    # Scale Dense Features
    scaler = StandardScaler()
    X_train_stylo = scaler.fit_transform(X_train_stylo)
    X_val_stylo = scaler.transform(X_val_stylo)
    X_test_stylo = scaler.transform(X_test_stylo)

    # Convert to sparse for efficient stacking
    X_train_stylo_sparse = scipy.sparse.csr_matrix(X_train_stylo)
    X_val_stylo_sparse = scipy.sparse.csr_matrix(X_val_stylo)
    X_test_stylo_sparse = scipy.sparse.csr_matrix(X_test_stylo)

    # 5. Combine Features
    print("Stacking features...")
    X_train = scipy.sparse.hstack(
        [X_train_word, X_train_char, X_train_stylo_sparse]
    ).tocsr()
    X_val = scipy.sparse.hstack([X_val_word, X_val_char, X_val_stylo_sparse]).tocsr()
    X_test = scipy.sparse.hstack(
        [X_test_word, X_test_char, X_test_stylo_sparse]
    ).tocsr()

    # 6. Save to Cache
    print(f"Saving features to {cache_dir}...")
    save_sparse_npy(train_prefix, X_train)
    save_sparse_npy(val_prefix, X_val)
    save_sparse_npy(test_prefix, X_test)

    np.save(y_train_path, y_train)
    np.save(y_val_path, y_val)
    np.save(test_ids_path, test_ids)

    print("Feature generation complete.")
    return X_train, y_train, X_val, y_val, X_test, test_ids
