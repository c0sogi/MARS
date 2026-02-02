import os
import scipy.sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from library.config import Config


class FeatureExtractor:
    """
    Feature extraction module for Toxicity Classification.
    Generates TF-IDF features (Word + Char n-grams) and handles caching
    of sparse matrices to optimize runtime.
    """

    def __init__(self):
        """
        Initialize vectorizers based on Config parameters.
        """
        self.cache_dir = Config.WORKING_DIR
        os.makedirs(self.cache_dir, exist_ok=True)

        # Word N-gram Vectorizer
        # Captures semantic meaning of words and phrases
        self.word_vectorizer = TfidfVectorizer(
            ngram_range=Config.WORD_NGRAM_RANGE,
            max_features=Config.WORD_MAX_FEATURES,
            min_df=Config.WORD_MIN_DF,
            analyzer="word",
            token_pattern=r"\w{1,}",
            strip_accents="unicode",
            sublinear_tf=True,
            use_idf=True,
        )

        # Character N-gram Vectorizer
        # Captures subword information and obfuscated spellings
        self.char_vectorizer = TfidfVectorizer(
            ngram_range=Config.CHAR_NGRAM_RANGE,
            max_features=Config.CHAR_MAX_FEATURES,
            min_df=Config.CHAR_MIN_DF,
            analyzer="char",
            strip_accents="unicode",
            sublinear_tf=True,
            use_idf=True,
        )

    def extract_features(self, train_df, val_df, test_df, load_cached_data=True):
        """
        Fits vectorizers on training data and transforms all datasets.
        Implements caching using scipy.sparse.save_npz.

        Args:
            train_df (pd.DataFrame): Training dataframe with 'comment_text'.
            val_df (pd.DataFrame): Validation dataframe with 'comment_text'.
            test_df (pd.DataFrame): Test dataframe with 'comment_text'.
            load_cached_data (bool): Whether to attempt loading from cache.

        Returns:
            tuple: (X_train, X_val, X_test) as sparse CSR matrices.
        """
        train_feat_path = os.path.join(self.cache_dir, "train_features.npz")
        val_feat_path = os.path.join(self.cache_dir, "val_features.npz")
        test_feat_path = os.path.join(self.cache_dir, "test_features.npz")

        # 1. Attempt to load from cache
        if load_cached_data:
            if (
                os.path.exists(train_feat_path)
                and os.path.exists(val_feat_path)
                and os.path.exists(test_feat_path)
            ):

                print(f"Loading features from cache: {self.cache_dir}")
                try:
                    X_train = scipy.sparse.load_npz(train_feat_path)
                    X_val = scipy.sparse.load_npz(val_feat_path)
                    X_test = scipy.sparse.load_npz(test_feat_path)
                    return X_train, X_val, X_test
                except Exception as e:
                    print(f"Cache load failed: {e}. Proceeding to recompute.")
            else:
                print("Cache not found. Proceeding to recompute.")

        # 2. Compute Features
        print("Starting TF-IDF Vectorization...")

        # Ensure text format and handle NaNs
        train_text = train_df["comment_text"].fillna("").astype(str)
        val_text = val_df["comment_text"].fillna("").astype(str)
        test_text = test_df["comment_text"].fillna("").astype(str)

        # Fit and Transform Word Vectorizer
        print(f"Fitting Word Vectorizer (Max Features: {Config.WORD_MAX_FEATURES})...")
        self.word_vectorizer.fit(train_text)
        X_train_word = self.word_vectorizer.transform(train_text)
        X_val_word = self.word_vectorizer.transform(val_text)
        X_test_word = self.word_vectorizer.transform(test_text)

        # Fit and Transform Char Vectorizer
        print(f"Fitting Char Vectorizer (Max Features: {Config.CHAR_MAX_FEATURES})...")
        self.char_vectorizer.fit(train_text)
        X_train_char = self.char_vectorizer.transform(train_text)
        X_val_char = self.char_vectorizer.transform(val_text)
        X_test_char = self.char_vectorizer.transform(test_text)

        # Stack Features
        print("Stacking Word and Char features...")
        # Use CSR format for efficient arithmetic operations in linear models
        X_train = scipy.sparse.hstack([X_train_word, X_train_char]).tocsr()
        X_val = scipy.sparse.hstack([X_val_word, X_val_char]).tocsr()
        X_test = scipy.sparse.hstack([X_test_word, X_test_char]).tocsr()

        # 3. Save to Cache
        # Only save if not in debug mode to preserve cache integrity
        if not Config.DEBUG:
            print(f"Saving features to cache: {self.cache_dir}")
            scipy.sparse.save_npz(train_feat_path, X_train)
            scipy.sparse.save_npz(val_feat_path, X_val)
            scipy.sparse.save_npz(test_feat_path, X_test)
        else:
            print("Debug mode active: Skipping cache save.")

        return X_train, X_val, X_test
