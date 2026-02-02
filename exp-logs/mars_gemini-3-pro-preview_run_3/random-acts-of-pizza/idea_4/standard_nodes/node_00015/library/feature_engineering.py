import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from library.config import Config
from library.utils import get_cached_data, load_data


class FeatureEngineer:
    """
    Handles feature engineering for the Tri-View Stacking Ensemble.
    Generates Lexical (TF-IDF), Style (Linguistic), and Metadata features.
    """

    def __init__(self):
        pass

    def _compute_lexical(self, train_df, val_df, test_df):
        """
        Generates TF-IDF features.
        Returns a dictionary of dense numpy arrays.
        """
        print("Computing Lexical Features (TF-IDF)...")
        vectorizer = TfidfVectorizer(
            max_features=Config.TFIDF_MAX_FEATURES,
            ngram_range=Config.TFIDF_NGRAM_RANGE,
            stop_words="english",
            sublinear_tf=True,
        )

        # Prepare text data
        train_text = train_df[Config.TEXT_COL].fillna("").astype(str)
        val_text = val_df[Config.TEXT_COL].fillna("").astype(str)
        test_text = test_df[Config.TEXT_COL].fillna("").astype(str)

        # Fit on Train
        vectorizer.fit(train_text)

        # Transform all and densify for storage/usage
        # Note: For very large datasets, sparse storage is preferred, but
        # with max_features=3000 and dataset size ~2k-4k, dense is manageable (~50MB).
        X_train = vectorizer.transform(train_text).toarray().astype(np.float32)
        X_val = vectorizer.transform(val_text).toarray().astype(np.float32)
        X_test = vectorizer.transform(test_text).toarray().astype(np.float32)

        return {"train": X_train, "val": X_val, "test": X_test}

    def _extract_style_features_from_df(self, df):
        """
        Helper to extract explicit linguistic features from a dataframe.
        """
        texts = df[Config.TEXT_COL].fillna("").astype(str)
        lower_texts = texts.str.lower()

        # 1. Structural Features
        char_len = texts.apply(len)
        word_len = texts.apply(lambda x: len(x.split()))
        # Avoid division by zero
        avg_word_len = char_len / (word_len + 1)

        # 2. Punctuation Features
        excl = texts.apply(lambda x: x.count("!"))
        quest = texts.apply(lambda x: x.count("?"))
        punct_density = (excl + quest) / (char_len + 1)

        # 3. Casing Features (Shouting)
        caps = texts.apply(lambda x: sum(1 for c in x if c.isupper()))
        caps_ratio = caps / (char_len + 1)

        # 4. Keyword Heuristics (Gratitude & Desperation)
        gratitude_words = ["thank", "appreciate", "grateful", "bless", "kindness"]
        desperation_words = [
            "help",
            "money",
            "broke",
            "hungry",
            "starving",
            "rent",
            "job",
            "lost",
            "fail",
            "desperate",
        ]

        gratitude = lower_texts.apply(
            lambda x: sum(x.count(w) for w in gratitude_words)
        )
        desperation = lower_texts.apply(
            lambda x: sum(x.count(w) for w in desperation_words)
        )

        # Stack features
        features = np.column_stack(
            [
                char_len,
                word_len,
                avg_word_len,
                excl,
                quest,
                punct_density,
                caps_ratio,
                gratitude,
                desperation,
            ]
        )
        return features.astype(np.float32)

    def _compute_style(self, train_df, val_df, test_df):
        """
        Generates explicit linguistic style features and scales them.
        """
        print("Computing Style Features...")
        X_train_raw = self._extract_style_features_from_df(train_df)
        X_val_raw = self._extract_style_features_from_df(val_df)
        X_test_raw = self._extract_style_features_from_df(test_df)

        # Scale features based on training statistics
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_val = scaler.transform(X_val_raw)
        X_test = scaler.transform(X_test_raw)

        return {"train": X_train, "val": X_val, "test": X_test}

    def _compute_meta(self, train_df, val_df, test_df):
        """
        Generates cleaned, imputed, and scaled metadata features.
        """
        print("Computing Metadata Features...")

        # Select safe numerical columns (excluding leakage and text)
        safe_cols = [
            "requester_account_age_in_days_at_request",
            "requester_days_since_first_post_on_raop_at_request",
            "requester_number_of_comments_at_request",
            "requester_number_of_comments_in_raop_at_request",
            "requester_number_of_posts_at_request",
            "requester_number_of_posts_on_raop_at_request",
            "requester_number_of_subreddits_at_request",
            "requester_upvotes_minus_downvotes_at_request",
            "requester_upvotes_plus_downvotes_at_request",
        ]

        # Extract values
        X_train_raw = train_df[safe_cols].values.astype(np.float32)
        X_val_raw = val_df[safe_cols].values.astype(np.float32)
        X_test_raw = test_df[safe_cols].values.astype(np.float32)

        # Impute missing values with median
        imputer = SimpleImputer(strategy="median")
        X_train_imp = imputer.fit_transform(X_train_raw)
        X_val_imp = imputer.transform(X_val_raw)
        X_test_imp = imputer.transform(X_test_raw)

        # Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_imp)
        X_val = scaler.transform(X_val_imp)
        X_test = scaler.transform(X_test_imp)

        return {"train": X_train, "val": X_val, "test": X_test}

    def _compute_targets(self, train_df, val_df):
        """
        Extracts target variables.
        """
        y_train = train_df[Config.TARGET_COL].values.astype(np.int32)
        y_val = val_df[Config.TARGET_COL].values.astype(np.int32)
        return {"train": y_train, "val": y_val}

    def process_data(self, load_cached_data=True, debug=False):
        """
        Main orchestration method to load data, compute features (with caching),
        and return a structured dictionary for training.

        Args:
            load_cached_data (bool): Whether to attempt loading from cache.
            debug (bool): Whether to use debug mode (smaller subset).

        Returns:
            dict: Nested dictionary containing 'train', 'val', 'test' splits with
                  'lexical', 'style', 'meta' features and 'y' targets.
        """
        # Load Raw Data
        train_df = load_data(Config.TRAIN_PATH, debug=debug)
        val_df = load_data(Config.VAL_PATH, debug=debug)
        test_df = load_data(Config.TEST_PATH, debug=debug)

        suffix = "_debug" if debug else ""

        # 1. Lexical Features (TF-IDF)
        lexical = get_cached_data(
            self._compute_lexical,
            f"lexical_features{suffix}",
            load_cached_data,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
        )

        # 2. Style Features (Linguistic)
        style = get_cached_data(
            self._compute_style,
            f"style_features{suffix}",
            load_cached_data,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
        )

        # 3. Metadata Features (Numerical)
        meta = get_cached_data(
            self._compute_meta,
            f"meta_features{suffix}",
            load_cached_data,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
        )

        # 4. Targets
        targets = get_cached_data(
            self._compute_targets,
            f"targets{suffix}",
            load_cached_data,
            train_df=train_df,
            val_df=val_df,
        )

        # 5. Test IDs (for submission)
        test_ids = test_df[Config.ID_COL].values

        return {
            "train": {
                "lexical": lexical["train"],
                "style": style["train"],
                "meta": meta["train"],
                "y": targets["train"],
            },
            "val": {
                "lexical": lexical["val"],
                "style": style["val"],
                "meta": meta["val"],
                "y": targets["val"],
            },
            "test": {
                "lexical": lexical["test"],
                "style": style["test"],
                "meta": meta["test"],
                "ids": test_ids,
            },
        }
