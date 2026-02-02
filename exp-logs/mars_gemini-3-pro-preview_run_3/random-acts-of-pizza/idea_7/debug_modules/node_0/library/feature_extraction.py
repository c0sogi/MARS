import os
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sentence_transformers import SentenceTransformer

from library.config import Config
from library.utils import setup_logger, set_seed
from library.data_loader import load_datasets

logger = setup_logger("feature_extraction")


class FeaturePipeline:
    """
    Manages the transformation of raw data into the three views (Lexical, Semantic, Behavioral)
    plus metadata for the Stacking Ensemble.
    """

    def __init__(self):
        self.lexical_vectorizer = TfidfVectorizer(**Config.LEXICAL_TFIDF_PARAMS)
        self.behavioral_vectorizer = TfidfVectorizer(**Config.BEHAVIORAL_TFIDF_PARAMS)
        # Median imputation for numerical features
        self.imputer = SimpleImputer(strategy="median")
        self.sbert_model = None

        # Define the columns we expect to output in metadata
        self.meta_feature_names = None

    def _get_sbert_model(self):
        if self.sbert_model is None:
            # Load model only when needed to save resources if using cache
            logger.info(f"Loading SBERT model: {Config.SBERT_MODEL_NAME}")
            self.sbert_model = SentenceTransformer(Config.SBERT_MODEL_NAME)
        return self.sbert_model

    def _prepare_behavioral_text(self, subreddits_series):
        """
        Converts a series of lists of subreddits into a series of space-separated strings.
        """
        return subreddits_series.apply(
            lambda x: " ".join(x) if isinstance(x, list) else ""
        )

    def _prepare_metadata(self, df):
        """
        Extracts numerical columns and generates temporal features.
        Returns a DataFrame ready for imputation.
        """
        # Select base numerical columns
        meta_df = df[Config.NUMERICAL_COLS].copy()

        # Temporal Feature Engineering
        # unix_timestamp_of_request is in Config.NUMERICAL_COLS, use it to derive others
        if "unix_timestamp_of_request" in meta_df.columns:
            dt_series = pd.to_datetime(meta_df["unix_timestamp_of_request"], unit="s")
            meta_df["request_hour"] = dt_series.dt.hour
            meta_df["request_day_of_week"] = dt_series.dt.dayofweek

        return meta_df

    def fit(self, df):
        """
        Fits the vectorizers and imputer on the training data.
        """
        logger.info("Fitting FeaturePipeline...")

        # 1. Fit Lexical (Text)
        text_data = df[Config.TEXT_COL].fillna("").astype(str)
        self.lexical_vectorizer.fit(text_data)

        # 2. Fit Behavioral (Subreddits)
        subreddit_data = self._prepare_behavioral_text(df[Config.SUBREDDIT_COL])
        self.behavioral_vectorizer.fit(subreddit_data)

        # 3. Fit Metadata (Imputer)
        meta_df = self._prepare_metadata(df)
        self.imputer.fit(meta_df)
        self.meta_feature_names = meta_df.columns.tolist()

        # Semantic model is pre-trained, no fitting required.
        logger.info("FeaturePipeline fitting complete.")

    def transform(self, df):
        """
        Transforms the dataframe into a dictionary of feature matrices.
        """
        logger.info("Transforming data...")

        # 1. Lexical Transformation
        text_data = df[Config.TEXT_COL].fillna("").astype(str)
        X_lexical = self.lexical_vectorizer.transform(text_data)

        # 2. Behavioral Transformation
        subreddit_data = self._prepare_behavioral_text(df[Config.SUBREDDIT_COL])
        X_behavioral = self.behavioral_vectorizer.transform(subreddit_data)

        # 3. Semantic Transformation (Dense Embeddings)
        model = self._get_sbert_model()
        # Encode returns a numpy array
        X_semantic = model.encode(
            text_data.tolist(),
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        # 4. Metadata Transformation
        meta_df = self._prepare_metadata(df)
        X_meta = self.imputer.transform(meta_df)

        return {
            "lexical": X_lexical,  # Sparse
            "behavioral": X_behavioral,  # Sparse
            "semantic": X_semantic,  # Dense
            "meta": X_meta,  # Dense
        }


def save_features(features_dict, prefix):
    """
    Saves the feature dictionary to the working directory using standardized filenames.
    """
    # Lexical (Sparse)
    lex_path = os.path.join(Config.WORKING_DIR, Config.CACHE_FILES[f"{prefix}_lexical"])
    sparse.save_npz(lex_path, features_dict["lexical"])

    # Behavioral (Sparse)
    beh_path = os.path.join(
        Config.WORKING_DIR, Config.CACHE_FILES[f"{prefix}_behavioral"]
    )
    sparse.save_npz(beh_path, features_dict["behavioral"])

    # Semantic (Dense)
    sem_path = os.path.join(
        Config.WORKING_DIR, Config.CACHE_FILES[f"{prefix}_semantic"]
    )
    np.save(sem_path, features_dict["semantic"])

    # Meta (Dense)
    meta_path = os.path.join(Config.WORKING_DIR, Config.CACHE_FILES[f"{prefix}_meta"])
    np.save(meta_path, features_dict["meta"])


def load_features(prefix):
    """
    Loads features from the working directory. Returns None if any file is missing.
    """
    try:
        lex_path = os.path.join(
            Config.WORKING_DIR, Config.CACHE_FILES[f"{prefix}_lexical"]
        )
        beh_path = os.path.join(
            Config.WORKING_DIR, Config.CACHE_FILES[f"{prefix}_behavioral"]
        )
        sem_path = os.path.join(
            Config.WORKING_DIR, Config.CACHE_FILES[f"{prefix}_semantic"]
        )
        meta_path = os.path.join(
            Config.WORKING_DIR, Config.CACHE_FILES[f"{prefix}_meta"]
        )

        if not all(
            os.path.exists(p) for p in [lex_path, beh_path, sem_path, meta_path]
        ):
            return None

        logger.info(f"Loading cached features for {prefix}...")
        return {
            "lexical": sparse.load_npz(lex_path),
            "behavioral": sparse.load_npz(beh_path),
            "semantic": np.load(sem_path),
            "meta": np.load(meta_path),
        }
    except Exception as e:
        logger.warning(f"Failed to load cache for {prefix}: {e}")
        return None


def extract_features(debug=False, load_cache=True):
    """
    Main entry point for feature extraction.

    Args:
        debug (bool): Whether to run in debug mode (smaller dataset).
        load_cache (bool): Whether to attempt loading from cache.

    Returns:
        (X_train_dict, y_train), (X_val_dict, y_val), (X_test_dict, test_ids)
    """
    set_seed(Config.SEED)

    # Check cache first (only if not debugging, as debug cache might be different)
    if load_cache and not debug:
        X_train = load_features("train")
        X_val = load_features("val")
        X_test = load_features("test")

        # Load targets and IDs
        y_train_path = os.path.join(
            Config.WORKING_DIR, Config.CACHE_FILES["train_target"]
        )
        y_val_path = os.path.join(Config.WORKING_DIR, Config.CACHE_FILES["val_target"])
        test_ids_path = os.path.join(Config.WORKING_DIR, Config.CACHE_FILES["test_ids"])

        targets_exist = all(
            os.path.exists(p) for p in [y_train_path, y_val_path, test_ids_path]
        )

        if X_train and X_val and X_test and targets_exist:
            logger.info("All features loaded from cache successfully.")
            y_train = np.load(y_train_path)
            y_val = np.load(y_val_path)
            test_ids = np.load(
                test_ids_path, allow_pickle=True
            )  # IDs are strings/objects
            return (X_train, y_train), (X_val, y_val), (X_test, test_ids)
        else:
            logger.info("Cache miss or incomplete. Reprocessing features...")

    # Load raw data
    (train_df, y_train), (val_df, y_val), (test_df, test_ids) = load_datasets(
        debug=debug
    )

    # Initialize and fit pipeline
    pipeline = FeaturePipeline()
    pipeline.fit(train_df)

    # Transform datasets
    X_train = pipeline.transform(train_df)
    X_val = pipeline.transform(val_df)
    X_test = pipeline.transform(test_df)

    # Save to cache (only if not debugging to avoid overwriting full cache with debug data)
    if not debug:
        logger.info("Saving features to cache...")
        save_features(X_train, "train")
        save_features(X_val, "val")
        save_features(X_test, "test")

        np.save(
            os.path.join(Config.WORKING_DIR, Config.CACHE_FILES["train_target"]),
            y_train.to_numpy(),
        )
        np.save(
            os.path.join(Config.WORKING_DIR, Config.CACHE_FILES["val_target"]),
            y_val.to_numpy(),
        )
        np.save(
            os.path.join(Config.WORKING_DIR, Config.CACHE_FILES["test_ids"]),
            test_ids.to_numpy(),
        )

    return (X_train, y_train), (X_val, y_val), (X_test, test_ids)
