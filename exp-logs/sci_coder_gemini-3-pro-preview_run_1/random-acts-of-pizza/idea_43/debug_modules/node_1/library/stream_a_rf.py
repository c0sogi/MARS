import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from library.config import Config
from library.utils import set_seed
from library.feature_engineering import FeatureProcessor


class RandomForestPipeline:
    """
    Stream A: Consistency-Augmented Top-K Random Forest Pipeline.

    Combines:
    1. High-Fidelity TF-IDF (Title + Body)
    2. Full-Spectrum Metadata (Numerical + Ratios + Text Stats)
    3. Top-K Binary Community Indicators
    4. Global Consistency Scalars (Request-History Alignment)

    Trains a Random Forest with balanced class weights and low regularization.
    """

    def __init__(self):
        """
        Initialize the pipeline with configuration parameters.
        """
        set_seed(Config.SEED)
        self.model = RandomForestClassifier(**Config.RF_PARAMS)
        self.vectorizer = TfidfVectorizer(
            max_features=Config.TFIDF_MAX_FEATURES,
            stop_words="english",
            dtype=np.float32,
        )

    def _get_text_data(self, df: pd.DataFrame) -> list:
        """
        Concatenates title and body for TF-IDF vectorization.
        """
        # Fill NaNs with empty strings to ensure valid text input
        titles = df[Config.TEXT_COL_TITLE].fillna("").astype(str)
        bodies = df[Config.TEXT_COL_BODY].fillna("").astype(str)
        return (titles + " " + bodies).tolist()

    def _compute_tfidf(self, train_text: list, val_text: list, test_text: list):
        """
        Fits TF-IDF on training data and transforms all splits.
        Returns sparse matrices.
        """
        print("Fitting TF-IDF Vectorizer...")
        X_train = self.vectorizer.fit_transform(train_text)
        X_val = self.vectorizer.transform(val_text)
        X_test = self.vectorizer.transform(test_text)
        return X_train, X_val, X_test

    def get_features(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        load_cached_data: bool = True,
    ):
        """
        Orchestrates feature generation, assembly, and caching.

        Args:
            train_df, val_df, test_df: DataFrames for each split.
            load_cached_data: Whether to attempt loading from cache.

        Returns:
            Tuple of sparse matrices (X_train, X_val, X_test).
        """
        # Define cache paths for assembled sparse matrices
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        cache_path_train = os.path.join(Config.WORKING_DIR, "rf_features_train.npz")
        cache_path_val = os.path.join(Config.WORKING_DIR, "rf_features_val.npz")
        cache_path_test = os.path.join(Config.WORKING_DIR, "rf_features_test.npz")

        # Check if cache exists
        if (
            load_cached_data
            and os.path.exists(cache_path_train)
            and os.path.exists(cache_path_val)
            and os.path.exists(cache_path_test)
        ):
            print("Loading assembled RF features from cache...")
            X_train = sp.load_npz(cache_path_train)
            X_val = sp.load_npz(cache_path_val)
            X_test = sp.load_npz(cache_path_test)
            return X_train, X_val, X_test

        print("Computing and assembling RF features from scratch...")

        # 1. Retrieve Base Features (Metadata, Top-K, Consistency)
        # FeatureProcessor handles its own caching for these intermediate outputs
        fp = FeatureProcessor()
        base_features = fp.process_data(
            train_df, val_df, test_df, load_cached_data=load_cached_data
        )

        # 2. Compute TF-IDF Features
        train_text = self._get_text_data(train_df)
        val_text = self._get_text_data(val_df)
        test_text = self._get_text_data(test_df)

        tfidf_train, tfidf_val, tfidf_test = self._compute_tfidf(
            train_text, val_text, test_text
        )

        # 3. Assemble Features (Dense + Sparse)
        def assemble(split_name, tfidf_mat):
            data = base_features[split_name]

            # Extract dense components
            # X_meta: (N, 13) - Scaled metadata
            # X_topk: (N, 50) - Binary indicators
            # consistency: (N, 2) - Cosine scalars

            # Ensure all are 2D arrays
            meta = data["X_meta"]
            topk = data["X_topk"]
            consistency = data["consistency"]

            # Horizontal stack of dense features
            dense_block = np.hstack([meta, topk, consistency])

            # Convert dense block to sparse CSR
            dense_sparse = sp.csr_matrix(dense_block)

            # Stack with TF-IDF
            combined = sp.hstack([dense_sparse, tfidf_mat], format="csr")
            return combined

        X_train = assemble("train", tfidf_train)
        X_val = assemble("val", tfidf_val)
        X_test = assemble("test", tfidf_test)

        # 4. Save to Cache
        print("Saving assembled RF features to cache...")
        sp.save_npz(cache_path_train, X_train)
        sp.save_npz(cache_path_val, X_val)
        sp.save_npz(cache_path_test, X_test)

        return X_train, X_val, X_test

    def run(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        load_cached_data: bool = True,
    ):
        """
        Executes the full pipeline: Feature extraction -> Training -> Prediction.

        Returns:
            val_preds (np.ndarray): Probabilities for validation set.
            test_preds (np.ndarray): Probabilities for test set.
        """
        # 1. Prepare Features
        X_train, X_val, X_test = self.get_features(
            train_df, val_df, test_df, load_cached_data=load_cached_data
        )

        # 2. Prepare Targets
        y_train = train_df[Config.TARGET_COL].astype(int).values
        y_val = val_df[Config.TARGET_COL].astype(int).values

        # 3. Train Model
        print(f"Training Random Forest with {X_train.shape[1]} features...")
        self.model.fit(X_train, y_train)

        # 4. Validation
        print("Generating validation predictions...")
        val_preds = self.model.predict_proba(X_val)[:, 1]

        auc = roc_auc_score(y_val, val_preds)
        print(f"Random Forest Validation AUC: {auc}")

        # 5. Test Prediction
        print("Generating test predictions...")
        test_preds = self.model.predict_proba(X_test)[:, 1]

        return val_preds, test_preds
