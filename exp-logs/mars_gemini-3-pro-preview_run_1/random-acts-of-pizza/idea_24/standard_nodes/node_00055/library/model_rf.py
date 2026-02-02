import os
import numpy as np
import pandas as pd
import scipy.sparse
from sklearn.ensemble import RandomForestClassifier
from library.config import RF_PARAMS, WORKING_DIR, TARGET_COL
from library.utils import calculate_auc, ensure_directory
from library.feature_engineering import (
    TextProcessor,
    MetadataExtractor,
    BayesianHistoryEncoder,
)


class RFModel:
    """
    Stream A: Bayesian Target-Encoded Random Forest.
    Wraps the feature engineering and Random Forest classification pipeline.
    """

    def __init__(self):
        self.model = RandomForestClassifier(**RF_PARAMS)
        self.text_processor = TextProcessor()
        self.metadata_extractor = MetadataExtractor()
        self.bayes_encoder = BayesianHistoryEncoder()

    def _get_features(
        self,
        df: pd.DataFrame,
        split_name: str,
        is_training: bool = False,
        load_cached_data: bool = True,
    ) -> scipy.sparse.csr_matrix:
        """
        Orchestrates feature generation, assembly, and caching.
        """
        # 0. Check Assembled Cache First (Cite debug_lesson_1)
        cache_path = os.path.join(
            WORKING_DIR, f"rf_features_assembled_{split_name}.npz"
        )

        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading assembled features from {cache_path}")
            X = scipy.sparse.load_npz(cache_path)
            if X.shape[0] == len(df):
                return X
            else:
                print(
                    f"Cache dimension mismatch ({X.shape[0]} vs {len(df)}). Discarding cache."
                )
                load_cached_data = False

        # 1. Text Features (TF-IDF)
        # We must call the processor methods to ensure internal state (vectorizer) is fitted/loaded
        if is_training:
            text_features = self.text_processor.fit_transform(
                df, split_name, load_cached_data
            )
        else:
            text_features = self.text_processor.transform(
                df, split_name, load_cached_data
            )

        # 2. Metadata Features (Dense -> Sparse)
        meta_df = self.metadata_extractor.process(df, split_name, load_cached_data)
        # Convert to sparse matrix to allow efficient stacking with TF-IDF
        meta_features = scipy.sparse.csr_matrix(meta_df.values.astype(float))

        # 3. Bayesian History Features (Dense -> Sparse)
        if is_training:
            # fit_transform_cv generates OOF features for training and fits global stats
            hist_df = self.bayes_encoder.fit_transform_train(
                df, split_name, load_cached_data
            )
        else:
            # transform uses global stats
            hist_df = self.bayes_encoder.transform(df, split_name, load_cached_data)
        hist_features = scipy.sparse.csr_matrix(hist_df.values.astype(float))

        # 4. Assembly
        print(f"Assembling features for {split_name}...")
        # Horizontal stack of all feature matrices
        X = scipy.sparse.hstack(
            [text_features, meta_features, hist_features], format="csr"
        )

        # Cache the assembled matrix
        ensure_directory(cache_path)
        scipy.sparse.save_npz(cache_path, X)
        print(f"Saved assembled features to {cache_path}")

        return X

    def train(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        load_cached_data: bool = True,
    ):
        """
        Trains the Random Forest model and evaluates on the validation set.
        """
        print("Preparing Training Data...")
        X_train = self._get_features(
            train_df, "train", is_training=True, load_cached_data=load_cached_data
        )
        y_train = train_df[TARGET_COL].values

        print("Preparing Validation Data...")
        X_val = self._get_features(
            val_df, "val", is_training=False, load_cached_data=load_cached_data
        )
        y_val = val_df[TARGET_COL].values

        print(f"Training Random Forest with params: {RF_PARAMS}")
        self.model.fit(X_train, y_train)

        print("Evaluating on Validation Set...")
        # Get probabilities for the positive class (1)
        y_pred_val = self.model.predict_proba(X_val)[:, 1]

        auc = calculate_auc(y_val, y_pred_val)
        print(f"Validation AUC: {auc}")

        return auc

    def predict_proba(
        self, test_df: pd.DataFrame, load_cached_data: bool = True
    ) -> np.ndarray:
        """
        Generates predictions for the test set.
        """
        print("Preparing Test Data...")
        X_test = self._get_features(
            test_df, "test", is_training=False, load_cached_data=load_cached_data
        )

        print("Generating Predictions...")
        y_pred = self.model.predict_proba(X_test)[:, 1]

        return y_pred
