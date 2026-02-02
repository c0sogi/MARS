import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler
from scipy import sparse
import os

from library.config import Config
from library.utils import seed_everything, compute_log_loss, clip_probabilities
from library.data_loader import (
    load_raw_data,
    get_tfidf_features,
    get_stylometric_features,
)


class StatisticalModel:
    """
    The Statistical Branch of the ensemble (Static Anchor).
    Combines sparse TF-IDF features with dense stylometric features.
    Uses a weighted ensemble of Logistic Regression and Multinomial Naive Bayes.
    """

    def __init__(self):
        self.lr_model = LogisticRegression(
            C=1.0,
            solver="saga",
            multi_class="multinomial",
            max_iter=1000,
            random_state=Config.SEED,
            n_jobs=-1,
        )
        self.nb_model = MultinomialNB(alpha=0.01)
        self.scaler = MinMaxScaler()  # Ensures non-negative features for MNB
        self.best_weights = None  # Tuple (weight_lr, weight_nb)
        self.is_fitted = False

    def _prepare_features(self, tfidf_matrix, dense_features, fit_scaler=False):
        """
        Combines sparse TF-IDF and dense stylometric features.
        Scales dense features to [0,1] to be compatible with MultinomialNB.
        """
        if fit_scaler:
            dense_scaled = self.scaler.fit_transform(dense_features)
        else:
            dense_scaled = self.scaler.transform(dense_features)

        # Convert dense to sparse csr for efficient stacking
        dense_sparse = sparse.csr_matrix(dense_scaled)

        # Stack horizontally
        combined_features = sparse.hstack([tfidf_matrix, dense_sparse]).tocsr()
        return combined_features

    def fit(self, load_cached_data=True):
        """
        Trains the constituent models on the training set and optimizes
        ensemble weights on the validation set.
        """
        seed_everything()

        # 1. Load Data
        train_df, val_df, test_df = load_raw_data()
        y_train = train_df["author"].map(Config.LABEL2ID).values
        y_val = val_df["author"].map(Config.LABEL2ID).values

        # 2. Load/Compute Features
        # TF-IDF (Sparse) - Returns tuple (train, val, test)
        train_tfidf, val_tfidf, _ = get_tfidf_features(
            train_df, val_df, test_df, load_cached_data=load_cached_data
        )

        # Stylometric (Dense)
        train_dense = get_stylometric_features(
            train_df, "train", load_cached_data=load_cached_data
        )
        val_dense = get_stylometric_features(
            val_df, "val", load_cached_data=load_cached_data
        )

        # 3. Feature Preparation
        # Fit scaler on train, transform val
        X_train = self._prepare_features(train_tfidf, train_dense, fit_scaler=True)
        X_val = self._prepare_features(val_tfidf, val_dense, fit_scaler=False)

        # 4. Train Models
        print("Training Logistic Regression...")
        self.lr_model.fit(X_train, y_train)

        print("Training Multinomial Naive Bayes...")
        self.nb_model.fit(X_train, y_train)

        # 5. Optimize Weights on Validation Set
        print("Optimizing ensemble weights...")
        p_val_lr = self.lr_model.predict_proba(X_val)
        p_val_nb = self.nb_model.predict_proba(X_val)

        best_loss = float("inf")
        best_w = 0.5

        # Grid search for mixing weight w (LR weight)
        # w * LR + (1-w) * NB
        # Search range [0, 1] with step 0.01
        for w in np.linspace(0, 1, 101):
            p_blend = w * p_val_lr + (1 - w) * p_val_nb
            # compute_log_loss handles clipping and normalization internally
            loss = compute_log_loss(y_val, p_blend)
            if loss < best_loss:
                best_loss = loss
                best_w = w

        self.best_weights = (best_w, 1 - best_w)
        self.is_fitted = True

        print(f"Optimal Weights -> LR: {best_w:.2f}, NB: {1-best_w:.2f}")
        print(f"Validation Log Loss: {best_loss}")

        return best_loss

    def predict_proba(self, dataset_type="test", load_cached_data=True):
        """
        Generates probability predictions for the specified dataset.

        Args:
            dataset_type (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to use cached features.

        Returns:
            np.ndarray: Predicted probabilities (clipped).
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction.")

        # Load raw data to get the dataframes (needed for text extraction if not cached)
        train_df, val_df, test_df = load_raw_data()

        # Determine which dataframe to use
        if dataset_type == "train":
            target_df = train_df
        elif dataset_type == "val":
            target_df = val_df
        elif dataset_type == "test":
            target_df = test_df
        else:
            raise ValueError("dataset_type must be 'train', 'val', or 'test'")

        # Load Features
        # TF-IDF
        train_tfidf, val_tfidf, test_tfidf = get_tfidf_features(
            train_df, val_df, test_df, load_cached_data=load_cached_data
        )

        if dataset_type == "train":
            target_tfidf = train_tfidf
        elif dataset_type == "val":
            target_tfidf = val_tfidf
        else:
            target_tfidf = test_tfidf

        # Stylometric
        target_dense = get_stylometric_features(
            target_df, dataset_type, load_cached_data=load_cached_data
        )

        # Prepare Features (Transform only, scaler is already fitted)
        X_target = self._prepare_features(target_tfidf, target_dense, fit_scaler=False)

        # Predict
        p_lr = self.lr_model.predict_proba(X_target)
        p_nb = self.nb_model.predict_proba(X_target)

        # Blend
        w_lr, w_nb = self.best_weights
        p_blend = w_lr * p_lr + w_nb * p_nb

        # Clip (using utility function)
        return clip_probabilities(p_blend)
