import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from library import config, data_loader, feature_engineering, text_processing


class RandomForestStream:
    """
    Implements Stream A: Multi-Objective Target-Encoded Random Forest.

    This class manages:
    1. Loading and combining engineered tabular features and TF-IDF text features.
    2. Imputing missing values using a median strategy.
    3. Caching the prepared training/validation/test matrices.
    4. Training a Random Forest Classifier with balanced class weights.
    5. Evaluating performance and generating predictions.
    """

    def __init__(self):
        # Initialize model with parameters from config
        self.model = RandomForestClassifier(**config.RF_PARAMS)
        # Initialize imputer (Median strategy as per requirements)
        self.imputer = SimpleImputer(strategy="median")

        # Caching paths
        self.cache_dir = config.WORKING_DIR
        self.cache_path_X = os.path.join(self.cache_dir, "rf_X_data.npz")
        self.cache_path_y = os.path.join(self.cache_dir, "rf_y_data.npz")

    def _prepare_data(self, load_cached_data=True):
        """
        Loads features, concatenates them, handles imputation, and manages caching.

        Args:
            load_cached_data (bool): Whether to attempt loading from cache.

        Returns:
            tuple: (X_train, X_val, X_test, y_train, y_val) as numpy arrays.
        """
        # 1. Try Loading from Cache
        if load_cached_data:
            if os.path.exists(self.cache_path_X) and os.path.exists(self.cache_path_y):
                print("Loading RF data from cache...")
                try:
                    data_X = np.load(self.cache_path_X)
                    data_y = np.load(self.cache_path_y)
                    return (
                        data_X["train"],
                        data_X["val"],
                        data_X["test"],
                        data_y["train"],
                        data_y["val"],
                    )
                except Exception as e:
                    print(f"Failed to load RF cache: {e}. Regenerating...")

        print("Preparing RF data from scratch...")

        # 2. Load Raw Data (for Targets)
        # Using data_loader to ensure consistent splits
        df_train, df_val, df_test = data_loader.load_tabular_data(
            load_cached_data=load_cached_data
        )

        # Extract targets
        if config.TARGET_COL in df_train.columns:
            y_train = df_train[config.TARGET_COL].values.astype(int)
        else:
            raise ValueError(
                f"Target column '{config.TARGET_COL}' not found in training data."
            )

        if config.TARGET_COL in df_val.columns:
            y_val = df_val[config.TARGET_COL].values.astype(int)
        else:
            raise ValueError(
                f"Target column '{config.TARGET_COL}' not found in validation data."
            )

        # 3. Load Tabular Features (Metadata + Target Encoding)
        # generate_features handles its own caching
        X_tab_train, X_tab_val, X_tab_test = feature_engineering.generate_features(
            load_cached_data=load_cached_data
        )

        # 4. Load Text Features (TF-IDF)
        # TfidfPipeline handles its own caching
        tfidf_pipeline = text_processing.TfidfPipeline()
        feats_text = tfidf_pipeline.run(
            df_train, df_val, df_test, load_cached_data=load_cached_data
        )

        # 5. Concatenate Features
        print("Concatenating tabular and text features...")
        # Ensure tabular data is float32 to match TF-IDF and save memory
        X_train = np.hstack(
            [X_tab_train.values.astype(np.float32), feats_text["train"]]
        )
        X_val = np.hstack([X_tab_val.values.astype(np.float32), feats_text["val"]])
        X_test = np.hstack([X_tab_test.values.astype(np.float32), feats_text["test"]])

        # 6. Impute Missing Values
        print("Imputing missing values (Median)...")
        # Fit on training data only to prevent leakage
        X_train = self.imputer.fit_transform(X_train)
        X_val = self.imputer.transform(X_val)
        X_test = self.imputer.transform(X_test)

        # 7. Save to Cache
        print(f"Saving RF data to {self.cache_dir}...")
        os.makedirs(self.cache_dir, exist_ok=True)
        np.savez(self.cache_path_X, train=X_train, val=X_val, test=X_test)
        np.savez(self.cache_path_y, train=y_train, val=y_val)

        return X_train, X_val, X_test, y_train, y_val

    def run(self, load_cached_data=True):
        """
        Executes the Random Forest pipeline.

        Args:
            load_cached_data (bool): Whether to use cached data.

        Returns:
            tuple: (val_probs, test_probs, model)
        """
        # Prepare Data
        X_train, X_val, X_test, y_train, y_val = self._prepare_data(
            load_cached_data=load_cached_data
        )

        # Train Model
        print(f"Training Random Forest with params: {config.RF_PARAMS}")
        self.model.fit(X_train, y_train)

        # Evaluate on Validation Set
        print("Evaluating on Validation set...")
        val_probs = self.model.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, val_probs)

        # Print metric with full precision
        print(f"Random Forest Validation AUC: {val_auc}")

        # Generate Test Predictions
        print("Generating Test predictions...")
        test_probs = self.model.predict_proba(X_test)[:, 1]

        return val_probs, test_probs, self.model
