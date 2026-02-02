import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from library.config import Config
from library.utils import load_metadata_splits
from library.feature_engineering import FeatureEngineer
from library.semantic_engine import SemanticProcessor


class RFManager:
    """
    Manages the 'Stream A' Random Forest pipeline.
    Handles feature assembly, model training, and inference.
    """

    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=Config.RF_N_ESTIMATORS,
            class_weight=Config.RF_CLASS_WEIGHT,
            min_samples_leaf=Config.RF_MIN_SAMPLES_LEAF,
            n_jobs=Config.RF_N_JOBS,
            random_state=Config.RANDOM_SEED,
            verbose=0,
        )
        self.feature_engineer = FeatureEngineer()
        self.semantic_processor = SemanticProcessor()

    def _get_assembled_features(
        self,
        df: pd.DataFrame,
        split_name: str,
        raw_json_path: str,
        load_cached_data: bool = True,
    ):
        """
        Retrieves features from processors, assembles them into a single matrix,
        and handles caching of the assembled matrix.

        Args:
            df (pd.DataFrame): Metadata dataframe.
            split_name (str): 'train', 'val', or 'test'.
            raw_json_path (str): Path to the raw JSON file.
            load_cached_data (bool): Whether to use cache.

        Returns:
            np.ndarray: Assembled feature matrix X.
        """
        # Define cache path for the assembled matrix
        cache_file = os.path.join(Config.IDEA_DIR, f"rf_assembled_{split_name}.npz")

        # 1. Try to load from cache
        if load_cached_data and os.path.exists(cache_file):
            print(f"Loading assembled RF features for '{split_name}' from {cache_file}")
            try:
                with np.load(cache_file) as data:
                    return data["X"]
            except Exception as e:
                print(f"Failed to load assembled cache: {e}. Recomputing...")

        # 2. Compute dependencies
        # We need semantic features first to generate interaction features
        sem_feats = self.semantic_processor.process_split(
            df, raw_json_path, split_name, load_cached_data
        )

        # Generate tabular features (includes TF-IDF, Top-K, Metadata, Interaction)
        tab_feats = self.feature_engineer.process_split(
            df, split_name, load_cached_data, semantic_features=sem_feats
        )

        # 3. Assemble Features
        # Components: metadata_rf, tfidf, top_k, interaction
        print(f"Assembling RF features for '{split_name}'...")

        X_parts = [
            tab_feats["metadata_rf"],
            tab_feats["tfidf"],
            tab_feats["top_k"],
            tab_feats["interaction"],
        ]

        # Concatenate horizontally
        X = np.hstack(X_parts).astype(np.float32)

        # 4. Save to cache
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        np.savez(cache_file, X=X)
        print(f"Saved assembled RF features to {cache_file}")

        return X

    def train(self, load_cached_data: bool = True):
        """
        Trains the Random Forest model and evaluates on the validation set.

        Args:
            load_cached_data (bool): Whether to use cached features.

        Returns:
            float: Validation ROC AUC score.
        """
        print("\n--- Starting Random Forest Training ---")

        # 1. Load Metadata
        train_df, val_df, _ = load_metadata_splits()

        # 2. Prepare Training Data
        X_train = self._get_assembled_features(
            train_df, "train", Config.TRAIN_JSON_PATH, load_cached_data
        )
        y_train = train_df[Config.TARGET_COL].values.astype(int)

        # 3. Prepare Validation Data
        X_val = self._get_assembled_features(
            val_df, "val", Config.TRAIN_JSON_PATH, load_cached_data
        )
        y_val = val_df[Config.TARGET_COL].values.astype(int)

        # 4. Train Model
        print(f"Training Random Forest with {Config.RF_N_ESTIMATORS} estimators...")
        self.model.fit(X_train, y_train)

        # 5. Evaluate
        print("Evaluating on Validation set...")
        # Predict probabilities for the positive class
        val_probs = self.model.predict_proba(X_val)[:, 1]

        auc_score = roc_auc_score(y_val, val_probs)
        print(f"RF Validation AUC: {auc_score}")

        return auc_score

    def predict_test(self, load_cached_data: bool = True):
        """
        Generates predictions for the test set.

        Args:
            load_cached_data (bool): Whether to use cached features.

        Returns:
            np.ndarray: Predicted probabilities for the test set.
        """
        print("\n--- Generating Random Forest Predictions for Test Set ---")

        # 1. Load Metadata
        _, _, test_df = load_metadata_splits()

        # 2. Prepare Test Data
        X_test = self._get_assembled_features(
            test_df, "test", Config.TEST_JSON_PATH, load_cached_data
        )

        # 3. Predict
        test_probs = self.model.predict_proba(X_test)[:, 1]

        return test_probs
