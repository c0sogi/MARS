import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from library.config import (
    ID_COL,
    TARGET_COL,
    SEED,
    N_FOLDS,
    NUM_CLASSES,
    L0_XGB_PARAMS,
    L0_NUM_BOOST_ROUND,
    L0_EARLY_STOPPING_ROUNDS,
    L1_XGB_PARAMS,
    L1_NUM_BOOST_ROUND,
    L1_EARLY_STOPPING_ROUNDS,
    WORKING_DIR,
    SUBMISSION_PATH,
)
from library.utils import set_seed, save_submission
from library.data_processing import load_and_process
from library.model_factory import XGBWrapper


class StackingManager:
    """
    Orchestrates the Two-Stage Homogeneous Stacking Ensemble.
    """

    def __init__(self):
        self.working_dir = WORKING_DIR
        os.makedirs(self.working_dir, exist_ok=True)

        # Define cache paths for Level 0 outputs
        self.l0_oof_path = os.path.join(self.working_dir, "level0_oof_preds.npy")
        self.l0_test_path = os.path.join(self.working_dir, "level0_test_preds.npy")

    def get_passthrough_features(self, df):
        """
        Extracts the geometric features to be used as passthrough features for Level 1.
        """
        # List of geometric features created in data_processing.py
        geo_features = [
            "Euclidean_Distance_To_Hydrology",
            "Elevation_Hydrology_Diff",
            "Aspect_Sin",
            "Aspect_Cos",
        ]

        # Select only those that exist in the dataframe
        selected_features = [col for col in geo_features if col in df.columns]
        return df[selected_features].values

    def run_level_0(self, train_df, test_df, load_cached_preds=True):
        """
        Executes Level 0: Base Learners (XGBoost).
        Performs Stratified K-Fold to generate OOF predictions for Train
        and averaged predictions for Test.
        """
        print("\n=== Running Level 0: Base Learners ===")

        # Check cache
        if (
            load_cached_preds
            and os.path.exists(self.l0_oof_path)
            and os.path.exists(self.l0_test_path)
        ):
            print(f"Loading cached Level 0 predictions from {self.working_dir}...")
            oof_preds = np.load(self.l0_oof_path)
            test_preds_avg = np.load(self.l0_test_path)
            return oof_preds, test_preds_avg

        print("Training Level 0 models from scratch...")

        # Prepare data
        # Drop ID and Target for X
        feature_cols = [c for c in train_df.columns if c not in [ID_COL, TARGET_COL]]
        X = train_df[feature_cols].values
        y = train_df[TARGET_COL].values

        # Test data
        X_test = test_df[feature_cols].values

        # Initialize containers
        # OOF preds: (n_samples, n_classes)
        oof_preds = np.zeros((len(train_df), NUM_CLASSES))
        # Test preds accumulator: (n_test_samples, n_classes)
        test_preds_sum = np.zeros((len(test_df), NUM_CLASSES))

        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            print(f"  L0 Fold {fold + 1}/{N_FOLDS}")

            X_train_fold, y_train_fold = X[train_idx], y[train_idx]
            X_val_fold, y_val_fold = X[val_idx], y[val_idx]

            # Initialize and Train Model
            model = XGBWrapper(
                params=L0_XGB_PARAMS,
                num_boost_round=L0_NUM_BOOST_ROUND,
                early_stopping_rounds=L0_EARLY_STOPPING_ROUNDS,
            )

            model.fit(X_train_fold, y_train_fold, X_val_fold, y_val_fold)

            # Predict OOF (Validation)
            val_probs = model.predict_proba(X_val_fold)
            oof_preds[val_idx] = val_probs

            # Predict Test
            test_probs = model.predict_proba(X_test)
            test_preds_sum += test_probs

        # Average test predictions
        test_preds_avg = test_preds_sum / N_FOLDS

        # Cache results
        print(f"Saving Level 0 predictions to {self.working_dir}...")
        np.save(self.l0_oof_path, oof_preds)
        np.save(self.l0_test_path, test_preds_avg)

        return oof_preds, test_preds_avg

    def prepare_meta_features(self, base_preds, df):
        """
        Concatenates base model probabilities with geometric passthrough features.

        Args:
            base_preds (np.ndarray): Class probabilities from Level 0 (N, n_classes).
            df (pd.DataFrame): The original dataframe containing geometric features.

        Returns:
            np.ndarray: Combined feature matrix for Level 1.
        """
        passthrough = self.get_passthrough_features(df)

        # Concatenate along the column axis
        # Shape: (N, n_classes + n_geometric_features)
        meta_features = np.hstack([base_preds, passthrough])

        return meta_features

    def run_level_1(self, oof_preds, test_preds_avg, train_df, test_df):
        """
        Executes Level 1: Meta Learners (XGBoost).
        Trains on Level 0 OOF + Geometric Features.
        """
        print("\n=== Running Level 1: Meta Learners ===")

        # Construct Meta-Datasets
        print("Constructing meta-features...")
        X_meta = self.prepare_meta_features(oof_preds, train_df)
        y = train_df[TARGET_COL].values

        X_meta_test = self.prepare_meta_features(test_preds_avg, test_df)

        # Initialize containers for final predictions
        # We will average the probabilities from the L1 folds
        final_test_preds_sum = np.zeros((len(test_df), NUM_CLASSES))

        # We can also track L1 CV performance
        l1_oof_preds = np.zeros((len(train_df), NUM_CLASSES))

        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_meta, y)):
            print(f"  L1 Fold {fold + 1}/{N_FOLDS}")

            X_train_fold, y_train_fold = X_meta[train_idx], y[train_idx]
            X_val_fold, y_val_fold = X_meta[val_idx], y[val_idx]

            # Initialize and Train Meta Model
            model = XGBWrapper(
                params=L1_XGB_PARAMS,
                num_boost_round=L1_NUM_BOOST_ROUND,
                early_stopping_rounds=L1_EARLY_STOPPING_ROUNDS,
            )

            model.fit(X_train_fold, y_train_fold, X_val_fold, y_val_fold)

            # Predict OOF (Validation) for scoring
            val_probs = model.predict_proba(X_val_fold)
            l1_oof_preds[val_idx] = val_probs

            # Predict Test
            test_probs = model.predict_proba(X_meta_test)
            final_test_preds_sum += test_probs

        # Compute L1 CV Accuracy
        # Argmax gives 0-based index, labels are 1-based
        l1_oof_labels = np.argmax(l1_oof_preds, axis=1) + 1
        l1_acc = np.mean(l1_oof_labels == y)
        print(f"Level 1 CV Accuracy: {l1_acc}")

        # Average test predictions
        final_test_probs = final_test_preds_sum / N_FOLDS

        # Convert to class labels (1-based)
        final_predictions = np.argmax(final_test_probs, axis=1) + 1

        return final_predictions

    def run(self, debug_sample_size=None, load_cached_data=True):
        """
        Main execution flow.
        """
        set_seed(SEED)

        # 1. Load and Process Data
        train_df, test_df = load_and_process(
            load_cached_data=load_cached_data, debug_sample_size=debug_sample_size
        )

        # 2. Level 0
        # Note: We pass load_cached_preds=load_cached_data to align caching behavior
        oof_preds, test_preds_avg = self.run_level_0(
            train_df, test_df, load_cached_preds=load_cached_data
        )

        # 3. Level 1
        final_predictions = self.run_level_1(
            oof_preds, test_preds_avg, train_df, test_df
        )

        # 4. Save Submission
        test_ids = test_df[ID_COL].values
        save_submission(final_predictions, test_ids, SUBMISSION_PATH)
