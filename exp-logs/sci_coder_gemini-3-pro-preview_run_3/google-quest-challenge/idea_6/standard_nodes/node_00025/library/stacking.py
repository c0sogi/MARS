import os
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import RidgeCV
from library.config import Config
from library.utils import compute_spearmanr


class StackingTrainer:
    """
    Manages the Level 1 and Level 2 stacking training processes.
    Implements Ridge Regression for both base layer (per fold) and meta layer.
    """

    def __init__(self):
        self.working_dir = Config.WORKING_DIR
        os.makedirs(self.working_dir, exist_ok=True)

    def train_l1_ridge(
        self, backbone_name, features, targets, test_features, folds, load_cached=True
    ):
        """
        Trains L1 Ridge models for a specific backbone using Cross-Validation.

        Args:
            backbone_name (str): Name of the backbone (e.g., 'deberta').
            features (np.ndarray): Training features of shape (N_train, Feature_Dim).
            targets (np.ndarray): Training targets of shape (N_train, 30).
            test_features (np.ndarray): Test features of shape (N_test, Feature_Dim).
            folds (np.ndarray): Fold indices for training data of shape (N_train,).
            load_cached (bool): Whether to load cached predictions if available.

        Returns:
            dict: Contains 'oof_preds' (N_train, 30) and 'test_preds' (N_test, 30).
        """
        cache_path = os.path.join(self.working_dir, f"{backbone_name}_l1_preds.joblib")

        if load_cached and os.path.exists(cache_path):
            print(
                f"Loading cached L1 predictions for {backbone_name} from {cache_path}"
            )
            return joblib.load(cache_path)

        print(f"Training L1 Ridge for {backbone_name}...")

        n_samples = features.shape[0]
        n_targets = targets.shape[1]
        n_test = test_features.shape[0]
        n_folds = Config.N_FOLDS

        oof_preds = np.zeros((n_samples, n_targets))
        test_preds_accum = np.zeros((n_test, n_targets))

        # Alphas for RidgeCV to search over
        alphas = [0.1, 1.0, 10.0, 100.0]

        for fold in range(n_folds):
            # Boolean indexing for current fold
            val_mask = folds == fold
            train_mask = ~val_mask

            X_train = features[train_mask]
            y_train = targets[train_mask]
            X_val = features[val_mask]
            y_val = targets[val_mask]

            # Train RidgeCV (automatically selects best alpha via LOOCV)
            # RidgeCV handles multi-output regression efficiently
            model = RidgeCV(alphas=alphas)
            model.fit(X_train, y_train)

            # Predict Validation (OOF)
            val_preds = model.predict(X_val)
            oof_preds[val_mask] = val_preds

            # Predict Test
            test_preds_accum += model.predict(test_features)

            # Log Score for this fold
            fold_score = compute_spearmanr(y_val, val_preds)
            print(f"  {backbone_name} | Fold {fold} | Val Spearman: {fold_score}")

        # Average Test Predictions across all folds
        avg_test_preds = test_preds_accum / n_folds

        # Calculate Overall OOF Score
        overall_score = compute_spearmanr(targets, oof_preds)
        print(f"  {backbone_name} | Overall OOF Spearman: {overall_score}")

        result = {"oof_preds": oof_preds, "test_preds": avg_test_preds}

        # Cache results
        joblib.dump(result, cache_path)
        print(f"Saved L1 predictions to {cache_path}")

        return result

    def train_l2_meta(self, l1_outputs, targets, load_cached=True):
        """
        Trains the Level 2 Meta-Learner (Ridge) on concatenated OOF predictions.

        Args:
            l1_outputs (dict): Dictionary mapping backbone names to their L1 output dicts.
                               Example: {'deberta': {'oof_preds': ..., 'test_preds': ...}, ...}
            targets (np.ndarray): Ground truth targets (N_train, 30).
            load_cached (bool): Whether to load cached predictions.

        Returns:
            np.ndarray: Final test predictions (N_test, 30).
        """
        cache_path = os.path.join(self.working_dir, "l2_meta_preds.joblib")

        if load_cached and os.path.exists(cache_path):
            print(f"Loading cached L2 predictions from {cache_path}")
            return joblib.load(cache_path)

        print("Training Level 2 Meta-Learner...")

        # Sort keys to ensure deterministic order of concatenation
        backbone_names = sorted(l1_outputs.keys())

        X_meta_train_list = []
        X_meta_test_list = []

        for name in backbone_names:
            X_meta_train_list.append(l1_outputs[name]["oof_preds"])
            X_meta_test_list.append(l1_outputs[name]["test_preds"])

        # Concatenate predictions from all backbones
        # Shape: (N_train, 30 * Num_Backbones)
        X_meta_train = np.hstack(X_meta_train_list)
        # Shape: (N_test, 30 * Num_Backbones)
        X_meta_test = np.hstack(X_meta_test_list)

        print(f"  Meta-Train Shape: {X_meta_train.shape}")
        print(f"  Meta-Test Shape:  {X_meta_test.shape}")

        # Train Meta Ridge
        # We use a slightly broader regularization search space for the meta-learner
        alphas = [0.1, 1.0, 10.0, 100.0, 1000.0]
        meta_model = RidgeCV(alphas=alphas)
        meta_model.fit(X_meta_train, targets)

        # Predict on Test
        final_preds = meta_model.predict(X_meta_test)

        # Clip Predictions to valid range [0, 1]
        final_preds = np.clip(final_preds, 0.0, 1.0)

        # Evaluate OOF performance of the Meta-Learner
        oof_meta_preds = meta_model.predict(X_meta_train)
        oof_meta_preds = np.clip(oof_meta_preds, 0.0, 1.0)
        score = compute_spearmanr(targets, oof_meta_preds)
        print(f"  Meta-Learner OOF Spearman: {score}")

        # Cache
        joblib.dump(final_preds, cache_path)
        print(f"Saved L2 predictions to {cache_path}")

        return final_preds

    def save_submission(self, predictions, test_ids):
        """
        Saves the final predictions to the submission file.

        Args:
            predictions (np.ndarray): Final predicted probabilities (N_test, 30).
            test_ids (np.ndarray or list): QA IDs for the test set.
        """
        if len(predictions) != len(test_ids):
            raise ValueError(
                f"Length mismatch: preds {len(predictions)} vs ids {len(test_ids)}"
            )

        df = pd.DataFrame(predictions, columns=Config.TARGET_COLS)
        df.insert(0, "qa_id", test_ids)

        df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
