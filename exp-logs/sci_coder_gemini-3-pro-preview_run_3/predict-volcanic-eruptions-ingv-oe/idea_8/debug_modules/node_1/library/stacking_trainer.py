import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error

from library.config import (
    WORKING_DIR,
    SUBMISSION_PATH,
    SEED,
    N_FOLDS,
)
from library.model_zoo import (
    LGBMRegressorWrapper,
    XGBRegressorWrapper,
    HGBRegressorWrapper,
    RidgeMetaLearnerWrapper,
)


class StackingEnsemble:
    """
    Implements a Stacking Ensemble with Level 0 Base Learners and a Level 1 Meta Learner.
    """

    def __init__(self):
        # Registry of base model classes
        self.base_model_registry = {
            "lgbm": LGBMRegressorWrapper,
            "xgb": XGBRegressorWrapper,
            "hgb": HGBRegressorWrapper,
        }
        self.meta_model_cls = RidgeMetaLearnerWrapper

        # Containers for trained models
        self.trained_base_models = {}
        self.trained_meta_model = None

    def _get_stratified_folds(self, y, n_splits=N_FOLDS):
        """
        Generates stratified folds for continuous target by binning.
        """
        # Create bins for stratification (approx 20 bins or fewer if small dataset)
        num_bins = min(20, len(y) // n_splits)
        # Drop duplicates in case of low cardinality
        y_bins = pd.qcut(y, q=num_bins, labels=False, duplicates="drop")

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        # Split based on bins, return generator
        return skf.split(np.zeros(len(y)), y_bins)

    def train_level_0(self, X, y):
        """
        Performs Stratified K-Fold CV to train base learners and generate OOF predictions.
        """
        print(f"--- Level 0: Generating OOF Predictions ({N_FOLDS} Folds) ---")

        # Initialize OOF DataFrame
        oof_preds = pd.DataFrame(index=X.index, columns=self.base_model_registry.keys())
        metrics = {k: [] for k in self.base_model_registry.keys()}

        # Iterate through folds
        for fold, (train_idx, val_idx) in enumerate(self._get_stratified_folds(y)):
            # Split data
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            for name, model_cls in self.base_model_registry.items():
                # Instantiate fresh model
                model = model_cls()

                # Fit with validation set for early stopping (if supported by wrapper)
                model.fit(X_train, y_train, X_val, y_val)

                # Predict on validation fold
                pred = model.predict(X_val)

                # Store OOF prediction
                oof_preds.loc[val_idx, name] = pred

                # Calculate metric
                score = mean_absolute_error(y_val, pred)
                metrics[name].append(score)

            print(f"Fold {fold + 1}/{N_FOLDS} completed.")

        print("\nLevel 0 OOF MAE Summary:")
        for name, scores in metrics.items():
            print(f"{name}: {np.mean(scores)}")

        return oof_preds

    def train_level_1(self, oof_preds, y):
        """
        Trains the Meta Learner (Ridge) on OOF predictions.
        """
        print("\n--- Level 1: Training Meta Learner ---")
        self.trained_meta_model = self.meta_model_cls()
        self.trained_meta_model.fit(oof_preds, y)

        # Check performance on OOF set (proxy for generalization error)
        meta_preds = self.trained_meta_model.predict(oof_preds)
        score = mean_absolute_error(y, meta_preds)
        print(f"Meta Learner OOF MAE: {score}")

    def retrain_base_models(self, X, y):
        """
        Retrains all base learners on the full training dataset for final inference.
        """
        print("\n--- Retraining Base Learners on Full Data ---")
        for name, model_cls in self.base_model_registry.items():
            print(f"Retraining {name}...")
            model = model_cls()
            # Fit on full data (no validation set passed)
            model.fit(X, y)
            self.trained_base_models[name] = model

    def fit(self, X, y):
        """
        Orchestrates the full training pipeline.
        """
        # 1. Level 0: Generate OOF predictions via CV
        oof_preds = self.train_level_0(X, y)

        # 2. Level 1: Train Meta Learner on OOF predictions
        self.train_level_1(oof_preds, y)

        # 3. Retrain Base Learners on full data
        self.retrain_base_models(X, y)

        # 4. Save the ensemble
        self.save_model()

    def predict(self, X):
        """
        Generates final predictions using the retrained base models and meta learner.
        """
        # 1. Generate Base Predictions
        base_preds = pd.DataFrame(
            index=X.index, columns=self.base_model_registry.keys()
        )
        for name, model in self.trained_base_models.items():
            base_preds[name] = model.predict(X)

        # 2. Generate Meta Prediction
        final_pred = self.trained_meta_model.predict(base_preds)
        return final_pred

    def save_model(self):
        """
        Persists the trained ensemble to disk.
        """
        path = os.path.join(WORKING_DIR, "stacking_ensemble.joblib")
        joblib.dump(self, path)
        print(f"Ensemble saved to {path}")

    @staticmethod
    def load_model():
        """
        Loads the ensemble from disk.
        """
        path = os.path.join(WORKING_DIR, "stacking_ensemble.joblib")
        if os.path.exists(path):
            return joblib.load(path)
        return None


def generate_submission(model, X_test, segment_ids):
    """
    Generates predictions for the test set and saves the submission file.
    """
    print("\n--- Generating Submission ---")

    # Predict
    predictions = model.predict(X_test)

    # Create DataFrame
    submission_df = pd.DataFrame(
        {"segment_id": segment_ids, "time_to_eruption": predictions}
    )

    # Ensure directory exists
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

    # Save
    submission_df.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")
    print(submission_df.head())


def run_stacking_pipeline(X_train, y_train, X_test, test_ids):
    """
    Helper function to run the complete training and submission pipeline.
    """
    ensemble = StackingEnsemble()
    ensemble.fit(X_train, y_train)
    generate_submission(ensemble, X_test, test_ids)
