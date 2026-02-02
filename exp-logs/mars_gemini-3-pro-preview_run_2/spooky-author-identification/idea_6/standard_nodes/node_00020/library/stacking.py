import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from library.config import Config
from library.utils import seed_everything, calculate_metric


class StackingEnsemble:
    """
    Level-2 Meta-Learner for Stacking.
    Aggregates predictions from base models and trains a Logistic Regression
    to generate final probabilities.
    """

    def __init__(self):
        """
        Initializes the Stacking Ensemble with a Logistic Regression meta-learner.
        """
        seed_everything(Config.SEED)
        self.meta_model = LogisticRegression(
            C=1.0,
            solver="lbfgs",
            multi_class="multinomial",
            random_state=Config.SEED,
            n_jobs=Config.NUM_WORKERS,
            max_iter=1000,
        )
        self.model_names = []  # Stores the order of models for consistency

    def _prepare_meta_features(self, preds_dict, is_train=True):
        """
        Concatenates probability vectors from multiple models into a single feature matrix.

        Args:
            preds_dict (dict): Dictionary where keys are model names and values are
                               numpy arrays of shape (n_samples, n_classes).
            is_train (bool): If True, defines the order of models based on keys.
                             If False, enforces the order defined during training.

        Returns:
            np.ndarray: Concatenated feature matrix of shape (n_samples, n_models * n_classes).
        """
        if is_train:
            # Sort keys to ensure deterministic order
            self.model_names = sorted(list(preds_dict.keys()))

        feature_list = []
        for name in self.model_names:
            if name not in preds_dict:
                raise ValueError(f"Model '{name}' missing from predictions dictionary.")
            feature_list.append(preds_dict[name])

        # Horizontal concatenation: [Model1_ProbA, Model1_ProbB, ..., Model2_ProbA, ...]
        X_meta = np.hstack(feature_list)
        return X_meta

    def fit_predict(self, oof_preds_dict, test_preds_dict, y_train, test_ids):
        """
        Trains the meta-learner on OOF predictions and generates final test predictions.

        Args:
            oof_preds_dict (dict): Dictionary of OOF predictions (Train/Val set).
            test_preds_dict (dict): Dictionary of Test predictions.
            y_train (np.ndarray): Ground truth labels for the training set (aligned with OOF).
            test_ids (list or np.ndarray): IDs corresponding to the test set rows.

        Returns:
            np.ndarray: Final predicted probabilities for the test set.
        """
        print("Preparing meta-features for stacking...")
        X_meta_train = self._prepare_meta_features(oof_preds_dict, is_train=True)
        X_meta_test = self._prepare_meta_features(test_preds_dict, is_train=False)

        print(
            f"Meta-Feature Shape (Train): {X_meta_train.shape}, (Test): {X_meta_test.shape}"
        )

        # Train Meta-Learner
        print("Training Meta-Learner (Logistic Regression)...")
        self.meta_model.fit(X_meta_train, y_train)

        # Evaluate on Training set (OOF) to check stacking performance
        meta_train_preds = self.meta_model.predict_proba(X_meta_train)
        score = calculate_metric(y_train, meta_train_preds)
        print(f"Stacking Ensemble OOF Log Loss: {score}")

        # Predict on Test set
        print("Generating final test predictions...")
        final_test_preds = self.meta_model.predict_proba(X_meta_test)

        # Generate Submission
        self._save_submission(test_ids, final_test_preds)

        return final_test_preds

    def _save_submission(self, test_ids, preds):
        """
        Formats and saves the submission file.
        Applies clipping as required by the metric specification.

        Args:
            test_ids (list): List of ID strings.
            preds (np.ndarray): Predicted probabilities (n_samples, 3).
        """
        # Ensure output directory exists
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

        # Clip probabilities to avoid log loss extremes: max(min(p, 1-1e-15), 1e-15)
        eps = 1e-15
        preds = np.clip(preds, eps, 1 - eps)

        # Create DataFrame
        # Column order must match Config.CLASSES which corresponds to the model output order
        # Config.CLASSES = ["EAP", "HPL", "MWS"]
        submission_df = pd.DataFrame(preds, columns=Config.CLASSES)
        submission_df.insert(0, "id", test_ids)

        # Save
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
        print("Submission Head:")
        print(submission_df.head())
