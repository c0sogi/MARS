import os
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from library.config import Config
from library.utils import setup_logger, calculate_auc, set_seed


class StackingMetaLearner:
    """
    Implements the Stacking Meta-Learner using Logistic Regression.
    Aggregates predictions from base models to produce a final probability.
    """

    def __init__(self, random_state=Config.SEED):
        """
        Initialize the meta-learner.

        Args:
            random_state (int): Seed for reproducibility.
        """
        self.logger = setup_logger("StackingMetaLearner")
        self.random_state = random_state
        set_seed(self.random_state)

        # Initialize Logistic Regression
        # liblinear is a good choice for binary classification on smaller datasets (meta-features)
        self.model = LogisticRegression(
            random_state=self.random_state, solver="liblinear", C=1.0, class_weight=None
        )
        self.feature_names = None

    def prepare_features(self, preds_dict):
        """
        Converts a dictionary of predictions into a feature matrix.
        Ensures consistent ordering of columns based on model names.

        Args:
            preds_dict (dict): Dictionary {model_name: np.array of probabilities}.
                               Arrays should be 1D or 2D (N, 1).

        Returns:
            np.array: Feature matrix of shape (N, n_models).
        """
        if not preds_dict:
            raise ValueError("Prediction dictionary is empty.")

        # Sort keys to ensure consistent column order between train and test phases
        sorted_keys = sorted(preds_dict.keys())

        # Validation of feature consistency
        if self.feature_names is None:
            self.feature_names = sorted_keys
        else:
            if sorted_keys != self.feature_names:
                self.logger.warning(
                    f"Feature mismatch! Training features: {self.feature_names}, "
                    f"Inference features: {sorted_keys}"
                )
                # Strictly enforce that the set of models matches
                if set(sorted_keys) != set(self.feature_names):
                    raise ValueError(
                        "Mismatch in model names between training and inference."
                    )

        # Stack features column-wise
        features = []
        for key in sorted_keys:
            arr = np.array(preds_dict[key])
            if arr.ndim > 1:
                arr = arr.flatten()
            features.append(arr)

        X = np.column_stack(features)
        return X

    def train(self, oof_preds_dict, y_true, save_path=None):
        """
        Trains the meta-learner on Out-Of-Fold predictions.

        Args:
            oof_preds_dict (dict): Dictionary of OOF predictions {model_name: probabilities}.
            y_true (array-like): Ground truth labels.
            save_path (str, optional): Path to save the trained model (uses pickle).

        Returns:
            float: The AUC score on the training set (OOF).
        """
        self.logger.info("Preparing meta-features for training...")
        # Reset feature names for a fresh training session
        self.feature_names = None

        X = self.prepare_features(oof_preds_dict)
        y = np.array(y_true)

        self.logger.info(
            f"Training Logistic Regression on feature matrix shape: {X.shape}"
        )
        self.model.fit(X, y)

        # Evaluate on OOF data (Self-evaluation of the ensemble on training data)
        preds = self.model.predict_proba(X)[:, 1]
        auc = calculate_auc(y, preds)

        self.logger.info(f"Meta-Learner OOF AUC: {auc}")

        # Print coefficients to understand model contribution
        coefs = self.model.coef_[0]
        coef_msg = ", ".join(
            [f"{name}: {coef:.4f}" for name, coef in zip(self.feature_names, coefs)]
        )
        self.logger.info(f"Model Coefficients: {coef_msg}")
        self.logger.info(f"Intercept: {self.model.intercept_[0]:.4f}")

        if save_path:
            try:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, "wb") as f:
                    pickle.dump(self.model, f)
                self.logger.info(f"Meta-learner model saved to {save_path}")
            except Exception as e:
                self.logger.error(f"Failed to save model: {e}")

        return auc

    def predict(self, test_preds_dict):
        """
        Generates predictions using the trained meta-learner.

        Args:
            test_preds_dict (dict): Dictionary of test predictions {model_name: probabilities}.

        Returns:
            np.array: Final probabilities.
        """
        self.logger.info("Preparing meta-features for inference...")
        X = self.prepare_features(test_preds_dict)

        # Predict probabilities for class 1
        preds = self.model.predict_proba(X)[:, 1]
        return preds

    def load(self, model_path):
        """
        Loads a trained model from disk.

        Args:
            model_path (str): Path to the pickle file.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

        self.logger.info(f"Model loaded from {model_path}")

    def generate_submission(self, test_ids, preds, output_path=Config.SUBMISSION_PATH):
        """
        Saves the final predictions to a CSV file.

        Args:
            test_ids (array-like): IDs of the test images.
            preds (array-like): Predicted probabilities.
            output_path (str): Path to save the submission file.
        """
        if len(test_ids) != len(preds):
            raise ValueError(
                f"Shape mismatch: {len(test_ids)} IDs vs {len(preds)} predictions."
            )

        df = pd.DataFrame({"id": test_ids, "has_cactus": preds})

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        self.logger.info(f"Submission saved to {output_path}")
