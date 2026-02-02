import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from library.config import Config
from library.utils import seed_everything, calculate_log_loss


class StackingEnsemble:
    """
    Implements the Level 2 Meta-Learner for stacking.
    Aggregates predictions from base models using Logistic Regression.
    """

    def __init__(self):
        seed_everything(Config.SEED)
        # Meta-learner: Simple Logistic Regression
        # We use multinomial logistic regression to optimally weight the
        # class probabilities from the base models.
        self.meta_model = LogisticRegression(
            C=1.0,
            solver="lbfgs",
            multi_class="multinomial",
            random_state=Config.SEED,
            max_iter=1000,
        )
        self.model_keys = None  # To ensure consistent ordering of input features

    def fit(self, oof_preds_dict, y_true):
        """
        Trains the meta-learner on OOF predictions.

        Args:
            oof_preds_dict (dict): Dictionary where keys are model identifiers (e.g., 'nn', 'xgb')
                                   and values are numpy arrays of shape (n_samples, n_classes).
            y_true (np.ndarray): Ground truth labels (n_samples,).
        """
        # 1. Prepare Meta-Features
        # Sort keys to ensure consistent column order between fit and predict
        self.model_keys = sorted(oof_preds_dict.keys())

        # Stack predictions horizontally: (N, 3) -> (N, 3 * n_models)
        # If we have 4 models, this results in 12 features per sample
        X_meta = np.hstack([oof_preds_dict[k] for k in self.model_keys])

        print(f"Training Meta-Learner on input shape: {X_meta.shape}")

        # 2. Train
        self.meta_model.fit(X_meta, y_true)

        # 3. Evaluate on Training Data (OOF)
        # The loss on the OOF predictions represents the CV score of the ensemble
        meta_preds = self.meta_model.predict_proba(X_meta)
        loss = calculate_log_loss(y_true, meta_preds)

        print(f"Meta-Learner OOF Log Loss: {loss:.15f}")

        # Optional: Print coefficients to see which models are weighted higher
        # self.meta_model.coef_ shape is (n_classes, n_features)
        # print("Meta-Learner Coefficients shape:", self.meta_model.coef_.shape)

    def predict(self, test_preds_dict):
        """
        Generates final predictions using the trained meta-learner.

        Args:
            test_preds_dict (dict): Dictionary of test set predictions from base models.
                                    Keys must match those provided to fit().

        Returns:
            np.ndarray: Final probabilities (n_test_samples, n_classes).
        """
        if self.model_keys is None:
            raise ValueError("Model must be fitted before prediction.")

        # Ensure keys match exactly
        if set(test_preds_dict.keys()) != set(self.model_keys):
            raise ValueError(
                f"Mismatch in model keys. Expected {self.model_keys}, got {list(test_preds_dict.keys())}"
            )

        # Stack test predictions in the same order as training
        X_test_meta = np.hstack([test_preds_dict[k] for k in self.model_keys])

        # Predict
        final_probs = self.meta_model.predict_proba(X_test_meta)

        return final_probs

    def create_submission(
        self, test_ids, predictions, output_path=Config.SUBMISSION_PATH
    ):
        """
        Creates and saves the submission CSV file.

        Args:
            test_ids (list/array): IDs for the test set.
            predictions (np.ndarray): Predicted probabilities (n_test, 3).
            output_path (str): Path to save the CSV.
        """
        # Ensure predictions are clipped for safety (metric requirement)
        eps = 1e-15
        predictions = np.clip(predictions, eps, 1 - eps)

        # Normalize rows to sum to 1
        row_sums = predictions.sum(axis=1)
        # Avoid division by zero (unlikely with clip, but safe practice)
        row_sums[row_sums == 0] = 1.0
        predictions = predictions / row_sums[:, np.newaxis]

        # Create DataFrame
        # Assuming LabelEncoder order: 0=EAP, 1=HPL, 2=MWS (Alphabetical)
        df_sub = pd.DataFrame(
            {
                "id": test_ids,
                "EAP": predictions[:, 0],
                "HPL": predictions[:, 1],
                "MWS": predictions[:, 2],
            }
        )

        # Save
        df_sub.to_csv(output_path, index=False)
        print(f"Submission saved to {output_path}")
