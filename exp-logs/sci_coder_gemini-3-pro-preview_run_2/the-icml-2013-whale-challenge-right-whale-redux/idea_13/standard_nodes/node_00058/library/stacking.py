import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from library.config import Config
from library.utils import seed_everything


def load_ground_truth():
    """
    Loads and concatenates the ground truth labels from train and val metadata.
    Respects the Config.DEBUG flag to slice the data identically to the data loader.

    Returns:
        np.ndarray: Array of binary labels (0 or 1).
    """
    # Load metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)

    # Concatenate to match the full training set used for Cross-Validation
    # Order must match data.py: train then val
    full_df = pd.concat([train_df, val_df], ignore_index=True)

    # Handle Debug Mode
    if Config.DEBUG:
        full_df = full_df.iloc[: Config.DEBUG_SIZE]
        print(f"DEBUG MODE: Loaded {len(full_df)} ground truth labels.")

    return full_df["label"].values.astype(float)


def prepare_meta_features(predictions_dict):
    """
    Converts a dictionary of predictions into a feature matrix for the meta-learner.

    Args:
        predictions_dict (dict): Dictionary where keys are model names and values
                                 are 1D numpy arrays of probabilities.

    Returns:
        np.ndarray: Feature matrix of shape (n_samples, n_models).
    """
    # Sort keys to ensure consistent column order between OOF (Train) and Test
    model_names = sorted(predictions_dict.keys())

    feature_list = []
    for name in model_names:
        preds = predictions_dict[name]

        # Ensure flat array
        if isinstance(preds, pd.Series):
            preds = preds.values
        if preds.ndim > 1:
            preds = preds.flatten()

        feature_list.append(preds)

    # Stack columns: (N, M)
    X = np.column_stack(feature_list)
    return X


class StackingMetaLearner:
    """
    Level-1 Meta-Learner using Logistic Regression.
    Aggregates predictions from base learners to produce a final calibrated probability.
    """

    def __init__(self):
        seed_everything(Config.SEED)
        self.model = LogisticRegression(
            solver=Config.META_SOLVER,
            C=Config.META_C,
            random_state=Config.SEED,
            fit_intercept=True,
        )
        self.is_fitted = False

    def fit(self, X_oof, y_true):
        """
        Trains the meta-learner on OOF predictions.

        Args:
            X_oof (np.ndarray): OOF prediction matrix (N_samples, N_models).
            y_true (np.ndarray): Ground truth labels.
        """
        print(
            f"Training Meta-Learner on {len(y_true)} samples with {X_oof.shape[1]} base models..."
        )

        self.model.fit(X_oof, y_true)
        self.is_fitted = True

        # Evaluate on the training set (OOF) to check calibration/performance
        # Note: Since these are OOF predictions, this metric is a valid estimate of generalization
        probs = self.model.predict_proba(X_oof)[:, 1]
        auc = roc_auc_score(y_true, probs)

        print("Meta-Learner Training Completed.")
        print(f"Stacked OOF AUC: {auc}")
        print(f"Learned Coefficients: {self.model.coef_}")
        print(f"Intercept: {self.model.intercept_}")

        return auc

    def predict(self, X_test):
        """
        Generates final probabilities for the test set.

        Args:
            X_test (np.ndarray): Test prediction matrix (N_test_samples, N_models).

        Returns:
            np.ndarray: Final probabilities.
        """
        if not self.is_fitted:
            raise RuntimeError("Meta-Learner must be fitted before prediction.")

        return self.model.predict_proba(X_test)[:, 1]


def generate_submission(test_probs):
    """
    Creates and saves the submission file.

    Args:
        test_probs (np.ndarray): Final predicted probabilities for the test set.
    """
    # Load test metadata to get clip names
    test_df = pd.read_csv(Config.TEST_CSV)

    # Handle Debug Mode for test set
    if Config.DEBUG:
        test_df = test_df.iloc[: Config.DEBUG_SIZE]

    # Validation
    if len(test_df) != len(test_probs):
        raise ValueError(
            f"Shape mismatch: Test metadata has {len(test_df)} rows, "
            f"but predictions have {len(test_probs)}."
        )

    # Create DataFrame
    submission = pd.DataFrame({"clip": test_df["clip"], "probability": test_probs})

    # Save
    output_path = Config.SUBMISSION_PATH
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission.to_csv(output_path, index=False)

    print(f"Submission saved to {output_path}")
    print("Head of submission:")
    print(submission.head())
