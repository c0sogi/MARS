import os
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss
from scipy.optimize import minimize_scalar
from library import config


class HybridEnsemble:
    """
    Implements the Multi-View Hybrid-Head Strong-Learner Ensemble strategy.

    This class manages two classification heads:
    1. Linear Head: LogisticRegressionCV (captures linear separability)
    2. Non-Linear Head: HistGradientBoostingClassifier (captures complex interactions)

    It optimizes a scalar weight to combine predictions from both heads to minimize Log Loss.
    """

    def __init__(self):
        # Initialize Linear Head with config parameters
        self.linear_head = LogisticRegressionCV(**config.LOGREG_PARAMS)

        # Initialize Non-Linear Head with config parameters
        self.nonlinear_head = HistGradientBoostingClassifier(**config.GB_PARAMS)

        # Initialize ensemble weight (default to equal weighting)
        self.w_linear = 0.5

        # Store class labels
        self.classes_ = None

    def fit(self, X_train, y_train):
        """
        Trains both classification heads on the provided training data.

        Args:
            X_train (np.ndarray): Training features.
            y_train (np.ndarray): Training labels.
        """
        print("Training Linear Head (LogisticRegressionCV)...")
        # LogisticRegressionCV automatically performs Cross-Validation to find best C
        self.linear_head.fit(X_train, y_train)

        print("Training Non-Linear Head (HistGradientBoostingClassifier)...")
        # HistGradientBoostingClassifier uses internal validation split (validation_fraction)
        # for early stopping as defined in config.GB_PARAMS.
        self.nonlinear_head.fit(X_train, y_train)

        # Store classes and ensure consistency
        self.classes_ = self.linear_head.classes_
        if not np.array_equal(self.classes_, self.nonlinear_head.classes_):
            raise ValueError(
                "Inconsistent classes between Linear and Non-Linear heads."
            )

        print("Training complete.")

    def optimize_weights(self, X_val, y_val):
        """
        Optimizes the ensemble weight on the validation set to minimize Log Loss.

        Args:
            X_val (np.ndarray): Validation features.
            y_val (np.ndarray): Validation labels.

        Returns:
            float: The optimized Log Loss on the validation set.
        """
        print("Optimizing Ensemble Weights on Validation Set...")

        # Get probabilities from both trained heads
        p_linear = self.linear_head.predict_proba(X_val)
        p_nonlinear = self.nonlinear_head.predict_proba(X_val)

        # Define the objective function (Log Loss)
        def objective(w):
            # w is the weight for the linear model
            # (1-w) is the weight for the non-linear model
            p_combined = w * p_linear + (1.0 - w) * p_nonlinear

            # Clip probabilities to prevent log(0)
            p_combined = np.clip(p_combined, 1e-15, 1 - 1e-15)

            # Normalize to ensure sum is exactly 1 (numerical stability)
            p_combined /= p_combined.sum(axis=1, keepdims=True)

            return log_loss(y_val, p_combined, labels=self.classes_)

        # Optimize w using bounded scalar minimization
        result = minimize_scalar(objective, bounds=(0.0, 1.0), method="bounded")

        self.w_linear = result.x
        val_loss = result.fun

        print(f"Optimization Complete.")
        print(f"Optimal Weight (Linear): {self.w_linear}")
        print(f"Optimal Weight (Non-Linear): {1.0 - self.w_linear}")
        print(f"Best Validation Log Loss: {val_loss}")

        return val_loss

    def predict_proba(self, X):
        """
        Predicts class probabilities for input X using the weighted ensemble.

        Args:
            X (np.ndarray): Input features.

        Returns:
            np.ndarray: Predicted probabilities (n_samples, n_classes).
        """
        p_linear = self.linear_head.predict_proba(X)
        p_nonlinear = self.nonlinear_head.predict_proba(X)

        # Weighted combination
        p_combined = self.w_linear * p_linear + (1.0 - self.w_linear) * p_nonlinear

        # Normalize
        p_combined /= p_combined.sum(axis=1, keepdims=True)

        return p_combined

    def save(self, directory):
        """
        Saves the trained models and ensemble metadata to the specified directory.

        Args:
            directory (str): Output directory path.
        """
        os.makedirs(directory, exist_ok=True)

        # Save sklearn models
        joblib.dump(self.linear_head, os.path.join(directory, "linear_head.joblib"))
        joblib.dump(
            self.nonlinear_head, os.path.join(directory, "nonlinear_head.joblib")
        )

        # Save metadata (weights and classes)
        meta = {"w_linear": self.w_linear, "classes": self.classes_}
        joblib.dump(meta, os.path.join(directory, "ensemble_meta.joblib"))
        print(f"Hybrid Ensemble saved to {directory}")

    def load(self, directory):
        """
        Loads the trained models and ensemble metadata from the specified directory.

        Args:
            directory (str): Input directory path.
        """
        self.linear_head = joblib.load(os.path.join(directory, "linear_head.joblib"))
        self.nonlinear_head = joblib.load(
            os.path.join(directory, "nonlinear_head.joblib")
        )

        meta = joblib.load(os.path.join(directory, "ensemble_meta.joblib"))
        self.w_linear = meta["w_linear"]
        self.classes_ = meta["classes"]
        print(f"Hybrid Ensemble loaded from {directory}")


def get_ordered_breeds():
    """
    Retrieves the sorted list of breed names from the training metadata.
    This ensures the columns in the submission file match the model's class indices.
    """
    df = pd.read_csv(config.TRAIN_METADATA_PATH)
    return sorted(df["breed"].unique())


def generate_submission(model, X_test, test_ids, output_path):
    """
    Generates predictions for the test set and saves them to a CSV file.

    Args:
        model (HybridEnsemble): The trained ensemble model.
        X_test (np.ndarray): Test features.
        test_ids (np.ndarray): Test image IDs.
        output_path (str): Path to save the submission CSV.
    """
    print(f"Generating predictions for {len(X_test)} test samples...")

    # Get probabilities
    preds = model.predict_proba(X_test)

    # Get column names (breeds)
    breed_names = get_ordered_breeds()

    # Verify shape consistency
    if preds.shape[1] != len(breed_names):
        raise ValueError(
            f"Model predicts {preds.shape[1]} classes, but found {len(breed_names)} breeds in metadata."
        )

    # Create DataFrame
    submission_df = pd.DataFrame(preds, columns=breed_names)

    # Insert ID column at the beginning
    submission_df.insert(0, "id", test_ids)

    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
