import os
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from library.config import Config


class StackingEnsemble:
    """
    A Stacking Ensemble meta-learner using Logistic Regression.
    Aggregates predictions from multiple base models.
    """

    def __init__(self, random_state=42):
        # Using liblinear solver as it's often robust for binary classification on smaller feature sets
        self.model = LogisticRegression(random_state=random_state, solver="liblinear")
        self.feature_names = None
        self.random_state = random_state

    def fit(self, preds_dict, y_true):
        """
        Trains the meta-learner.

        Args:
            preds_dict: Dictionary {model_name: np.array of predictions}
            y_true: np.array of ground truth labels
        """
        # Sort keys to ensure consistent feature order between fit and predict
        self.feature_names = sorted(list(preds_dict.keys()))

        # Stack predictions column-wise to create feature matrix X
        X = np.column_stack([preds_dict[name] for name in self.feature_names])

        print(
            f"Training Meta-Learner on {X.shape[0]} samples with {X.shape[1]} base models..."
        )
        self.model.fit(X, y_true)

        # Print coefficients to see which models are contributing most
        print("Meta-Learner Coefficients:")
        for name, coef in zip(self.feature_names, self.model.coef_[0]):
            print(f"  {name}: {coef:.4f}")
        print(f"  Intercept: {self.model.intercept_[0]:.4f}")

        return self

    def predict(self, preds_dict):
        """
        Generates predictions using the trained meta-learner.

        Args:
            preds_dict: Dictionary {model_name: np.array of predictions}

        Returns:
            np.array of probabilities for the positive class
        """
        if self.feature_names is None:
            raise ValueError("Model has not been fitted yet.")

        # Validate that all expected features are present
        missing_features = [f for f in self.feature_names if f not in preds_dict]
        if missing_features:
            raise ValueError(f"Missing predictions for models: {missing_features}")

        # Stack predictions in the exact same order as training
        X = np.column_stack([preds_dict[name] for name in self.feature_names])

        # Return probability of class 1
        return self.model.predict_proba(X)[:, 1]


def fit_meta_learner(oof_preds_dict, y_true, save_path=None):
    """
    Trains the stacking ensemble on OOF predictions.

    Args:
        oof_preds_dict: Dict mapping model names to OOF prediction arrays.
        y_true: Ground truth labels corresponding to the OOF predictions.
        save_path: Optional path to save the trained meta-model.

    Returns:
        trained StackingEnsemble instance
    """
    # Initialize and train
    ensemble = StackingEnsemble(random_state=Config.SEED)
    ensemble.fit(oof_preds_dict, y_true)

    # Evaluate on the training set (OOF) to get an estimate of performance
    # Note: This is technically the training score for the meta-learner,
    # but since inputs are OOF, it's a good proxy for ensemble performance.
    meta_preds = ensemble.predict(oof_preds_dict)
    score = roc_auc_score(y_true, meta_preds)
    print(f"Meta-Learner OOF ROC AUC: {score}")  # Full precision printing

    # Save model if requested
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(ensemble, save_path)
        print(f"Meta-learner saved to {save_path}")

    return ensemble


def predict_ensemble(meta_model, test_preds_dict):
    """
    Generates final ensemble predictions for the test set.

    Args:
        meta_model: Trained StackingEnsemble instance (or path to one).
        test_preds_dict: Dict mapping model names to Test prediction arrays.

    Returns:
        np.array of final probabilities
    """
    # Load model if a path is provided
    if isinstance(meta_model, str):
        if os.path.exists(meta_model):
            print(f"Loading meta-learner from {meta_model}...")
            meta_model = joblib.load(meta_model)
        else:
            raise FileNotFoundError(f"Meta-model not found at {meta_model}")

    print("Generating ensemble predictions...")
    final_probs = meta_model.predict(test_preds_dict)
    return final_probs


def save_submission(ids, probs, output_path=None):
    """
    Saves the predictions to a CSV file in the required submission format.

    Args:
        ids: Array of image IDs.
        probs: Array of predicted probabilities.
        output_path: Path to save the CSV. If None, uses Config default.
    """
    if output_path is None:
        output_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df = pd.DataFrame({"id": ids, "has_cactus": probs})

    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
    print("First 5 rows:")
    print(df.head())
