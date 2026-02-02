import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from library.config import Config
from library.utils import save_submission, save_model, clip_probabilities


class MetaLearner:
    """
    Level-2 Stacking Meta-Learner.
    Wraps a Logistic Regression model to blend predictions from base models.
    """

    def __init__(self):
        self.model = LogisticRegression(
            C=1.0,
            solver="lbfgs",
            multi_class="multinomial",
            random_state=Config.SEED,
            max_iter=1000,
            n_jobs=-1,
        )

    def fit(self, X, y):
        """
        Fits the meta-learner on the meta-features.

        Args:
            X (np.ndarray): Meta-features (concatenated base model probs).
            y (np.ndarray): True labels.
        """
        self.model.fit(X, y)

    def predict_proba(self, X):
        """
        Predicts probabilities using the meta-learner.

        Args:
            X (np.ndarray): Meta-features.

        Returns:
            np.ndarray: Predicted probabilities.
        """
        return self.model.predict_proba(X)


def prepare_meta_features(preds_dict):
    """
    Concatenates predictions from multiple models into a single feature matrix.
    Ensures consistent ordering of columns based on model names.

    Args:
        preds_dict (dict): Dictionary where keys are model names and values are
                           probability arrays (N_samples, N_classes).

    Returns:
        np.ndarray: Concatenated feature matrix (N_samples, N_models * N_classes).
    """
    # Sort keys to ensure deterministic feature order
    model_names = sorted(preds_dict.keys())

    # Collect arrays in order
    arrays = []
    for name in model_names:
        preds = preds_dict[name]
        # Ensure it's a numpy array
        if not isinstance(preds, np.ndarray):
            preds = np.array(preds)
        arrays.append(preds)

    # Stack horizontally
    meta_features = np.hstack(arrays)
    return meta_features


def run_ensemble(oof_preds, test_preds, y_true, test_ids):
    """
    Orchestrates the ensemble process:
    1. Prepares meta-features.
    2. Trains the MetaLearner.
    3. Evaluates on OOF data.
    4. Generates and saves test submission.

    Args:
        oof_preds (dict): Dictionary of OOF predictions from base models.
        test_preds (dict): Dictionary of Test predictions from base models.
        y_true (np.ndarray): True integer labels for the training set.
        test_ids (list or np.ndarray): IDs for the test set rows.

    Returns:
        np.ndarray: Final blended test probabilities.
    """
    print("Preparing meta-features for ensemble...")
    X_train_meta = prepare_meta_features(oof_preds)
    X_test_meta = prepare_meta_features(test_preds)

    print(f"Meta-features shape (Train): {X_train_meta.shape}")
    print(f"Meta-features shape (Test): {X_test_meta.shape}")

    # Initialize Meta-Learner
    meta_learner = MetaLearner()

    print("Training Meta-Learner (Logistic Regression)...")
    meta_learner.fit(X_train_meta, y_true)

    # Evaluate on Training Data (OOF)
    # This gives an estimate of the ensemble's performance
    train_probs = meta_learner.predict_proba(X_train_meta)

    # Clip probabilities for metric calculation consistency
    train_probs_clipped = clip_probabilities(train_probs)

    ensemble_score = log_loss(y_true, train_probs_clipped)
    print(f"Ensemble OOF Log Loss: {ensemble_score}")

    # Generate Test Predictions
    print("Generating predictions for Test set...")
    final_test_probs = meta_learner.predict_proba(X_test_meta)

    # Save the Meta-Learner model
    save_model(meta_learner.model, "meta_learner_lr.joblib", model_type="sklearn")

    # Save Submission
    print(f"Saving submission to {Config.SUBMISSION_DIR}/submission.csv...")
    save_submission(test_ids, final_test_probs, filename="submission.csv")

    return final_test_probs
