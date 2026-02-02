import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from library.config import Config
from library.utils import seed_everything, calculate_roc_auc


class MetaLearner:
    """
    A wrapper around Logistic Regression for stacking base model predictions.
    """

    def __init__(self, random_state=42):
        self.model = LogisticRegression(
            solver="liblinear",
            penalty="l2",
            C=1.0,
            random_state=random_state,
            fit_intercept=True,
        )
        self.feature_names = []

    def fit(self, X, y):
        """
        Fits the logistic regression model.
        """
        self.model.fit(X, y)

    def predict(self, X):
        """
        Predicts probabilities for the positive class.
        """
        return self.model.predict_proba(X)[:, 1]

    def get_coefficients(self):
        """
        Returns the learned coefficients and intercept.
        """
        return self.model.coef_[0], self.model.intercept_[0]


def prepare_meta_features(preds_dict):
    """
    Converts a dictionary of predictions into a feature matrix.
    Ensures consistent ordering of columns based on sorted model names.

    Args:
        preds_dict (dict): Dictionary where keys are model names and values are
                           numpy arrays of probabilities (N,) or (N, 1).

    Returns:
        tuple: (X, model_names) where X is the (N, M) feature matrix and
               model_names is the list of M model names in order.
    """
    # Sort keys to ensure consistent column order between train and test
    model_names = sorted(preds_dict.keys())

    features = []
    for name in model_names:
        pred = preds_dict[name]
        # Ensure shape is (N, 1)
        if pred.ndim == 1:
            pred = pred.reshape(-1, 1)
        features.append(pred)

    # Stack horizontally: (N, M)
    X = np.hstack(features)
    return X, model_names


def train_meta_learner(oof_preds_dict, targets):
    """
    Trains the meta-learner on OOF predictions.

    Args:
        oof_preds_dict (dict): Dictionary mapping model names to OOF probability arrays.
        targets (array-like): Ground truth labels corresponding to the OOF predictions.

    Returns:
        MetaLearner: The trained meta-learner instance.
        float: The ROC AUC score on the OOF set.
    """
    seed_everything(Config.SEED)

    # Prepare features
    X, model_names = prepare_meta_features(oof_preds_dict)

    # Initialize and train
    meta_learner = MetaLearner(random_state=Config.SEED)
    meta_learner.feature_names = model_names
    meta_learner.fit(X, targets)

    # Evaluate on OOF data
    oof_preds = meta_learner.predict(X)
    auc = calculate_roc_auc(targets, oof_preds)

    print("Meta-Learner Training Complete.")
    print(f"Input Models: {model_names}")
    coefs, intercept = meta_learner.get_coefficients()
    print(f"Coefficients: {coefs}")
    print(f"Intercept: {intercept}")
    print(f"Stacking OOF AUC: {auc}")

    return meta_learner, auc


def generate_submission(
    meta_learner, test_preds_dict, test_ids, output_path=Config.SUBMISSION_PATH
):
    """
    Generates predictions for the test set and saves the submission file.

    Args:
        meta_learner (MetaLearner): Trained meta-learner.
        test_preds_dict (dict): Dictionary mapping model names to Test probability arrays.
        test_ids (array-like): IDs for the test images.
        output_path (str): Path to save the submission CSV.

    Returns:
        pd.DataFrame: The submission dataframe.
    """
    # Prepare features
    X_test, model_names = prepare_meta_features(test_preds_dict)

    # Verify model names match the training phase
    if model_names != meta_learner.feature_names:
        raise ValueError(
            f"Feature mismatch! Train models: {meta_learner.feature_names}, "
            f"Test models: {model_names}"
        )

    # Predict
    final_preds = meta_learner.predict(X_test)

    # Create DataFrame
    df = pd.DataFrame({"id": test_ids, "has_cactus": final_preds})

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")

    return df
