import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from library.config import Config


def train_meta_learner(oof_df, feature_cols, target_col="label"):
    """
    Trains the Level-2 Logistic Regression meta-learner using OOF predictions.

    Args:
        oof_df (pd.DataFrame): DataFrame containing OOF predictions from Level-1 models
                               and the ground truth labels.
        feature_cols (list): List of column names in oof_df to use as input features.
        target_col (str): Name of the target column.

    Returns:
        model: The trained scikit-learn model.
        float: The Log Loss score on the OOF data.
    """
    print(f"Training Meta-Learner on {len(oof_df)} samples.")
    print(f"Features: {feature_cols}")

    # Prepare X and y
    X = oof_df[feature_cols].values
    y = oof_df[target_col].values

    # Initialize Logistic Regression with Config parameters
    # We use the parameters specified in Config.META_LEARNER_PARAMS
    params = Config.META_LEARNER_PARAMS
    model = LogisticRegression(**params, random_state=Config.SEED)

    # Fit the model
    model.fit(X, y)

    # Predict probabilities on the training set (OOF predictions)
    # This serves as the validation score for the ensemble
    preds_proba = model.predict_proba(X)[:, 1]

    # Calculate Log Loss
    # We use the full precision for printing as requested
    loss = log_loss(y, preds_proba)
    print(f"Meta-Learner OOF Log Loss: {loss}")

    # Save the trained model
    save_path = os.path.join(Config.WORKING_DIR, "meta_learner.joblib")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(model, save_path)
    print(f"Meta-Learner model saved to {save_path}")

    return model, loss


def predict_meta_learner(test_df, feature_cols, model=None):
    """
    Generates calibrated predictions for the test set using the trained meta-learner.

    Args:
        test_df (pd.DataFrame): DataFrame containing test set predictions from Level-1 models.
        feature_cols (list): List of column names to use as input features.
        model (sklearn.base.BaseEstimator, optional): Pre-trained model instance.
                                                      If None, loads from disk.

    Returns:
        np.array: Calibrated probability predictions for the positive class (dog).
    """
    # Load model if not provided
    if model is None:
        load_path = os.path.join(Config.WORKING_DIR, "meta_learner.joblib")
        if not os.path.exists(load_path):
            raise FileNotFoundError(
                f"Meta-learner model not found at {load_path}. Train it first."
            )
        model = joblib.load(load_path)

    # Prepare X
    X_test = test_df[feature_cols].values

    # Predict probabilities (Class 1: Dog)
    preds = model.predict_proba(X_test)[:, 1]

    # Safety: Clip probabilities to prevent infinite Log Loss
    # Clipping to [1e-6, 1-1e-6]
    preds = np.clip(preds, 1e-6, 1 - 1e-6)

    return preds


def generate_submission(test_ids, predictions, output_path=Config.SUBMISSION_PATH):
    """
    Saves the final predictions to a CSV file in the required format.

    Args:
        test_ids (array-like): Sequence of image IDs.
        predictions (array-like): Sequence of probability predictions.
        output_path (str): Path to save the submission CSV.
    """
    if len(test_ids) != len(predictions):
        raise ValueError(
            f"Length mismatch: IDs ({len(test_ids)}) vs Predictions ({len(predictions)})"
        )

    submission_df = pd.DataFrame({"id": test_ids, "label": predictions})

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to CSV
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
