import os
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from library.config import Config
from library.engine import save_submission


def train_meta_learner(X_train, y_train, save_path=None, random_state=Config.SEED):
    """
    Trains the Level-1 Logistic Regression Meta-Learner on Out-Of-Fold (OOF) predictions.

    Args:
        X_train (np.ndarray): Input features (OOF predictions from base models).
                              Shape: (n_samples, n_base_models).
        y_train (np.ndarray): Target labels. Shape: (n_samples,).
        save_path (str, optional): Path to save the trained model (e.g., .pkl file).
        random_state (int): Seed for reproducibility.

    Returns:
        sklearn.linear_model.LogisticRegression: The trained meta-learner.
    """
    print(
        f"Training Meta-Learner on {len(y_train)} samples with {X_train.shape[1]} base models..."
    )

    # Initialize Logistic Regression
    # We use 'lbfgs' solver which is standard for this task.
    # No class_weight='balanced' is strictly necessary if the metric is AUC and
    # base models output probabilities, but the meta-learner will learn the bias.
    model = LogisticRegression(solver="lbfgs", random_state=random_state, n_jobs=-1)

    # Fit the model
    model.fit(X_train, y_train)

    # Evaluate on the training data (which represents the OOF validation set)
    # We use the probability of the positive class (1)
    preds = model.predict_proba(X_train)[:, 1]

    # Calculate AUC
    # Handle edge case where only one class is present (though unlikely in full train)
    if len(np.unique(y_train)) > 1:
        auc = roc_auc_score(y_train, preds)
    else:
        auc = 0.5

    print(f"Meta-Learner OOF AUC: {auc}")

    # Save the model if a path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(model, save_path)
        print(f"Meta-Learner saved to {save_path}")

    return model


def predict_meta_learner(model, X_test):
    """
    Generates predictions using the trained meta-learner.

    Args:
        model (sklearn.linear_model.LogisticRegression): The trained meta-learner.
        X_test (np.ndarray): Input features (Test predictions from base models).
                             Shape: (n_test_samples, n_base_models).

    Returns:
        np.ndarray: Array of shape (n_test_samples,) containing final probabilities.
    """
    # Predict probabilities for the positive class
    preds = model.predict_proba(X_test)[:, 1]
    return preds


def load_meta_learner(load_path):
    """
    Loads a trained meta-learner from disk.

    Args:
        load_path (str): Path to the saved model file.

    Returns:
        sklearn.linear_model.LogisticRegression: The loaded model.
    """
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Meta-learner model not found at {load_path}")

    model = joblib.load(load_path)
    return model


def generate_meta_submission(
    model, X_test, test_clips, output_path=Config.SUBMISSION_PATH
):
    """
    Generates predictions for the test set and saves them to a CSV file.

    Args:
        model (sklearn.linear_model.LogisticRegression): The trained meta-learner.
        X_test (np.ndarray): Test predictions from base models.
        test_clips (np.ndarray): Array of clip filenames corresponding to X_test.
        output_path (str): Path to save the submission CSV.
    """
    print("Generating Meta-Learner predictions for submission...")

    # Generate probabilities
    final_probs = predict_meta_learner(model, X_test)

    # Save using the engine utility
    save_submission(final_probs, test_clips, output_path)
