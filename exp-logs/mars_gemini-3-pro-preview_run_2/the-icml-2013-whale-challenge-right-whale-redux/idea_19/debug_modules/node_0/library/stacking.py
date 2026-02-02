import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from library import config, utils


def format_features(pred_dict):
    """
    Formats the prediction dictionary into a dense feature matrix X.
    Enforces a deterministic column order based on config.MODEL_ARCHITECTURES
    and config.SAVE_METRICS to ensure alignment between training and inference.

    Args:
        pred_dict (dict): Dictionary where keys are '{arch}_{metric}' (e.g., 'resnet34_auc')
                          and values are 1D arrays or (N,1) arrays of probabilities.

    Returns:
        tuple: (X (np.ndarray), feature_names (list))
    """
    feature_names = []
    # Generate keys deterministically based on config
    for arch in config.MODEL_ARCHITECTURES:
        for metric in config.SAVE_METRICS:
            key = f"{arch}_{metric}"
            feature_names.append(key)

    # Collect columns corresponding to the generated keys
    cols = []
    for name in feature_names:
        if name not in pred_dict:
            raise KeyError(
                f"Expected prediction key '{name}' not found in input dictionary. "
                f"Available keys: {list(pred_dict.keys())}"
            )

        arr = np.array(pred_dict[name])
        # Flatten to ensure 1D shape
        arr = arr.ravel()
        cols.append(arr)

    # Stack columns to create matrix: (N_samples, N_features)
    X = np.column_stack(cols)
    return X, feature_names


def train_meta_learner(oof_preds, targets):
    """
    Trains a Logistic Regression meta-learner on Out-Of-Fold (OOF) predictions.
    Saves the learned coefficients and intercept to .npy files to avoid pickle usage.

    Args:
        oof_preds (dict): Dictionary of OOF predictions from base models.
        targets (array-like): Ground truth binary labels.

    Returns:
        tuple: (coef, intercept, auc_score)
    """
    print("Formatting OOF features for Meta-Learner...")
    X, feature_names = format_features(oof_preds)
    y = np.array(targets)

    print(f"Meta-Feature Matrix Shape: {X.shape}")

    # Initialize Logistic Regression
    # We use 'liblinear' solver which is efficient for small feature sets
    clf = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="liblinear",
        random_state=config.SEED,
        fit_intercept=True,
    )

    print("Fitting Meta-Learner...")
    clf.fit(X, y)

    # Extract parameters
    coef = clf.coef_[0]
    intercept = clf.intercept_[0]

    # Evaluate on OOF (Meta-Validation)
    # Calculate logits manually: Xw + b
    logits = np.dot(X, coef) + intercept
    # Sigmoid function: 1 / (1 + e^-z)
    probs = 1.0 / (1.0 + np.exp(-logits))

    auc = utils.calculate_roc_auc(y, probs)
    print(f"Meta-Learner OOF AUC: {auc}")

    print("Learned Coefficients:")
    for name, w in zip(feature_names, coef):
        print(f"  {name}: {w:.4f}")
    print(f"  Intercept: {intercept:.4f}")

    # Save parameters to .npy files
    os.makedirs(config.WORK_DIR, exist_ok=True)
    coef_path = os.path.join(config.WORK_DIR, "meta_learner_coef.npy")
    int_path = os.path.join(config.WORK_DIR, "meta_learner_intercept.npy")

    np.save(coef_path, coef)
    np.save(int_path, np.array([intercept]))
    print(f"Meta-learner parameters saved to {config.WORK_DIR}")

    return coef, intercept, auc


def load_meta_learner():
    """
    Loads the meta-learner parameters from the saved .npy files.

    Returns:
        tuple: (coef, intercept)
    """
    coef_path = os.path.join(config.WORK_DIR, "meta_learner_coef.npy")
    int_path = os.path.join(config.WORK_DIR, "meta_learner_intercept.npy")

    if not os.path.exists(coef_path) or not os.path.exists(int_path):
        raise FileNotFoundError(
            "Meta-learner parameters not found. "
            "Ensure train_meta_learner has been run first."
        )

    coef = np.load(coef_path)
    intercept = np.load(int_path)[0]  # Extract scalar from array

    return coef, intercept


def predict_stack(test_preds, coef, intercept):
    """
    Generates final predictions for the test set using the meta-learner parameters.

    Args:
        test_preds (dict): Dictionary of test set predictions from base models.
        coef (np.ndarray): Learned weights.
        intercept (float): Learned bias.

    Returns:
        np.ndarray: Final calibrated probabilities.
    """
    # Format test features exactly as training features
    X_test, _ = format_features(test_preds)

    # Compute Logits: z = w1*x1 + ... + wn*xn + b
    logits = np.dot(X_test, coef) + intercept

    # Apply Sigmoid Activation: P(y=1) = 1 / (1 + e^-z)
    probs = 1.0 / (1.0 + np.exp(-logits))

    return probs


def create_submission(probabilities, clips, output_path=config.SUBMISSION_PATH):
    """
    Creates and saves the submission CSV file.

    Args:
        probabilities (np.ndarray): Final predicted probabilities.
        clips (np.ndarray): Clip filenames corresponding to the predictions.
        output_path (str): Destination path for the CSV file.
    """
    # Ensure inputs are 1D arrays
    probabilities = np.array(probabilities).ravel()
    clips = np.array(clips).ravel()

    if len(probabilities) != len(clips):
        raise ValueError(
            f"Length mismatch: {len(probabilities)} probabilities vs {len(clips)} clips."
        )

    df = pd.DataFrame({"clip": clips, "probability": probabilities})

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df.to_csv(output_path, index=False)
    print(f"Submission file saved to {output_path}")
    print(f"Submission shape: {df.shape}")
