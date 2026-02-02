import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from library.config import Config
from library.utils import seed_everything

# Ensure reproducibility
seed_everything(Config.SEED)


def sigmoid(x):
    """Numerically stable sigmoid function."""
    return 1 / (1 + np.exp(-x))


def prepare_meta_features(predictions_dict, file_name_prefix, load_cached_data=True):
    """
    Constructs the feature matrix X for the meta-learner from a dictionary of predictions.

    Args:
        predictions_dict (dict): Dictionary {model_name: np.array(predictions)}.
        file_name_prefix (str): Prefix for cache files (e.g., 'oof' or 'test').
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (X (np.ndarray), feature_names (np.ndarray))
    """
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    cache_path_X = os.path.join(cache_dir, f"{file_name_prefix}_meta_features.npy")
    cache_path_cols = os.path.join(cache_dir, f"{file_name_prefix}_meta_cols.npy")

    # 1. Try Load from Cache
    if (
        load_cached_data
        and os.path.exists(cache_path_X)
        and os.path.exists(cache_path_cols)
    ):
        print(f"Loading cached meta-features from {cache_path_X}")
        X = np.load(cache_path_X)
        # Allow pickle is required for string arrays in numpy
        feature_names = np.load(cache_path_cols, allow_pickle=True)

        # Validation: check if cached features match current dictionary keys
        # If the dictionary has different keys than cached, we must recompute
        current_keys = sorted(predictions_dict.keys())
        if list(feature_names) != current_keys:
            print(
                "Warning: Cached feature names do not match provided prediction keys. Recomputing..."
            )
        else:
            return X, feature_names

    # 2. Compute from Scratch
    print(f"Constructing meta-features for {file_name_prefix}...")

    # Sort keys to ensure deterministic column order
    model_names = sorted(predictions_dict.keys())

    features_list = []
    for name in model_names:
        preds = predictions_dict[name]
        # Ensure shape is (N, 1)
        if preds.ndim == 1:
            preds = preds.reshape(-1, 1)
        features_list.append(preds)

    if not features_list:
        raise ValueError("No predictions provided to create meta-features.")

    X = np.hstack(features_list).astype(np.float32)
    feature_names = np.array(model_names)

    # 3. Save to Cache
    print(f"Saving meta-features to {cache_path_X}...")
    np.save(cache_path_X, X)
    np.save(cache_path_cols, feature_names)

    return X, feature_names


def train_meta_learner(oof_preds_dict, y_true, load_cached_data=True):
    """
    Trains a Logistic Regression meta-learner on OOF predictions.
    Saves coefficients and intercept to .npy files (avoiding pickle).

    Args:
        oof_preds_dict (dict): Dictionary of OOF predictions {model_name: pred_array}.
        y_true (np.array): Ground truth labels.
        load_cached_data (bool): Whether to use cached feature matrices.

    Returns:
        dict: Dictionary containing 'coef' and 'intercept'.
    """
    print("\n[Meta-Learner] Training...")

    # Prepare Feature Matrix
    X, feature_names = prepare_meta_features(
        oof_preds_dict, "oof", load_cached_data=load_cached_data
    )

    print(f"Meta-Learner Input Shape: {X.shape}")
    print(f"Features: {feature_names}")

    # Initialize Logistic Regression
    # Using 'liblinear' for robust binary classification on small-medium datasets
    clf = LogisticRegression(
        random_state=Config.SEED, solver="liblinear", C=1.0, penalty="l2"
    )

    # Cite debug_lesson_12: Strictly Segregate Hold-Out Data When Training Stacked Meta-Learners
    # Perform Cross-Validation to generate unbiased OOF predictions for the meta-learner
    print("Performing Cross-Validation for Meta-Learner evaluation...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=Config.SEED)
    oof_preds = np.zeros(len(X))

    for train_idx, val_idx in skf.split(X, y_true):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y_true[train_idx], y_true[val_idx]

        fold_clf = LogisticRegression(
            random_state=Config.SEED, solver="liblinear", C=1.0, penalty="l2"
        )
        fold_clf.fit(X_train, y_train)
        oof_preds[val_idx] = fold_clf.predict_proba(X_val)[:, 1]

    auc_score = roc_auc_score(y_true, oof_preds)
    print(f"Meta-Learner CV AUC: {auc_score}")

    # Retrain on full data for export (Test set inference)
    clf.fit(X, y_true)

    # Save Model Parameters (Coefficients & Intercept)
    # We avoid pickling the whole model object to be strictly compliant with data processing rules
    # and to ensure portability.
    coef_path = os.path.join(Config.WORKING_DIR, "meta_learner_coef.npy")
    intercept_path = os.path.join(Config.WORKING_DIR, "meta_learner_intercept.npy")

    np.save(coef_path, clf.coef_)
    np.save(intercept_path, clf.intercept_)
    print(f"Meta-learner weights saved to {Config.WORKING_DIR}")

    return {
        "coef": clf.coef_,
        "intercept": clf.intercept_,
        "auc": auc_score,
        "oof_preds": oof_preds,
    }


def predict_meta_learner(test_preds_dict, load_cached_data=True):
    """
    Generates predictions using the saved meta-learner weights.

    Args:
        test_preds_dict (dict): Dictionary of Test predictions {model_name: pred_array}.
        load_cached_data (bool): Whether to use cached feature matrices.

    Returns:
        np.array: Final probabilities.
    """
    print("\n[Meta-Learner] Predicting...")

    # Prepare Feature Matrix
    X, feature_names = prepare_meta_features(
        test_preds_dict, "test", load_cached_data=load_cached_data
    )

    # Load Model Parameters
    coef_path = os.path.join(Config.WORKING_DIR, "meta_learner_coef.npy")
    intercept_path = os.path.join(Config.WORKING_DIR, "meta_learner_intercept.npy")

    if not (os.path.exists(coef_path) and os.path.exists(intercept_path)):
        raise FileNotFoundError(
            f"Meta-learner weights not found in {Config.WORKING_DIR}. Train first."
        )

    coef = np.load(coef_path)  # Shape: (1, n_features)
    intercept = np.load(intercept_path)  # Shape: (1,)

    # Manual Linear Projection & Sigmoid
    # z = X @ coef.T + intercept
    # X: (N, F), coef.T: (F, 1) -> z: (N, 1)
    z = np.dot(X, coef.T) + intercept
    probs = sigmoid(z).flatten()

    return probs


def create_submission_file(
    clip_names, probabilities, output_path=Config.SUBMISSION_PATH
):
    """
    Creates and saves the submission CSV file.

    Args:
        clip_names (np.array): Array of clip filenames.
        probabilities (np.array): Array of predicted probabilities.
        output_path (str): Path to save the CSV.
    """
    print(f"\nGenerating submission file at {output_path}...")

    df = pd.DataFrame({"clip": clip_names, "probability": probabilities})

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df.to_csv(output_path, index=False)
    print("Submission saved successfully.")
