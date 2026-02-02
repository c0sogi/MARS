import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

from library.config import Config
from library.utils import seed_everything, calculate_roc_auc


class MetaLearner:
    """
    Logistic Regression Meta-Learner that combines base model predictions
    with metadata features (file size) to produce a final probability.
    """

    def __init__(self):
        # Initialize Logistic Regression with fixed seed for reproducibility.
        # liblinear is good for smaller datasets and binary classification.
        self.model = LogisticRegression(
            random_state=Config.SEED, solver="liblinear", C=1.0
        )
        # StandardScaler is crucial when mixing probabilities (0-1) with
        # file size features (potentially different scale/variance).
        self.scaler = StandardScaler()

    def fit(self, X, y):
        """
        Fits the scaler and the logistic regression model.
        """
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        return self

    def predict_proba(self, X):
        """
        Predicts class 1 probabilities.
        """
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]


def normalize_file_sizes(file_sizes):
    """
    Normalizes file sizes using the same logic as the neural network dataset
    to provide a reasonable scale for the linear model.
    Logic: (size - 1000) / 500
    """
    return (np.array(file_sizes, dtype=np.float32) - 1000.0) / 500.0


def prepare_meta_features(preds_dict, file_sizes):
    """
    Constructs the feature matrix for the meta-learner.

    Features:
    1. Base Model Probabilities (one column per architecture)
    2. Normalized File Size (Metadata Injection)

    Args:
        preds_dict (dict): Dictionary {model_name: probabilities_array}.
                           Each array should be shape (N,).
        file_sizes (array-like): Array of file sizes, shape (N,).

    Returns:
        np.array: Feature matrix of shape (N, num_models + 1).
    """
    # Sort keys to ensure consistent column ordering between train and test
    model_names = sorted(preds_dict.keys())

    feature_columns = []

    # 1. Add Base Model Predictions
    for name in model_names:
        preds = preds_dict[name]
        # Ensure it's a column vector
        feature_columns.append(preds.reshape(-1, 1))

    # 2. Add Normalized File Size
    norm_fs = normalize_file_sizes(file_sizes).reshape(-1, 1)
    feature_columns.append(norm_fs)

    # Concatenate horizontally
    X = np.hstack(feature_columns)
    return X


def train_stacker(oof_preds_dict, file_sizes, y_true):
    """
    Trains the stacking meta-learner on Out-Of-Fold predictions.
    Uses Nested Cross-Validation to calculate an unbiased AUC score.

    Args:
        oof_preds_dict (dict): Dictionary of OOF predictions per model.
        file_sizes (array-like): File sizes for the training set.
        y_true (array-like): Ground truth labels.

    Returns:
        tuple: (trained_learner, oof_auc_score)
    """
    seed_everything(Config.SEED)

    print("Preparing meta-features for stacking...")
    X = prepare_meta_features(oof_preds_dict, file_sizes)

    print(
        f"Feature matrix shape: {X.shape} (Samples: {X.shape[0]}, Features: {X.shape[1]})"
    )

    # 1. Nested Cross-Validation for Unbiased Scoring
    print("Performing Nested CV for Stacker Evaluation...")
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )
    meta_oof_preds = np.zeros(len(y_true))

    # Ensure integer labels for stratification
    y_true_int = y_true.astype(int)

    for train_idx, val_idx in skf.split(X, y_true_int):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train = y_true[train_idx]

        # Train temporary learner on this fold's training set
        temp_learner = MetaLearner()
        temp_learner.fit(X_train, y_train)

        # Predict on this fold's validation set (unseen by temp_learner)
        meta_oof_preds[val_idx] = temp_learner.predict_proba(X_val)

    score = calculate_roc_auc(y_true, meta_oof_preds)
    print(f"Stacker Nested CV ROC AUC: {score}")

    # 2. Final Training on Full Data for Inference
    print("Training Final Logistic Regression Meta-Learner on Full Data...")
    learner = MetaLearner()
    learner.fit(X, y_true)

    return learner, score


def predict_stacker(learner, test_preds_dict, file_sizes):
    """
    Generates final predictions for the test set using the trained stacker.

    Args:
        learner (MetaLearner): Trained meta-learner instance.
        test_preds_dict (dict): Dictionary of test predictions per model.
        file_sizes (array-like): File sizes for the test set.

    Returns:
        np.array: Final probabilities for the test set.
    """
    X_test = prepare_meta_features(test_preds_dict, file_sizes)
    return learner.predict_proba(X_test)
