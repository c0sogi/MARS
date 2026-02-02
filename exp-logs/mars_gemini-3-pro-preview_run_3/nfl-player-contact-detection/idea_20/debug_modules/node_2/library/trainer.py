import pandas as pd
import numpy as np
import xgboost as xgb
import os
import library.config as C
import library.utils as U
from sklearn.metrics import matthews_corrcoef


def undersample_training_data(X, y, ratio=C.NEG_POS_RATIO, seed=C.SEED):
    """
    Performs Targeted Majority Undersampling.
    Keeps 100% of positive samples and subsamples negative samples to a specific ratio.

    Args:
        X (pd.DataFrame): Feature matrix.
        y (np.array): Target vector.
        ratio (float): Ratio of negative to positive samples (e.g., 10.0).
        seed (int): Random seed for reproducibility.

    Returns:
        tuple: (X_resampled, y_resampled)
    """
    np.random.seed(seed)

    # Identify indices
    pos_indices = np.where(y == 1)[0]
    neg_indices = np.where(y == 0)[0]

    n_pos = len(pos_indices)
    n_neg_total = len(neg_indices)

    # Calculate number of negatives to keep
    n_neg_keep = int(n_pos * ratio)
    n_neg_keep = min(
        n_neg_keep, n_neg_total
    )  # Ensure we don't request more than available

    # Randomly sample negatives
    neg_indices_sampled = np.random.choice(neg_indices, size=n_neg_keep, replace=False)

    # Combine and shuffle
    all_indices = np.concatenate([pos_indices, neg_indices_sampled])
    np.random.shuffle(all_indices)

    X_resampled = X.iloc[all_indices].copy()
    y_resampled = y[all_indices]

    return X_resampled, y_resampled


def optimize_threshold(y_true, y_pred_proba):
    """
    Performs a linear search to find the probability threshold that maximizes MCC.

    Args:
        y_true (np.array): Ground truth labels.
        y_pred_proba (np.array): Predicted probabilities.

    Returns:
        tuple: (best_threshold, best_mcc)
    """
    best_mcc = -1.0
    best_thresh = 0.5

    # Search space: 0.01 to 0.99
    thresholds = np.linspace(0.01, 0.99, 99)

    for thresh in thresholds:
        y_pred = (y_pred_proba >= thresh).astype(int)
        mcc = matthews_corrcoef(y_true, y_pred)

        if mcc > best_mcc:
            best_mcc = mcc
            best_thresh = thresh

    return best_thresh, best_mcc


def train_stream_model(stream_type, X_train, y_train, X_val, y_val, save_model=True):
    """
    Trains an XGBoost model for a specific stream (A or B).
    Handles undersampling, training, and threshold optimization.

    Args:
        stream_type (str): 'A' or 'B'.
        X_train (pd.DataFrame): Training features.
        y_train (np.array): Training labels.
        X_val (pd.DataFrame): Validation features.
        y_val (np.array): Validation labels.
        save_model (bool): Whether to save the model to disk.

    Returns:
        tuple: (model, best_threshold, best_mcc)
    """
    print(f"Starting training for Stream {stream_type}...")

    # 1. Select Configuration
    if stream_type == "A":
        params = C.STREAM_A_PARAMS
    elif stream_type == "B":
        params = C.STREAM_B_PARAMS
    else:
        raise ValueError("stream_type must be 'A' or 'B'")

    # 2. Undersample Training Data
    print(f"Undersampling training data (Ratio {C.NEG_POS_RATIO}:1)...")
    print(f"Original Train Shape: {X_train.shape}, Positives: {np.sum(y_train)}")
    X_train_res, y_train_res = undersample_training_data(X_train, y_train)
    print(
        f"Resampled Train Shape: {X_train_res.shape}, Positives: {np.sum(y_train_res)}"
    )

    # 3. Initialize Model
    # Note: XGBClassifier handles the DMatrix conversion internally
    clf = xgb.XGBClassifier(
        **params,
        callbacks=[
            xgb.callback.EarlyStopping(rounds=C.EARLY_STOPPING_ROUNDS, save_best=True)
        ],
    )

    # 4. Train
    print("Fitting model...")
    # We pass eval_set for early stopping
    clf.fit(
        X_train_res,
        y_train_res,
        eval_set=[(X_val, y_val)],
        verbose=False,  # Suppress iteration logs as requested
    )

    # 5. Evaluate on Validation Set
    print("Evaluating on validation set...")
    # Predict probabilities (class 1)
    y_val_proba = clf.predict_proba(X_val)[:, 1]

    # Optimize Threshold
    best_thresh, best_mcc = optimize_threshold(y_val, y_val_proba)

    print(f"Stream {stream_type} Validation Results:")
    print(f"Best Threshold: {best_thresh}")
    print(f"Best MCC: {best_mcc}")  # Printing full precision

    # 6. Save Model
    if save_model:
        model_filename = f"model_stream_{stream_type.lower()}.json"
        model_path = os.path.join(C.WORKING_DIR, model_filename)
        print(f"Saving model to {model_path}...")
        clf.save_model(model_path)

    return clf, best_thresh, best_mcc
