import os
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from library.config import Config


def train_meta_learner(oof_preds_dict=None, targets=None, load_cached_data=True):
    """
    Trains a Logistic Regression meta-learner on OOF predictions.

    This function implements a caching mechanism for the OOF data. If a cache exists
    and load_cached_data is True, it loads the data from disk. Otherwise, it constructs
    the dataset from the provided dictionary and saves it to a Parquet file.

    Args:
        oof_preds_dict (dict, optional): Dictionary where keys are model names and values
                                         are arrays of predicted probabilities (N_samples,).
        targets (array-like, optional): Ground truth binary labels (N_samples,).
        load_cached_data (bool): If True, attempts to load pre-processed OOF data from cache.

    Returns:
        float: The ROC AUC score of the meta-learner on the training (OOF) set.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    cache_path = os.path.join(Config.WORKING_DIR, "stacked_oof_data.parquet")
    model_path = os.path.join(Config.WORKING_DIR, "meta_learner.pth")

    df = None

    # 1. Attempt to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            print(f"Loaded OOF data from {cache_path}")
        except Exception as e:
            print(
                f"Warning: Failed to load cache ({e}). Proceeding to create from scratch."
            )

    # 2. Create data from inputs if not loaded
    if df is None:
        if oof_preds_dict is None or targets is None:
            raise ValueError(
                "No cached OOF data found and no inputs provided to create it."
            )

        # Convert dictionary to DataFrame
        df = pd.DataFrame(oof_preds_dict)

        # Ensure targets are a flat numpy array
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        df["label"] = np.array(targets).ravel()

        # Sort feature columns alphabetically to ensure deterministic feature order
        feature_cols = sorted([c for c in df.columns if c != "label"])
        df = df[feature_cols + ["label"]]

        # Save to cache
        try:
            df.to_parquet(cache_path, index=False)
            print(f"Saved OOF data to {cache_path}")
        except Exception as e:
            print(f"Warning: Failed to save OOF cache: {e}")

    # 3. Prepare Training Data
    feature_cols = sorted([c for c in df.columns if c != "label"])
    X = df[feature_cols]
    y = df["label"]

    print(f"Training Meta-Learner on {len(df)} samples.")
    print(f"Features: {feature_cols}")

    # 4. Train Logistic Regression
    # Using 'liblinear' solver which is robust for binary classification
    meta_model = LogisticRegression(
        random_state=Config.SEED, solver="liblinear", C=1.0, penalty="l2"
    )
    meta_model.fit(X, y)

    # 5. Evaluate
    preds = meta_model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, preds)
    print(f"Meta-Learner OOF AUC: {auc}")

    # 6. Save the trained model
    # Using torch.save to serialize the sklearn object (wrapper around pickle)
    # to avoid potential dependency issues with direct joblib imports if not listed.
    torch.save(meta_model, model_path)
    print(f"Saved Meta-Learner model to {model_path}")

    return auc


def predict_with_meta_learner(test_preds_dict):
    """
    Generates predictions for the test set using the trained meta-learner.

    Args:
        test_preds_dict (dict): Dictionary where keys are model names and values
                                are arrays of predicted probabilities for the test set.

    Returns:
        np.ndarray: Final ensemble probabilities.
    """
    model_path = os.path.join(Config.WORKING_DIR, "meta_learner.pth")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Meta-learner model not found at {model_path}. Please train it first."
        )

    # Load the model
    meta_model = torch.load(model_path)

    # Construct DataFrame
    df = pd.DataFrame(test_preds_dict)

    # Ensure columns are sorted alphabetically to match training order
    feature_cols = sorted(df.columns)
    X = df[feature_cols]

    # Predict
    preds = meta_model.predict_proba(X)[:, 1]

    return preds


def create_submission(test_ids, predictions):
    """
    Creates and saves the submission CSV file.

    Args:
        test_ids (list or np.array): The IDs of the test samples.
        predictions (list or np.array): The predicted probabilities.
    """
    if len(test_ids) != len(predictions):
        raise ValueError(
            f"Length mismatch: IDs ({len(test_ids)}) vs Predictions ({len(predictions)})"
        )

    submission_df = pd.DataFrame({"id": test_ids, "label": predictions})

    # Ensure submission directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_FILE), exist_ok=True)

    submission_df.to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {Config.SUBMISSION_FILE}")
