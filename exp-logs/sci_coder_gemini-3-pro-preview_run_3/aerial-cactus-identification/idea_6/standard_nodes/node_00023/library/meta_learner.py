import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from library.config import Config
from library.utils import calculate_roc_auc


def get_ground_truth_map():
    """
    Loads training and validation metadata to create a mapping from image ID to label.

    Returns:
        dict: A dictionary mapping image filenames (ids) to their binary labels.
    """
    # Load metadata files
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)

    # Concatenate to get the full labeled dataset
    full_df = pd.concat([train_df, val_df], ignore_index=True)

    # Create dictionary {id: label}
    gt_map = pd.Series(full_df.has_cactus.values, index=full_df.id).to_dict()
    return gt_map


def prepare_meta_features(predictions_dict, gt_map=None, is_train=True):
    """
    Converts a dictionary of model predictions into a feature matrix X and target vector y.
    Ensures deterministic ordering of samples and features.

    Args:
        predictions_dict (dict): Dictionary of structure {model_name: {img_id: prob}}.
        gt_map (dict, optional): Dictionary mapping img_id to ground truth label. Required if is_train=True.
        is_train (bool): Flag to indicate if preparing for training (returns X, y) or inference (returns X, ids).

    Returns:
        pd.DataFrame: The feature matrix X where columns are model predictions.
        np.ndarray or list: The target vector y (if is_train) or list of image IDs (if not is_train).
    """
    # Sort model names to ensure consistent column order
    model_names = sorted(list(predictions_dict.keys()))

    # Identify the set of image IDs (intersection of all models to be safe, or just from the first one)
    # We assume all models have predicted on the same set of IDs.
    first_model = model_names[0]
    img_ids = sorted(list(predictions_dict[first_model].keys()))

    # Construct the data dictionary for DataFrame creation
    data = {}
    for model_name in model_names:
        # Extract probabilities in the sorted order of img_ids
        # Using .get() with a default could mask errors, so we access directly to ensure integrity
        preds = [predictions_dict[model_name][img_id] for img_id in img_ids]
        data[model_name] = preds

    X = pd.DataFrame(data, index=img_ids)

    if is_train:
        if gt_map is None:
            raise ValueError("gt_map is required when preparing training features.")

        # Construct target vector aligned with img_ids
        y = np.array([gt_map[img_id] for img_id in img_ids])
        return X, y
    else:
        return X, img_ids


def train_meta_learner(oof_predictions):
    """
    Trains a Logistic Regression meta-learner on Out-Of-Fold (OOF) predictions.

    Args:
        oof_predictions (dict): A dictionary containing OOF predictions from base models.
                                Structure: {model_name: {img_id: probability}}

    Returns:
        sklearn.linear_model.LogisticRegression: The trained meta-learner model.
    """
    print("Preparing meta-features for stacking...")
    gt_map = get_ground_truth_map()

    # Prepare feature matrix X and target y
    X_train, y_train = prepare_meta_features(oof_predictions, gt_map, is_train=True)

    print(
        f"Training Meta-Learner on {len(X_train)} samples using {X_train.shape[1]} base models."
    )

    # Initialize Logistic Regression
    # Using 'liblinear' solver which is good for smaller datasets and binary classification
    meta_model = LogisticRegression(
        C=Config.META_C, random_state=Config.SEED, solver="liblinear"
    )

    # Fit the model
    meta_model.fit(X_train, y_train)

    # Evaluate on the OOF set (training set for the meta-learner)
    preds = meta_model.predict_proba(X_train)[:, 1]
    auc_score = calculate_roc_auc(y_train, preds)

    print(f"Meta-Learner OOF ROC AUC: {auc_score}")
    print("Meta-Learner Coefficients:")
    for name, coef in zip(X_train.columns, meta_model.coef_[0]):
        print(f"  {name}: {coef:.6f}")

    return meta_model


def predict_meta(meta_model, test_predictions):
    """
    Generates final predictions for the test set using the trained meta-learner
    and saves the submission file.

    Args:
        meta_model (sklearn.linear_model.LogisticRegression): The trained meta-learner.
        test_predictions (dict): A dictionary containing test predictions from base models.
                                 Structure: {model_name: {img_id: probability}}

    Returns:
        pd.DataFrame: The submission dataframe containing IDs and final probabilities.
    """
    print("Preparing test meta-features...")
    X_test, test_ids = prepare_meta_features(test_predictions, is_train=False)

    print(f"Generating final predictions for {len(X_test)} test samples...")

    # Predict probabilities (class 1)
    final_probs = meta_model.predict_proba(X_test)[:, 1]

    # Create submission DataFrame
    submission_df = pd.DataFrame({"id": test_ids, "has_cactus": final_probs})

    # Ensure submission directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    # Save submission file
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")

    return submission_df
