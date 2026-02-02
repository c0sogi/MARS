import os
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
from typing import Dict, Tuple, List
from library.config import Config
from library.utils import calculate_auc


def prepare_stacking_data(
    oof_predictions: Dict[str, pd.DataFrame],
    test_predictions: Dict[str, pd.DataFrame],
    train_labels: pd.DataFrame,
    test_ids: pd.DataFrame,
    load_cached_data: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepares and caches the feature matrices for the Level-1 meta-learner.

    Merges the Out-Of-Fold (OOF) predictions from base models to create the training set,
    and the averaged test predictions to create the test set.

    Args:
        oof_predictions (Dict[str, pd.DataFrame]): Dictionary mapping model names to their OOF prediction DataFrames.
                                                   Each DataFrame must contain ['id', 'pred'].
        test_predictions (Dict[str, pd.DataFrame]): Dictionary mapping model names to their Test prediction DataFrames.
                                                    Each DataFrame must contain ['id', 'pred'].
        train_labels (pd.DataFrame): DataFrame containing ground truth ['id', 'label'].
        test_ids (pd.DataFrame): DataFrame containing test ['id'].
        load_cached_data (bool): If True, attempts to load pre-computed features from Parquet cache.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (train_df, test_df) ready for the meta-learner.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    train_cache_path = os.path.join(Config.WORKING_DIR, "stacked_train_data.parquet")
    test_cache_path = os.path.join(Config.WORKING_DIR, "stacked_test_data.parquet")

    # 1. Try to load from cache
    if (
        load_cached_data
        and os.path.exists(train_cache_path)
        and os.path.exists(test_cache_path)
    ):
        try:
            train_df = pd.read_parquet(train_cache_path)
            test_df = pd.read_parquet(test_cache_path)
            return train_df, test_df
        except Exception:
            # Proceed to re-compute if cache load fails
            pass

    # 2. Prepare Training Data (Features + Labels)
    # Initialize with ground truth labels
    train_df = train_labels[["id", "label"]].copy()

    # Merge OOF predictions from each base model
    for model_name, df_preds in oof_predictions.items():
        # Rename 'pred' column to 'pred_{model_name}' to avoid collisions
        # We assume df_preds has 'id' and 'pred'
        temp_df = df_preds[["id", "pred"]].rename(
            columns={"pred": f"pred_{model_name}"}
        )
        train_df = train_df.merge(temp_df, on="id", how="left")

    # 3. Prepare Test Data (Features only)
    # Initialize with test IDs
    test_df = test_ids[["id"]].copy()

    # Merge Test predictions from each base model
    for model_name, df_preds in test_predictions.items():
        temp_df = df_preds[["id", "pred"]].rename(
            columns={"pred": f"pred_{model_name}"}
        )
        test_df = test_df.merge(temp_df, on="id", how="left")

    # 4. Save to cache
    try:
        train_df.to_parquet(train_cache_path, index=False)
        test_df.to_parquet(test_cache_path, index=False)
    except Exception as e:
        print(f"Warning: Failed to save stacking data cache: {e}")

    return train_df, test_df


def train_meta_learner(train_df: pd.DataFrame) -> Tuple[xgb.XGBClassifier, List[str]]:
    """
    Trains the XGBoost meta-learner on the OOF predictions.

    Args:
        train_df (pd.DataFrame): The prepared training DataFrame containing labels and model predictions.

    Returns:
        Tuple[xgb.XGBClassifier, List[str]]: The trained model and the list of feature column names used.
    """
    # Identify feature columns (all columns starting with 'pred_')
    feature_cols = [col for col in train_df.columns if col.startswith("pred_")]

    if not feature_cols:
        raise ValueError("No prediction feature columns found in training data.")

    X = train_df[feature_cols]
    y = train_df["label"]

    print(f"Training Meta-Learner (XGBoost) on {len(X)} samples.")
    print(f"Features: {feature_cols}")

    # Initialize XGBoost with parameters from Config
    model = xgb.XGBClassifier(**Config.META_MODEL_PARAMS)

    # Train the model
    model.fit(X, y)

    # Evaluate on the training set (which is the OOF set of the base models)
    # This gives the Cross-Validation score of the Stack
    preds = model.predict_proba(X)[:, 1]
    auc_score = calculate_auc(y, preds)

    print(f"Meta-Learner OOF Ensemble AUC: {auc_score}")

    # Save the trained meta-learner
    model_path = os.path.join(Config.WORKING_DIR, "meta_learner.joblib")
    joblib.dump(model, model_path)

    return model, feature_cols


def predict_meta_learner(
    model: xgb.XGBClassifier, test_df: pd.DataFrame, feature_cols: List[str]
) -> pd.DataFrame:
    """
    Generates predictions for the test set using the trained meta-learner and saves the submission file.

    Args:
        model (xgb.XGBClassifier): The trained meta-learner.
        test_df (pd.DataFrame): The prepared test DataFrame containing model predictions.
        feature_cols (List[str]): List of feature names to use for prediction.

    Returns:
        pd.DataFrame: The submission DataFrame.
    """
    # Ensure features are in the same order as training
    X_test = test_df[feature_cols]

    # Generate probabilities (class 1)
    preds = model.predict_proba(X_test)[:, 1]

    # Create submission DataFrame
    submission_df = pd.DataFrame({"id": test_df["id"], "label": preds})

    # Ensure submission directory exists
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Save submission file
    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    submission_df.to_csv(submission_path, index=False)

    # print(f"Submission saved to {submission_path}") # Silent execution preferred

    return submission_df
