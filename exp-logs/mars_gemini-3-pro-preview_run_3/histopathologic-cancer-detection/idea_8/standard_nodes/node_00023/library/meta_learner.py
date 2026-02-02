import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from library.config import Config
from library.utils import calculate_auc, seed_everything


def train_xgboost_meta_learner(oof_df: pd.DataFrame, load_cached_model: bool = True):
    """
    Trains an XGBoost meta-learner on the Out-Of-Fold (OOF) predictions from base models.

    Args:
        oof_df (pd.DataFrame): DataFrame containing 'label' and columns for each model architecture
                               defined in Config.MODEL_ARCHS.
        load_cached_model (bool): If True, attempts to load a pre-trained model from disk.

    Returns:
        xgb.XGBClassifier: The trained or loaded XGBoost model.
    """
    seed_everything(Config.SEED)

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    model_path = os.path.join(Config.WORKING_DIR, "meta_learner.json")

    # Define feature columns based on architecture names
    feature_cols = Config.MODEL_ARCHS

    # Check if columns exist
    missing_cols = [col for col in feature_cols if col not in oof_df.columns]
    if missing_cols:
        raise ValueError(f"OOF DataFrame missing columns: {missing_cols}")

    if "label" not in oof_df.columns:
        raise ValueError("OOF DataFrame missing 'label' column")

    # --- Caching Logic ---
    if load_cached_model and os.path.exists(model_path):
        print(f"Loading cached meta-learner from {model_path}")
        model = xgb.XGBClassifier()
        model.load_model(model_path)
        return model

    print("Training meta-learner (XGBoost)...")

    X = oof_df[feature_cols]
    y = oof_df["label"]

    # Split OOF data for internal validation (Early Stopping)
    # We use a stratified split to maintain class balance
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=Config.SEED, stratify=y
    )

    # Initialize XGBoost Classifier
    # Using conservative parameters for stacking to avoid overfitting
    model = xgb.XGBClassifier(
        n_estimators=2000,
        learning_rate=0.01,
        max_depth=3,  # Shallow trees are often better for stacking
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="auc",
        n_jobs=Config.NUM_WORKERS,
        random_state=Config.SEED,
        early_stopping_rounds=50,
    )

    # Train with early stopping
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    # Evaluate
    val_preds = model.predict_proba(X_val)[:, 1]
    val_auc = calculate_auc(y_val, val_preds)
    print(f"Meta-Learner Validation AUC: {val_auc:.10f}")

    # Save model using JSON (avoiding pickle)
    try:
        model.save_model(model_path)
        print(f"Saved meta-learner to {model_path}")
    except Exception as e:
        print(f"Warning: Could not save meta-learner: {e}")

    return model


def inference_meta_learner(model: xgb.XGBClassifier, test_pred_df: pd.DataFrame):
    """
    Generates final predictions using the trained meta-learner.

    Args:
        model (xgb.XGBClassifier): The trained meta-learner.
        test_pred_df (pd.DataFrame): DataFrame containing 'id' and columns for each model architecture.

    Returns:
        pd.DataFrame: Submission DataFrame with 'id' and 'label'.
    """
    # Define feature columns
    feature_cols = Config.MODEL_ARCHS

    # Check columns
    missing_cols = [col for col in feature_cols if col not in test_pred_df.columns]
    if missing_cols:
        raise ValueError(f"Test Prediction DataFrame missing columns: {missing_cols}")

    X_test = test_pred_df[feature_cols]

    # Predict probabilities
    # predict_proba returns [prob_class_0, prob_class_1]
    final_probs = model.predict_proba(X_test)[:, 1]

    # Create submission DataFrame
    submission = pd.DataFrame({"id": test_pred_df["id"], "label": final_probs})

    # Save submission
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Saved final submission to {Config.SUBMISSION_PATH}")

    return submission
