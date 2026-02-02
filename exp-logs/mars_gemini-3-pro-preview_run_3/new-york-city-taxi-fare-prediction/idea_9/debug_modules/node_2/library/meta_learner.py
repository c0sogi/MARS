import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

from library.config import (
    WORKING_DIR,
    SUBMISSION_DIR,
    META_PARAMS,
    SEED,
    PATH_SUBMISSION,
)
from library.utils import seed_everything
from library.model_tree import predict_tree_model
from library.model_nn import predict_nn_model

# Ensure directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)


def train_meta_learner(models, df_meta_tree, df_meta_nn, load_cached_model=True):
    """
    Trains the Meta Learner (Ridge Regression) on the out-of-sample predictions
    from the base models (Level 1 stacking).

    Args:
        models (dict): Dictionary containing trained base models:
                       {'xgboost': model, 'lgbm': model, 'nn': model}
        df_meta_tree (pd.DataFrame): Meta-training data processed for tree models.
        df_meta_nn (pd.DataFrame): Meta-training data processed for NN models.
        load_cached_model (bool): Whether to load a saved model if available.

    Returns:
        sklearn.linear_model.Ridge: Trained meta-learner model.
    """
    seed_everything(SEED)
    model_path = os.path.join(WORKING_DIR, "meta_model.joblib")

    if load_cached_model and os.path.exists(model_path):
        print(f"Loading Meta Learner from {model_path}...")
        return joblib.load(model_path)

    print("Training Meta Learner (Ridge Regression)...")

    # 1. Generate Base Predictions on Meta Set
    print("Generating base predictions for meta-training...")

    # XGBoost
    pred_xgb = predict_tree_model(models["xgboost"], df_meta_tree)

    # LightGBM
    pred_lgbm = predict_tree_model(models["lgbm"], df_meta_tree)

    # Spatial ResNet
    pred_nn = predict_nn_model(models["nn"], df_meta_nn)

    # 2. Construct Meta Features Matrix
    # Stack predictions as columns: (N_samples, 3)
    X_meta = np.column_stack([pred_xgb, pred_lgbm, pred_nn])

    # 3. Get Target
    # Assuming datasets are aligned (guaranteed by process_features logic)
    if "fare_amount" not in df_meta_tree.columns:
        raise ValueError("Target 'fare_amount' missing from meta training set.")

    y_meta = df_meta_tree["fare_amount"].values

    # 4. Train Ridge Regression
    meta_model = Ridge(**META_PARAMS)
    meta_model.fit(X_meta, y_meta)

    # 5. Evaluate on Meta Set (Training Error for Meta Model)
    preds_meta = meta_model.predict(X_meta)
    rmse = np.sqrt(mean_squared_error(y_meta, preds_meta))
    print(f"Meta Learner Training RMSE: {rmse}")

    # Print learned coefficients to see model contribution
    print(
        f"Meta Learner Coefficients: XGB={meta_model.coef_[0]:.4f}, LGBM={meta_model.coef_[1]:.4f}, NN={meta_model.coef_[2]:.4f}"
    )
    print(f"Meta Learner Intercept: {meta_model.intercept_:.4f}")

    # 6. Save Model
    print(f"Saving Meta Learner to {model_path}...")
    joblib.dump(meta_model, model_path)

    return meta_model


def predict_meta(models, meta_model, df_test_tree, df_test_nn):
    """
    Generates final predictions using the stacked ensemble.

    Args:
        models (dict): Dictionary of base models.
        meta_model (sklearn.linear_model.Ridge): Trained meta-learner.
        df_test_tree (pd.DataFrame): Test data for tree models.
        df_test_nn (pd.DataFrame): Test data for NN models.

    Returns:
        np.array: Final predictions.
    """
    print("Generating base predictions for test set...")

    # XGBoost
    pred_xgb = predict_tree_model(models["xgboost"], df_test_tree)

    # LightGBM
    pred_lgbm = predict_tree_model(models["lgbm"], df_test_tree)

    # Spatial ResNet
    pred_nn = predict_nn_model(models["nn"], df_test_nn)

    # Stack
    X_test_meta = np.column_stack([pred_xgb, pred_lgbm, pred_nn])

    # Final Prediction
    print("Generating final ensemble predictions...")
    final_preds = meta_model.predict(X_test_meta)

    return final_preds


def generate_submission(df_test_raw, predictions):
    """
    Creates and saves the submission CSV file.

    Args:
        df_test_raw (pd.DataFrame): Original test dataframe containing 'key'.
        predictions (np.array): Predicted fare amounts.
    """
    print(f"Generating submission file at {PATH_SUBMISSION}...")

    if len(df_test_raw) != len(predictions):
        raise ValueError(
            f"Length mismatch: Test data ({len(df_test_raw)}) vs Predictions ({len(predictions)})"
        )

    submission = pd.DataFrame({"key": df_test_raw["key"], "fare_amount": predictions})

    # Ensure formatting matches sample_submission (float with 2 decimals often preferred, but standard float is fine)
    # The prompt example shows 11.00, 12.05, etc.
    submission.to_csv(PATH_SUBMISSION, index=False)
    print("Submission saved successfully.")
