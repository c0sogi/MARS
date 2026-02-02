import os
import numpy as np
import pandas as pd

from library.config import (
    WORKING_DIR,
    SUBMISSION_PATH,
    EXPERT_LGBM_PARAMS,
    EXPERT_XGB_PARAMS,
    RANDOM_STATE,
)
from library.utils import seed_everything
from library.feature_engineering import generate_features
from library.model_zoo import LGBMWrapper, XGBWrapper, HeterogeneousEnsemble
from library.data_loader import load_sample_submission

# Set global seed
seed_everything(RANDOM_STATE)


def get_feature_columns(df):
    """
    Identifies feature columns by excluding known metadata columns.
    Must match the logic used in training_pipeline.py.
    """
    exclude_cols = [
        "contact_id",
        "game_play",
        "step",
        "nfl_player_id_1",
        "nfl_player_id_2",
        "contact",
        "datetime",
        "video_path_endzone",
        "video_path_sideline",
        "video_path_all29",
    ]
    return [c for c in df.columns if c not in exclude_cols]


def predict_and_submit(load_cached_data=True, nrows=None):
    """
    Generates features for the test set, loads the trained ensemble,
    runs inference, applies the optimal threshold, and creates the submission file.

    Args:
        load_cached_data (bool): Whether to use cached feature files.
        nrows (int, optional): Number of rows to process for debugging.
    """
    print("\n=== Starting Inference Pipeline ===")

    # 1. Generate Test Features (No Gating)
    # We must predict for ALL rows in the sample submission, so gating is False.
    df_test = generate_features(
        split="test", load_cached_data=load_cached_data, nrows=nrows, gating=False
    )

    # Identify feature columns
    feature_cols = get_feature_columns(df_test)
    X_test = df_test[feature_cols]
    print(f"Test Features Shape: {X_test.shape}")

    # 2. Load Models
    print("Loading trained models...")
    models_dir = os.path.join(WORKING_DIR, "models")

    # Load LightGBM
    lgbm_path = os.path.join(models_dir, "expert_lgbm.joblib")
    if not os.path.exists(lgbm_path):
        raise FileNotFoundError(f"Expert LightGBM model not found at {lgbm_path}")

    lgbm_model = LGBMWrapper(EXPERT_LGBM_PARAMS, name="expert_lgbm")
    lgbm_model.load(lgbm_path)

    # Load XGBoost
    xgb_path = os.path.join(models_dir, "expert_xgb.joblib")
    if not os.path.exists(xgb_path):
        raise FileNotFoundError(f"Expert XGBoost model not found at {xgb_path}")

    xgb_model = XGBWrapper(EXPERT_XGB_PARAMS, name="expert_xgb")
    xgb_model.load(xgb_path)

    # Create Ensemble
    ensemble = HeterogeneousEnsemble([lgbm_model, xgb_model])

    # 3. Load Threshold
    thresh_path = os.path.join(models_dir, "best_threshold.npy")
    if not os.path.exists(thresh_path):
        raise FileNotFoundError(f"Threshold file not found at {thresh_path}")

    best_threshold = np.load(thresh_path)[0]
    print(f"Loaded optimal decision threshold: {best_threshold:.4f}")

    # 4. Inference
    print("Running ensemble inference...")
    probs = ensemble.predict(X_test)

    # Apply threshold
    predictions = (probs >= best_threshold).astype(int)

    # 5. Create Submission
    print("Formatting submission...")

    # Load template to ensure correct order and row count
    df_sub_template = load_sample_submission()

    # Create a dataframe of predictions with contact_id
    df_preds = pd.DataFrame(
        {"contact_id": df_test["contact_id"], "contact_pred": predictions}
    )

    # Merge predictions onto the template
    # Left join ensures we keep all rows from sample_submission
    # We drop the original 'contact' column (which is all 0s) from template first
    df_final = df_sub_template.drop(columns=["contact"]).merge(
        df_preds, on="contact_id", how="left"
    )

    # Fill missing values with 0
    # (Rows might be missing if tracking data was completely absent for a play,
    # though generate_features handles this by left joining and filling 0s)
    fill_count = df_final["contact_pred"].isna().sum()
    if fill_count > 0:
        print(f"Warning: {fill_count} rows missing predictions. Filling with 0.")

    df_final["contact"] = df_final["contact_pred"].fillna(0).astype(int)

    # Select final columns
    submission_df = df_final[["contact_id", "contact"]]

    # Save
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)
    submission_df.to_csv(SUBMISSION_PATH, index=False)

    print(f"Submission saved to {SUBMISSION_PATH}")
    print(f"Submission Shape: {submission_df.shape}")
    print("=== Inference Complete ===")
