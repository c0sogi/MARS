import os
import pandas as pd
import numpy as np
import joblib
import json

from library.config import Config
from library.utils import setup_seed, calc_mcc, optimize_threshold
from library.feature_engineering import process_data
from library.model import ContactModel


def train_pipeline(load_cached_data=True):
    """
    Orchestrates the training of the Dual-Stream architecture.

    1. Loads features for Stream A (Interaction) and Stream B (Impact).
    2. Trains XGBoost models for each stream with asymmetric configs.
    3. Optimizes thresholds per stream on validation data.
    4. Saves models and thresholds.
    5. Prints validation metrics.
    """
    setup_seed(Config.SEED)

    # Define paths for artifacts
    model_a_path = os.path.join(Config.WORKING_DIR, "model_stream_a.joblib")
    model_b_path = os.path.join(Config.WORKING_DIR, "model_stream_b.joblib")
    thresholds_path = os.path.join(Config.WORKING_DIR, "thresholds.joblib")

    print("Loading Training and Validation Data...")
    # Load Train Data
    X_train_a, ids_train_a, y_train_a, X_train_b, ids_train_b, y_train_b = process_data(
        "train", load_cached_data=load_cached_data
    )

    # Load Validation Data
    X_val_a, ids_val_a, y_val_a, X_val_b, ids_val_b, y_val_b = process_data(
        "validation", load_cached_data=load_cached_data
    )

    # --- Stream A: Interaction Model (Player vs Player) ---
    print("\n--- Training Stream A: Interaction Model ---")
    model_a = ContactModel(Config.STREAM_A_PARAMS)
    model_a.fit(X_train_a, y_train_a, X_val=X_val_a, y_val=y_val_a, verbose=True)

    # Predict on validation to find threshold
    print("Optimizing Stream A Threshold...")
    y_pred_proba_a = model_a.predict(X_val_a)
    best_thresh_a, best_score_a = optimize_threshold(y_val_a, y_pred_proba_a)
    print(f"Stream A - Best Threshold: {best_thresh_a}")
    print(f"Stream A - Validation MCC: {best_score_a}")

    # Save Model A
    model_a.save(model_a_path)

    # --- Stream B: Impact Model (Player vs Ground) ---
    print("\n--- Training Stream B: Impact Model ---")
    model_b = ContactModel(Config.STREAM_B_PARAMS)
    model_b.fit(X_train_b, y_train_b, X_val=X_val_b, y_val=y_val_b, verbose=True)

    # Predict on validation to find threshold
    print("Optimizing Stream B Threshold...")
    y_pred_proba_b = model_b.predict(X_val_b)
    best_thresh_b, best_score_b = optimize_threshold(y_val_b, y_pred_proba_b)
    print(f"Stream B - Best Threshold: {best_thresh_b}")
    print(f"Stream B - Validation MCC: {best_score_b}")

    # Save Model B
    model_b.save(model_b_path)

    # --- Global Validation Metrics ---
    print("\n--- Global Validation Evaluation ---")

    # Binarize predictions
    y_pred_bin_a = (y_pred_proba_a >= best_thresh_a).astype(int)
    y_pred_bin_b = (y_pred_proba_b >= best_thresh_b).astype(int)

    # Concatenate all validation samples
    y_val_total = np.concatenate([y_val_a, y_val_b])
    y_pred_total = np.concatenate([y_pred_bin_a, y_pred_bin_b])

    global_mcc = calc_mcc(y_val_total, y_pred_total)
    print(f"Global Validation MCC: {global_mcc}")

    # Save Thresholds
    thresholds = {"stream_a": best_thresh_a, "stream_b": best_thresh_b}
    joblib.dump(thresholds, thresholds_path)
    print(f"Thresholds saved to {thresholds_path}")


def inference_pipeline(load_cached_data=True):
    """
    Orchestrates the inference process.

    1. Loads test features.
    2. Loads trained models and thresholds.
    3. Generates predictions for both streams.
    4. Merges and formats predictions into submission.csv.
    """
    setup_seed(Config.SEED)

    model_a_path = os.path.join(Config.WORKING_DIR, "model_stream_a.joblib")
    model_b_path = os.path.join(Config.WORKING_DIR, "model_stream_b.joblib")
    thresholds_path = os.path.join(Config.WORKING_DIR, "thresholds.joblib")

    # Check artifacts
    if not (
        os.path.exists(model_a_path)
        and os.path.exists(model_b_path)
        and os.path.exists(thresholds_path)
    ):
        raise FileNotFoundError(
            "Models or thresholds not found. Run train_pipeline first."
        )

    print("Loading Test Data...")
    X_test_a, ids_test_a, X_test_b, ids_test_b = process_data(
        "test", load_cached_data=load_cached_data
    )

    print("Loading Models and Thresholds...")
    model_a = ContactModel.load(model_a_path)
    model_b = ContactModel.load(model_b_path)
    thresholds = joblib.load(thresholds_path)

    thresh_a = thresholds["stream_a"]
    thresh_b = thresholds["stream_b"]

    # --- Stream A Inference ---
    print("Predicting Stream A...")
    if len(ids_test_a) > 0:
        proba_a = model_a.predict(X_test_a)
        pred_a = (proba_a >= thresh_a).astype(int)
        df_a = pd.DataFrame({"contact_id": ids_test_a, "contact": pred_a})
    else:
        df_a = pd.DataFrame(columns=["contact_id", "contact"])

    # --- Stream B Inference ---
    print("Predicting Stream B...")
    if len(ids_test_b) > 0:
        proba_b = model_b.predict(X_test_b)
        pred_b = (proba_b >= thresh_b).astype(int)
        df_b = pd.DataFrame({"contact_id": ids_test_b, "contact": pred_b})
    else:
        df_b = pd.DataFrame(columns=["contact_id", "contact"])

    # --- Merge and Format ---
    print("Generating Submission...")
    df_preds = pd.concat([df_a, df_b], ignore_index=True)

    # Load sample submission to ensure correct order and completeness
    df_sample = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)

    # Merge predictions onto sample submission
    # This ensures we have the exact rows required by the competition
    df_final = df_sample[["contact_id"]].merge(df_preds, on="contact_id", how="left")

    # Fill missing values with 0 (default no contact) if any IDs were missed (should not happen)
    missing_count = df_final["contact"].isnull().sum()
    if missing_count > 0:
        print(
            f"Warning: {missing_count} contact_ids were missing from predictions. Filling with 0."
        )
        df_final["contact"] = df_final["contact"].fillna(0)

    # Ensure integer type
    df_final["contact"] = df_final["contact"].astype(int)

    # Save
    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    df_final.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
