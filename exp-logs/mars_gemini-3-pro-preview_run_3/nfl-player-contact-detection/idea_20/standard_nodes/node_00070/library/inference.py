import pandas as pd
import numpy as np
import xgboost as xgb
import os
import library.config as C
import library.utils as U
from library.trainer import optimize_threshold


def load_model(stream_type):
    """
    Loads a trained XGBoost model for the specified stream.

    Args:
        stream_type (str): 'A' or 'B'.

    Returns:
        xgb.XGBClassifier: The loaded model.
    """
    model_filename = f"model_stream_{stream_type.lower()}.json"
    model_path = os.path.join(C.WORKING_DIR, model_filename)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    # Initialize a blank classifier with the same parameters to load into
    # Note: The parameters here are just for initialization; load_model overwrites tree structure
    params = C.STREAM_A_PARAMS if stream_type == "A" else C.STREAM_B_PARAMS
    clf = xgb.XGBClassifier(**params)
    clf.load_model(model_path)

    return clf


def predict_stream(model, X, ids, threshold):
    """
    Generates binary predictions for a specific stream using a given threshold.

    Args:
        model (xgb.XGBClassifier): The trained model.
        X (pd.DataFrame): Feature matrix.
        ids (np.array): Contact IDs corresponding to X.
        threshold (float): Probability threshold for positive class.

    Returns:
        pd.DataFrame: DataFrame with 'contact_id' and 'contact' columns.
    """
    # Predict probabilities for the positive class (1)
    # XGBoost predict_proba returns [prob_0, prob_1]
    probas = model.predict_proba(X)[:, 1]

    # Apply threshold
    predictions = (probas >= threshold).astype(int)

    # Create DataFrame
    df_pred = pd.DataFrame({"contact_id": ids, "contact": predictions})

    return df_pred


def generate_submission_file(combined_preds_df, output_path=C.SUBMISSION_PATH):
    """
    Merges predictions with the sample submission to ensure correct format and order.

    Args:
        combined_preds_df (pd.DataFrame): Predictions containing 'contact_id' and 'contact'.
        output_path (str): Path to save the submission CSV.
    """
    sample_sub_path = os.path.join(C.INPUT_DIR, "sample_submission.csv")
    if not os.path.exists(sample_sub_path):
        print(
            f"Warning: Sample submission not found at {sample_sub_path}. Creating from predictions directly."
        )
        combined_preds_df.to_csv(output_path, index=False)
        return

    # Load sample submission
    df_sample = pd.read_csv(sample_sub_path)

    # Drop the placeholder 'contact' column from sample
    if "contact" in df_sample.columns:
        df_sample = df_sample.drop(columns=["contact"])

    # Merge predictions
    # Left join ensures we keep all rows from sample submission in the correct order
    df_sub = pd.merge(df_sample, combined_preds_df, on="contact_id", how="left")

    # Fill missing values with 0 (no contact)
    # This handles any IDs that might have been filtered out during processing
    df_sub["contact"] = df_sub["contact"].fillna(0).astype(int)

    # Save
    print(f"Saving submission to {output_path}...")
    df_sub.to_csv(output_path, index=False)
    print(f"Submission shape: {df_sub.shape}")


def run_inference_pipeline(
    X_val_a, y_val_a, X_val_b, y_val_b, X_test_a, ids_test_a, X_test_b, ids_test_b
):
    """
    Orchestrates the full inference process:
    1. Loads models.
    2. Optimizes thresholds on validation data.
    3. Predicts on test data.
    4. Generates submission file.

    Args:
        X_val_a, y_val_a: Validation data for Stream A.
        X_val_b, y_val_b: Validation data for Stream B.
        X_test_a, ids_test_a: Test data for Stream A.
        X_test_b, ids_test_b: Test data for Stream B.
    """
    print("Starting Inference Pipeline...")

    # --- 1. Load Models ---
    print("Loading models...")
    model_a = load_model("A")
    model_b = load_model("B")

    # --- 2. Optimize Thresholds (Stream A) ---
    print("Optimizing threshold for Stream A (Interaction)...")
    probas_val_a = model_a.predict_proba(X_val_a)[:, 1]
    thresh_a, mcc_a = optimize_threshold(y_val_a, probas_val_a)
    print(f"Stream A - Optimized Threshold: {thresh_a:.4f}, Validation MCC: {mcc_a}")

    # --- 3. Optimize Thresholds (Stream B) ---
    print("Optimizing threshold for Stream B (Impact)...")
    probas_val_b = model_b.predict_proba(X_val_b)[:, 1]
    thresh_b, mcc_b = optimize_threshold(y_val_b, probas_val_b)
    print(f"Stream B - Optimized Threshold: {thresh_b:.4f}, Validation MCC: {mcc_b}")

    # --- 4. Predict on Test Set ---
    print("Generating predictions for Test Set...")

    # Stream A Predictions
    df_pred_a = predict_stream(model_a, X_test_a, ids_test_a, thresh_a)
    print(f"Stream A predictions: {len(df_pred_a)} rows")

    # Stream B Predictions
    df_pred_b = predict_stream(model_b, X_test_b, ids_test_b, thresh_b)
    print(f"Stream B predictions: {len(df_pred_b)} rows")

    # Combine Predictions
    df_combined = pd.concat([df_pred_a, df_pred_b], ignore_index=True)

    # --- 5. Generate Submission ---
    generate_submission_file(df_combined)
    print("Inference Pipeline Completed.")
