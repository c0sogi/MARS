import os
import numpy as np
import pandas as pd
from library import config, utils, feature_engineering, models


def run_inference(model=None, threshold=None, load_cached_data=True):
    """
    Executes the inference pipeline:
    1. Loads trained Dual-Ensemble model and optimized threshold.
    2. Generates/Loads test features.
    3. Predicts contact probabilities.
    4. Merges with sample_submission to handle gated rows.
    5. Saves final submission.

    Args:
        model (DualEnsemble, optional): Pre-loaded model instance. If None, loads from disk.
        threshold (float, optional): Decision threshold. If None, loads from disk.
        load_cached_data (bool): Whether to use cached feature files.
    """
    # 1. Setup Logging
    log_path = os.path.join(config.WORKING_DIR, "inference.log")
    utils.setup_logging(log_path)
    print("Starting Inference Pipeline...")

    # 2. Load Model
    if model is None:
        print("Initializing and loading Dual Ensemble model...")
        model = models.DualEnsemble()
        model.load()
    else:
        print("Using provided model instance.")

    # 3. Load Threshold
    if threshold is None:
        thresh_path = os.path.join(config.MODEL_DIR, "best_threshold.npy")
        if os.path.exists(thresh_path):
            threshold = float(np.load(thresh_path)[0])
            print(f"Loaded optimized threshold: {threshold}")
        else:
            print("Warning: Threshold file not found. Defaulting to 0.5")
            threshold = 0.5
    else:
        print(f"Using provided threshold: {threshold}")

    # 4. Generate/Load Test Features
    print("Generating test features...")
    # This handles caching internally via the decorator in feature_engineering
    df_test = feature_engineering.generate_test_features(
        load_cached_data=load_cached_data
    )

    # 5. Prepare Data for Prediction
    # Define metadata columns to exclude from features
    meta_cols = [
        "contact_id",
        "game_play",
        "step",
        "nfl_player_id_1",
        "nfl_player_id_2",
        "contact",
    ]

    # Select feature columns
    feature_cols = [c for c in df_test.columns if c not in meta_cols]

    if not feature_cols:
        raise ValueError("No feature columns found in test dataframe.")

    print(f"Test Data Shape: {df_test.shape}")
    print(f"Feature Count: {len(feature_cols)}")

    X_test = df_test[feature_cols]
    contact_ids = df_test["contact_id"]

    # 6. Generate Predictions
    print("Predicting probabilities...")
    probs = model.predict(X_test)

    # Apply threshold
    preds = (probs >= threshold).astype(int)

    # Create prediction dataframe
    pred_df = pd.DataFrame({"contact_id": contact_ids, "contact": preds})

    # 7. Merge with Sample Submission
    # The preprocessing pipeline applies gating (filtering out distant pairs).
    # We must ensure the final submission contains ALL rows from sample_submission.csv.
    # Rows missing from our predictions are those that were gated out; we predict 0 for them.

    print(f"Loading sample submission from {config.SAMPLE_SUBMISSION_PATH}...")
    sample_sub = pd.read_csv(config.SAMPLE_SUBMISSION_PATH)
    initial_len = len(sample_sub)

    print("Merging predictions...")
    # Left merge ensures we keep all IDs from sample_submission
    final_sub = sample_sub[["contact_id"]].merge(pred_df, on="contact_id", how="left")

    # Fill missing predictions (gated rows) with 0
    fill_count = final_sub["contact"].isna().sum()
    if fill_count > 0:
        print(f"Filled {fill_count} gated rows with 0 (No Contact).")
        final_sub["contact"] = final_sub["contact"].fillna(0)

    # Ensure integer type
    final_sub["contact"] = final_sub["contact"].astype(int)

    # Validation
    if len(final_sub) != initial_len:
        raise RuntimeError(
            f"Submission length mismatch! Expected {initial_len}, got {len(final_sub)}"
        )

    # 8. Save Submission
    print(f"Saving submission to {config.SUBMISSION_FILE}...")
    final_sub.to_csv(config.SUBMISSION_FILE, index=False)

    print("Inference Complete.")
