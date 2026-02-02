import os
import numpy as np
import pandas as pd
from library import config, utils, data_loader, models


def generate_test_features(load_cached: bool = True, limit_rows: int = None):
    """
    Generates features for the test set using the full feature engineering pipeline.
    Delegates to the data_loader and features library which handle caching and processing.

    Args:
        load_cached (bool): Whether to load features from cache if available.
        limit_rows (int, optional): If provided, limits the number of rows for debugging.

    Returns:
        pd.DataFrame: The test features dataframe.
    """
    # Load test data via DatasetBuilder (handles caching internally)
    # The underlying features.generate_features function ensures no gating is applied for 'test' split
    df_test = data_loader.DatasetBuilder().load_data("test", load_cached=load_cached)

    if limit_rows is not None:
        print(f"Limiting test features to {limit_rows} rows for debugging.")
        df_test = df_test.head(limit_rows)

    return df_test


def create_submission(load_cached_features: bool = True, limit_rows: int = None):
    """
    Generates the submission file using the trained Expert Ensemble and optimized threshold.

    Args:
        load_cached_features (bool): Whether to use cached test features.
        limit_rows (int, optional): Limits the number of rows for debugging.
    """
    print("=== Starting Submission Generation ===")

    # 1. Generate/Load Test Features
    df_test = generate_test_features(
        load_cached=load_cached_features, limit_rows=limit_rows
    )
    print(f"Test features shape: {df_test.shape}")

    # 2. Initialize Models
    # The handlers will load the model artifacts from disk upon first prediction
    lgbm_handler = models.LGBMHandler(model_name="expert_lgbm.joblib")
    xgb_handler = models.XGBHandler(model_name="expert_xgb.joblib")

    ensemble = models.EnsemblePredictor(lgbm_handler, xgb_handler)

    # 3. Load Optimized Threshold
    thresh_path = os.path.join(config.WORKING_DIR, "models", "best_threshold.npy")
    if os.path.exists(thresh_path):
        threshold = float(np.load(thresh_path)[0])
        print(f"Loaded optimized threshold: {threshold}")
    else:
        threshold = 0.5
        print(f"Threshold file not found at {thresh_path}. Using default: {threshold}")

    # 4. Generate Predictions
    print("Running inference...")
    # ensemble.predict calculates probabilities and applies the threshold
    predictions = ensemble.predict(df_test, threshold=threshold)

    # 5. Format Submission
    print("Formatting submission...")

    # Create a dataframe of predictions
    # We rely on 'contact_id' being present in the features
    pred_df = pd.DataFrame(
        {"contact_id": df_test["contact_id"], "contact": predictions}
    )

    # Load the sample submission to ensure correct row order and completeness
    sample_sub_path = os.path.join(config.INPUT_DIR, "sample_submission.csv")
    if os.path.exists(sample_sub_path):
        sample_sub = pd.read_csv(sample_sub_path)

        # Left merge onto sample submission to keep all required rows
        # This aligns our predictions with the required output format
        submission = sample_sub[["contact_id"]].merge(
            pred_df, on="contact_id", how="left"
        )

        # Fill missing values with 0 (No Contact)
        # Missing values occur if features were not generated for some plays (e.g. missing tracking data)
        # or if limit_rows was used.
        fill_count = submission["contact"].isna().sum()
        if fill_count > 0:
            print(
                f"Warning: {fill_count} rows were missing predictions. Filling with 0."
            )
            submission["contact"] = submission["contact"].fillna(0)

        submission["contact"] = submission["contact"].astype(int)

    else:
        # Fallback if sample_submission is missing (unlikely in comp env)
        print(
            "Warning: sample_submission.csv not found. Using generated predictions directly."
        )
        submission = pred_df

    # 6. Save Submission
    # Requirement: Save to ./submission/submission.csv
    output_dir = "./submission"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "submission.csv")

    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
    print(submission.head())

    return submission
