import sys
import os
import pandas as pd
import numpy as np
import warnings

# Ensure the library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything
from library.data_manager import DataManager
from library.linear_models import DualModel, generate_submission

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    # 1. Setup and Reproducibility
    seed_everything(Config.SEED)
    print("=== Dual-Moment Axial-Quantile Pipeline Started ===")

    # 2. Data Preparation
    # Initialize DataManager and load data
    # load_cached_data=True ensures we use pre-computed features if available to save time
    dm = DataManager()
    print("\n[Step 1] Preparing Data...")
    train_data, val_data, test_data = dm.prepare_data(load_cached_data=True)

    # 3. Model Training
    # Initialize the DualModel (Quantile Regressor + ElasticNet)
    model = DualModel()
    print("\n[Step 2] Training Model...")
    model.fit(train_data, val_data)

    # 4. Validation & Metrics
    print("\n[Step 3] Validating Performance...")
    val_score = model.evaluate(val_data)
    # REQUIRED FORMAT: Print the full precision validation metric
    print(f"Final Validation Metric: {val_score}")

    # 5. Failure Analysis
    print("\n[Step 4] Performing Failure Analysis...")
    # Load validation metadata to correlate errors with original features
    df_val_meta = pd.read_csv(Config.VAL_METADATA_PATH)

    # Generate predictions on validation set
    y_true = val_data["y"]
    y_pred, sigma_pred = model.predict(val_data)

    # Calculate metrics for analysis
    abs_error = np.abs(y_true - y_pred)

    # Add analysis columns to a temporary dataframe
    # Note: df_val_meta aligns with val_data because DataManager processes them sequentially
    analysis_df = df_val_meta.copy()
    analysis_df["AbsError"] = abs_error
    analysis_df["PredFVC"] = y_pred
    analysis_df["PredSigma"] = sigma_pred

    # Calculate correlations
    # We focus on numerical clinical features and the target
    features_to_analyze = ["Weeks", "Age", "Percent", "FVC", "PredSigma"]
    # Ensure columns exist before correlation
    existing_features = [c for c in features_to_analyze if c in analysis_df.columns]

    if existing_features:
        correlations = analysis_df[existing_features + ["AbsError"]].corr()["AbsError"]
        print("\nCorrelation between Absolute Error and Features:")
        print(correlations.sort_values(ascending=False))

        # specific insight print
        max_corr_feat = correlations.drop("AbsError").idxmax()
        max_corr_val = correlations.drop("AbsError").max()
        print(
            f"\nInsight: Error is most positively correlated with {max_corr_feat} ({max_corr_val:.4f})"
        )
    else:
        print("Could not perform correlation analysis due to missing columns.")

    # 6. Submission Generation
    # Threshold defined in task description
    THRESHOLD = -6.805292148096688

    print("\n[Step 5] Checking Submission Criteria...")
    if val_score > THRESHOLD:
        print(f"Validation Score ({val_score}) exceeds threshold ({THRESHOLD}).")
        generate_submission(model, test_data)
    else:
        print(
            f"Validation Score ({val_score}) does not exceed threshold ({THRESHOLD}). Submission skipped."
        )

    print("\n=== Pipeline Complete ===")


if __name__ == "__main__":
    main()
