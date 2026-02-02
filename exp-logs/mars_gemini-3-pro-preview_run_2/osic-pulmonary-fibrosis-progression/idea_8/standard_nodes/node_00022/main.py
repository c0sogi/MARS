import os
import sys
import numpy as np
import pandas as pd
import warnings

# Import from provided libraries
from library.config import SEED, SUBMISSION_PATH
from library.feature_pipeline import run_feature_pipeline
from library.model_wrapper import StratifiedQuantileGLM, calculate_laplace_metric

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def seed_everything(seed=42):
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def run():
    # 1. Setup
    seed_everything(SEED)

    # 2. Load Data
    # load_cached_data=True ensures we use the pre-computed features from ./working
    print("Loading data...")
    train_data, val_data, test_data = run_feature_pipeline(load_cached_data=True)

    X_fvc_train, y_fvc_train, X_unc_train, df_train = train_data
    X_fvc_val, y_fvc_val, X_unc_val, df_val = val_data
    X_fvc_test, _, X_unc_test, df_test = test_data

    # 3. Model Training
    # The dataset size is small (~1500), so we use the full set.
    # The 'highs' solver in QuantileRegressor is efficient for this scale.
    print("Training StratifiedQuantileGLM...")
    model = StratifiedQuantileGLM(quantile=0.5, max_iter=1000)
    model.fit(X_fvc_train, y_fvc_train, X_unc_train)

    # 4. Validation
    print("Validating...")
    val_fvc_pred, val_sigma_pred = model.predict(X_fvc_val, X_unc_val)

    # Calculate Metric
    score = calculate_laplace_metric(y_fvc_val, val_fvc_pred, val_sigma_pred)
    print(f"Final Validation Metric: {score}")

    # 5. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Calculate absolute error
    abs_error = np.abs(y_fvc_val - val_fvc_pred)

    # Create analysis dataframe
    analysis_df = df_val.copy()
    analysis_df["AbsError"] = abs_error
    analysis_df["PredFVC"] = val_fvc_pred
    analysis_df["PredSigma"] = val_sigma_pred

    # Select numerical columns for correlation
    # We focus on clinical metadata to understand systematic bias
    corr_cols = ["Age", "Weeks", "Baseline_FVC", "Baseline_Percent", "AbsError"]
    # Filter to existing columns
    corr_cols = [c for c in corr_cols if c in analysis_df.columns]

    correlations = (
        analysis_df[corr_cols].corr()["AbsError"].sort_values(ascending=False)
    )
    print("Correlation of features with Absolute Error:")
    print(correlations.drop("AbsError"))

    # 6. Submission
    THRESHOLD = -6.805292148096688

    if score > THRESHOLD:
        print(
            f"\nMetric ({score}) meets threshold ({THRESHOLD}). Generating submission..."
        )

        # Predict on Test Set
        test_fvc_pred, test_sigma_pred = model.predict(X_fvc_test, X_unc_test)

        # Format Submission
        submission = pd.DataFrame(
            {
                "Patient_Week": df_test["Patient_Week"],
                "FVC": test_fvc_pred,
                "Confidence": test_sigma_pred,
            }
        )

        # Round to integers as per sample submission
        submission["FVC"] = submission["FVC"].round().astype(int)
        submission["Confidence"] = submission["Confidence"].round().astype(int)

        # Save
        os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)
        submission.to_csv(SUBMISSION_PATH, index=False)
        print(f"Submission saved to {SUBMISSION_PATH}")

    else:
        print(
            f"\nMetric ({score}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    run()
