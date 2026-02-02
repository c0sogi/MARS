import os
import numpy as np
import pandas as pd
import warnings
from library.feature_builder import MultiHypothesisFeaturizer
from library.model_engine import ResidualLGBMEnsemble
from library.utils import haversine_distance

# Suppress warnings
warnings.filterwarnings("ignore")

# Configuration
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_15"
SUBMISSION_DIR = "./submission"
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
VAL_PREDS_PATH = os.path.join(WORKING_DIR, "val_predictions.csv")
SEED = 42
METRIC_THRESHOLD = 4.202107392205921

# Set random seeds
np.random.seed(SEED)


def calculate_competition_metric(df_merged):
    """
    Calculates the competition metric: Mean of the 50th and 95th percentile distance errors,
    averaged across all phones (trips).
    """
    score_list = []
    # Group by tripId (representing a unique phone-drive)
    for trip, group in df_merged.groupby("tripId"):
        errors = group["dist_error"].values
        p50 = np.percentile(errors, 50)
        p95 = np.percentile(errors, 95)
        score_list.append((p50 + p95) / 2)

    if not score_list:
        return 0.0

    return np.mean(score_list)


def run_pipeline():
    print("Initializing Multi-Hypothesis Newton-Boost Pipeline...")

    # 1. Initialize Featurizer and Load Data
    # We use sample_frac=1.0 because the dataset (~200k rows) is small enough for fast LGBM training
    featurizer = MultiHypothesisFeaturizer(
        input_dir=INPUT_DIR, metadata_dir=METADATA_DIR, cache_dir=WORKING_DIR
    )

    # Load Training and Validation Data
    # load_cached_data=True will look for parquets in working/idea_15/
    train_df, val_df = featurizer.get_train_data(load_cached_data=True, sample_frac=1.0)

    feature_cols = featurizer.get_feature_names()
    target_cols = featurizer.get_target_names()

    print(f"Features ({len(feature_cols)}): {feature_cols}")
    print(f"Targets: {target_cols}")

    # 2. Initialize Model Engine
    model_engine = ResidualLGBMEnsemble(
        output_dir=os.path.join(WORKING_DIR, "models"), n_folds=5, seed=SEED
    )

    # 3. Train Models
    # Using conservative parameters for robust baseline
    lgbm_params = {
        "objective": "regression_l1",  # MAE Loss for robustness against outliers
        "metric": "mae",
        "boosting_type": "gbdt",
        "learning_rate": 0.1,
        "num_leaves": 31,
        "max_depth": -1,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "n_jobs": -1,
        "verbose": -1,
        "seed": SEED,
    }

    print("\n--- Starting Training ---")
    # Train on ENU residuals
    cv_scores = model_engine.train_group_kfold(
        train_df,
        feature_cols,
        target_cols,
        params=lgbm_params,
        n_estimators=1000,
        early_stopping_rounds=50,
    )

    # 4. Validation Inference
    print("\n--- Running Validation Inference ---")
    # Generate predictions for the validation set using the ensemble
    # This converts predicted ENU residuals back to Lat/Lon
    val_preds_df = model_engine.predict_ensemble(
        val_df, feature_cols, target_cols, submission_path=VAL_PREDS_PATH
    )

    # 5. Metric Calculation
    # Merge predictions with Ground Truth
    # val_df contains the GT LatitudeDegrees and LongitudeDegrees
    val_eval = val_df[
        ["tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]
    ].copy()
    val_eval.rename(
        columns={"LatitudeDegrees": "lat_gt", "LongitudeDegrees": "lon_gt"},
        inplace=True,
    )

    # Merge predictions (ensure alignment)
    val_eval = val_eval.merge(
        val_preds_df[
            ["tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]
        ],
        on=["tripId", "UnixTimeMillis"],
        how="left",
    )
    val_eval.rename(
        columns={"LatitudeDegrees": "lat_pred", "LongitudeDegrees": "lon_pred"},
        inplace=True,
    )

    # Calculate Haversine Distance
    val_eval["dist_error"] = haversine_distance(
        val_eval["lat_gt"],
        val_eval["lon_gt"],
        val_eval["lat_pred"],
        val_eval["lon_pred"],
    )

    final_metric = calculate_competition_metric(val_eval)
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Correlate error magnitude with features
    # Add error column to features dataframe for correlation
    analysis_df = val_df[feature_cols].copy()
    analysis_df["error_magnitude"] = val_eval["dist_error"].values

    correlations = (
        analysis_df.corr()["error_magnitude"]
        .drop("error_magnitude")
        .sort_values(ascending=False)
    )

    print("Top 5 Features correlated with Error (Positive):")
    print(correlations.head(5))
    print("\nTop 5 Features correlated with Error (Negative):")
    print(correlations.tail(5))

    # 7. Test Inference & Submission
    if final_metric < METRIC_THRESHOLD:
        print(f"\nMetric {final_metric} < {METRIC_THRESHOLD}. Generating submission...")

        # Load Test Data
        test_df = featurizer.get_test_data(load_cached_data=True)

        # Predict
        model_engine.predict_ensemble(
            test_df, feature_cols, target_cols, submission_path=SUBMISSION_PATH
        )
        print("Submission generation complete.")
    else:
        print(f"\nMetric {final_metric} >= {METRIC_THRESHOLD}. Skipping submission.")


if __name__ == "__main__":
    run_pipeline()
