import os
import numpy as np
import pandas as pd
import warnings
from library.data_loader import load_train_data, load_val_data, load_test_data
from library.model import PhysicsEnsembleModel
from library.utils import haversine
from library.config import FEATURE_COLS, SUBMISSION_PATH, SEED, WORKING_DIR

# Suppress warnings
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def calculate_metric(df):
    """
    Calculates the competition metric:
    Mean of the (Mean of 50th and 95th percentile distance errors per phone).
    """
    # Calculate distance error for each point
    df["dist_error"] = haversine(
        df["LatitudeDegrees"], df["LongitudeDegrees"], df["pred_lat"], df["pred_lon"]
    )

    # Group by phone_name (or tripId, but description says "For every phone")
    # Usually in this dataset tripId = drive_id + phone_name.
    # The metric is calculated per phone trace (trip).

    score_per_phone = []
    for phone, group in df.groupby("tripId"):
        errors = group["dist_error"].values
        p50 = np.percentile(errors, 50)
        p95 = np.percentile(errors, 95)
        score_per_phone.append((p50 + p95) / 2)

    final_metric = np.mean(score_per_phone)
    return final_metric


def run_failure_analysis(val_df, feature_cols):
    """
    Correlates prediction error with features to identify failure modes.
    """
    print("\n=== Failure Analysis ===")
    analysis_df = val_df.copy()

    # Calculate correlations
    correlations = {}
    for col in feature_cols:
        if col in analysis_df.columns:
            # Handle NaN values in features for correlation calculation
            valid_idx = ~analysis_df[col].isna() & ~analysis_df["dist_error"].isna()
            if valid_idx.sum() > 1:
                corr = np.corrcoef(
                    analysis_df.loc[valid_idx, col],
                    analysis_df.loc[valid_idx, "dist_error"],
                )[0, 1]
                correlations[col] = corr

    # Sort by absolute correlation
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

    print("Top Feature Correlations with Error Magnitude:")
    for feat, corr in sorted_corr[:10]:
        print(f"{feat:<30}: {corr:.4f}")


def main():
    set_seed(SEED)

    print("--- Starting Pipeline ---")

    # Cite debug_lesson_3: Invalidate Data Caches When Modifying Data Processing Logic
    # Explicitly remove cache files to ensure new data cleaning logic is applied
    for split in ["train", "val", "test"]:
        cache_path = os.path.join(WORKING_DIR, f"{split}_features.parquet")
        if os.path.exists(cache_path):
            os.remove(cache_path)
            print(f"Removed stale cache: {cache_path}")

    # 1. Load Data
    # Limit training data for fast baseline execution as required
    print("Loading Training Data...")
    train_df = load_train_data(load_cached_data=False, limit=50000)
    print(f"Training samples: {len(train_df)}")

    print("Loading Validation Data...")
    val_df = load_val_data(load_cached_data=False)
    print(f"Validation samples: {len(val_df)}")

    # 2. Train Model
    # We pass only train_df to model.train. The model class uses GroupKFold internally on the provided data.
    # We keep val_df as a strict hold-out.
    model = PhysicsEnsembleModel()
    model.train(train_df)

    # 3. Validation Inference
    print("\n--- Running Validation Inference ---")
    val_pred_east, val_pred_north = model.predict(val_df)

    # Reconstruct coordinates
    val_pred_lat, val_pred_lon = model.reconstruct_coords(
        val_df, val_pred_east, val_pred_north
    )

    # Add predictions to dataframe for metric calculation
    val_df["pred_lat"] = val_pred_lat
    val_df["pred_lon"] = val_pred_lon

    # 4. Calculate Metric
    metric = calculate_metric(val_df)
    print(f"Final Validation Metric: {metric}")

    # 5. Failure Analysis
    run_failure_analysis(val_df, FEATURE_COLS)

    # 6. Submission Generation
    THRESHOLD = 4.32379283550646
    if metric < THRESHOLD:
        print(
            f"\nMetric ({metric}) is below threshold ({THRESHOLD}). Generating submission..."
        )

        print("Loading Test Data...")
        test_df = load_test_data(load_cached_data=False)

        print("Predicting on Test Set...")
        test_pred_east, test_pred_north = model.predict(test_df)

        print("Reconstructing Test Coordinates...")
        test_pred_lat, test_pred_lon = model.reconstruct_coords(
            test_df, test_pred_east, test_pred_north
        )

        submission = pd.DataFrame(
            {
                "tripId": test_df["tripId"],
                "UnixTimeMillis": test_df["UnixTimeMillis"],
                "LatitudeDegrees": test_pred_lat,
                "LongitudeDegrees": test_pred_lon,
            }
        )

        print(f"Saving submission to {SUBMISSION_PATH}...")
        submission.to_csv(SUBMISSION_PATH, index=False)
        print("Submission saved successfully.")

    else:
        print(
            f"\nMetric ({metric}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
