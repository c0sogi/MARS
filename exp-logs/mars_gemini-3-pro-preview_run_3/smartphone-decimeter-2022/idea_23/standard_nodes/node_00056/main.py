import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings

# Import from provided libraries
from library.config import SEED, WORKING_DIR, SUBMISSION_DIR
from library.data_loader import GnssLoader
from library.feature_eng import FeatureEngine
from library.kinematics import KinematicsEngine
from library.model_wrapper import LGBMEnsemble
from library.graph_optimizer import GraphOptimizer, generate_submission

# Set random seeds for reproducibility
np.random.seed(SEED)
warnings.filterwarnings("ignore")


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculates the Haversine distance between two sets of latitude/longitude coordinates.
    """
    R = 6371000  # Radius of Earth in meters
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c


def calculate_validation_metric(df):
    """
    Computes the competition metric: Mean of (50th + 95th percentile errors) averaged over phones.
    """
    # Extract phone name from tripId (format: drive_id-phone_name)
    # Assuming standard format where phone name is the last part after the last hyphen
    # However, tripId structure might vary slightly, but usually ends with phone name.
    # Based on sample: 2020-06-04-US-MTV-1-GooglePixel4 -> GooglePixel4
    df["phone_name"] = df["tripId"].apply(lambda x: x.split("-")[-1])

    phone_scores = []
    unique_phones = df["phone_name"].unique()

    for phone in unique_phones:
        group = df[df["phone_name"] == phone]
        if len(group) == 0:
            continue

        errors = group["error"].values
        p50 = np.percentile(errors, 50)
        p95 = np.percentile(errors, 95)

        avg_score = (p50 + p95) / 2
        phone_scores.append(avg_score)

    if not phone_scores:
        return 0.0

    return np.mean(phone_scores)


def main():
    print("Starting End-to-End Pipeline...")

    # -------------------------------------------------------------------------
    # 1. Feature Engineering (Train)
    # -------------------------------------------------------------------------
    print("\n[Step 1] Generating Training Features...")
    fe = FeatureEngine()
    # load_cached_data=True will use pre-computed parquet files if available in ./working
    train_df = fe.create_features(split="train", load_cached_data=True)

    if train_df.empty:
        print("Error: No training data generated.")
        return

    # -------------------------------------------------------------------------
    # 2. Model Training
    # -------------------------------------------------------------------------
    print("\n[Step 2] Training LightGBM Ensemble...")
    model = LGBMEnsemble()
    model.fit(train_df, load_cached_data=True)

    # -------------------------------------------------------------------------
    # 3. Validation Pipeline
    # -------------------------------------------------------------------------
    print("\n[Step 3] Running Validation...")

    # Generate Validation Features
    val_df = fe.create_features(split="val", load_cached_data=True)

    if val_df.empty:
        print("Error: No validation data generated.")
        return

    # Predict Anchors (ML Step)
    print("Predicting Validation Anchors...")
    pred_e, pred_n = model.predict(val_df)
    val_df["Pred_E"] = pred_e
    val_df["Pred_N"] = pred_n

    # Prepare for Graph Optimization
    kin_engine = KinematicsEngine()
    optimizer = GraphOptimizer()
    loader = GnssLoader()

    # Load validation metadata to identify trips and ground truth paths
    val_metadata = loader.load_metadata(split="val")
    unique_val_trips = val_metadata[
        ["tripId", "drive_id", "phone_name"]
    ].drop_duplicates()

    val_results = []

    print(f"Optimizing {len(unique_val_trips)} validation trips...")

    for _, row in tqdm(
        unique_val_trips.iterrows(),
        total=len(unique_val_trips),
        desc="Val Optimization",
    ):
        trip_id = row["tripId"]
        drive_id = row["drive_id"]
        phone_name = row["phone_name"]

        # Get ML Anchors for this trip
        trip_anchors = val_df[val_df["tripId"] == trip_id].copy()

        if trip_anchors.empty:
            continue

        try:
            # Load Raw GNSS for Kinematics
            gnss_df, _, _ = loader.get_drive_data(
                drive_id, phone_name, split="val", load_cached_data=True
            )

            # Compute Kinematics (Stream B)
            kin_df = kin_engine.compute_displacements(
                gnss_df, drive_id, phone_name, load_cached_data=True
            )

            # Run Graph Optimizer
            opt_res = optimizer.solve_trajectory(
                drive_id, phone_name, trip_anchors, kin_df, load_cached_data=True
            )

            if not opt_res.empty:
                val_results.append(opt_res)

        except Exception as e:
            print(f"Error optimizing validation trip {trip_id}: {e}")

    if not val_results:
        print("Error: No validation results produced.")
        return

    val_pred_df = pd.concat(val_results, ignore_index=True)

    # -------------------------------------------------------------------------
    # 4. Metric Calculation
    # -------------------------------------------------------------------------
    print("\n[Step 4] Calculating Validation Metric...")

    # We need Ground Truth Lat/Lon for the validation timestamps
    # The metadata file contains the GT Lat/Lon
    gt_df = val_metadata[
        ["tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]
    ].copy()
    gt_df.rename(
        columns={"LatitudeDegrees": "lat_gt", "LongitudeDegrees": "lon_gt"},
        inplace=True,
    )

    # Merge Predictions with Ground Truth
    # Use inner join to ensure we only evaluate on timestamps where we have GT
    eval_df = val_pred_df.merge(gt_df, on=["tripId", "UnixTimeMillis"], how="inner")

    # Calculate Distance Error
    eval_df["error"] = haversine_distance(
        eval_df["LatitudeDegrees"],
        eval_df["LongitudeDegrees"],
        eval_df["lat_gt"],
        eval_df["lon_gt"],
    )

    # Compute Metric
    final_metric = calculate_validation_metric(eval_df)
    print(f"Final Validation Metric: {final_metric}")

    # -------------------------------------------------------------------------
    # 5. Failure Analysis
    # -------------------------------------------------------------------------
    print("\n[Step 5] Performing Failure Analysis...")

    # Merge features back to evaluation dataframe to correlate error with features
    # val_df has the features and predictions
    analysis_df = eval_df.merge(
        val_df, on=["tripId", "UnixTimeMillis"], how="inner", suffixes=("", "_feat")
    )

    # Features to analyze
    feature_cols = model.feature_cols

    correlations = {}
    print(f"{'Feature':<20} | {'Correlation with Error':<25}")
    print("-" * 50)

    for col in feature_cols:
        if col in analysis_df.columns:
            corr = analysis_df["error"].corr(analysis_df[col])
            correlations[col] = corr

    # Sort by absolute correlation
    for feat, corr in sorted(
        correlations.items(), key=lambda x: abs(x[1]), reverse=True
    ):
        print(f"{feat:<20} | {corr:<25.4f}")

    # -------------------------------------------------------------------------
    # 6. Submission Generation
    # -------------------------------------------------------------------------
    THRESHOLD = 4.160290813847215

    if final_metric < THRESHOLD:
        print(
            f"\n[Step 6] Metric ({final_metric}) is better than threshold ({THRESHOLD}). Generating Submission..."
        )

        # Generate Test Features
        print("Generating Test Features...")
        test_df = fe.create_features(split="test", load_cached_data=True)

        # Load Test Metadata for trip list
        test_metadata = loader.load_metadata(split="test")

        # Run generation pipeline
        # Note: generate_submission handles prediction, kinematics, optimization, and formatting
        generate_submission(
            test_metadata=test_metadata,
            feature_df=test_df,
            kinematics_engine=kin_engine,
            model_wrapper=model,
            output_path=os.path.join(SUBMISSION_DIR, "submission.csv"),
        )
    else:
        print(
            f"\n[Step 6] Metric ({final_metric}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
