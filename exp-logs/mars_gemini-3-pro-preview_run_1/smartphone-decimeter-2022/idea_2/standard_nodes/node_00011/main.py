import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from scipy.stats import pearsonr

# Import library modules
from library.config import Config
from library.model import SensorFusionTCN
from library.data_loader import SmartphoneDataset
from library.trainer import train_model
from library.inference import generate_submission


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371000  # Radius of earth in meters
    return c * r


def compute_metric(df):
    """
    Computes the competition metric: mean of the 50th and 95th percentile distance errors.
    """
    # Calculate distance error for each point
    df["dist_error"] = haversine_distance(
        df["LatitudeDegrees"], df["LongitudeDegrees"], df["Pred_Lat"], df["Pred_Lon"]
    )

    # Create unique trip identifier
    df["tripId"] = df["drive_id"] + "-" + df["phone_name"]

    score_list = []
    for trip, group in df.groupby("tripId"):
        errors = group["dist_error"].values
        p50 = np.percentile(errors, 50)
        p95 = np.percentile(errors, 95)
        score_list.append((p50 + p95) / 2)

    return np.mean(score_list)


def run_pipeline():
    # 1. Configuration Overrides
    print("Configuring parameters...")
    # Increasing epochs to ensure convergence (Cite solution_lesson_node_00003)
    Config.EPOCHS = 30
    # Using default model capacity (64 channels, 4 layers) as defined in Config
    Config.BATCH_SIZE = 128

    set_seed(Config.SEED)

    # 2. Training
    print("\n" + "=" * 40)
    print("STARTING TRAINING")
    print("=" * 40)
    model = train_model(
        train_meta_path=Config.TRAIN_METADATA_PATH,
        val_meta_path=Config.VAL_METADATA_PATH,
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE,
    )

    # 3. Validation and Metric Computation
    print("\n" + "=" * 40)
    print("VALIDATION & METRIC CALCULATION")
    print("=" * 40)

    # Load validation dataset
    val_dataset = SmartphoneDataset(
        Config.VAL_METADATA_PATH, Config.WINDOW_SIZE, mode="val"
    )
    val_loader = DataLoader(
        val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2
    )

    model.eval()
    all_residuals = []
    all_features = []  # For failure analysis

    with torch.no_grad():
        for features, targets in val_loader:
            features = features.to(Config.DEVICE)
            # Predict
            residuals = model(features).cpu().numpy()
            all_residuals.append(residuals)

            # Store features for analysis (taking the mean of the window or last step)
            # features shape: [Batch, Channels, Seq_Len]
            # We take the last time step features
            all_features.append(features[:, :, -1].cpu().numpy())

    residuals = np.concatenate(all_residuals, axis=0)
    features_array = np.concatenate(all_features, axis=0)

    # Reconstruct predictions
    # Get the dataframe rows corresponding to the valid windows
    val_indices = val_dataset.indices
    val_df_subset = val_dataset.full_df.iloc[val_indices].copy()

    # WLS Baseline from the dataframe
    wls_lat = val_df_subset["WlsLat"].values
    wls_lon = val_df_subset["WlsLon"].values

    # Predicted Lat/Lon
    pred_lat = wls_lat + residuals[:, 0]
    pred_lon = wls_lon + residuals[:, 1]

    val_df_subset["Pred_Lat"] = pred_lat
    val_df_subset["Pred_Lon"] = pred_lon

    # Compute Metric
    metric = compute_metric(val_df_subset)
    print(f"Final Validation Metric: {metric}")

    # 4. Failure Analysis
    print("\n" + "=" * 40)
    print("FAILURE ANALYSIS")
    print("=" * 40)

    # Calculate error magnitude
    errors = val_df_subset["dist_error"].values

    # Feature names from Config
    feature_names = Config.INPUT_FEATURES

    print("Correlation between Input Features and Error Magnitude:")
    correlations = {}
    for i, feat_name in enumerate(feature_names):
        # features_array is [N, Num_Features]
        feat_values = features_array[:, i]
        # Handle potential NaNs or constants
        if np.std(feat_values) == 0:
            corr = 0.0
        else:
            corr, _ = pearsonr(feat_values, errors)
        correlations[feat_name] = corr

    # Sort by absolute correlation
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    for name, corr in sorted_corr:
        print(f"  {name}: {corr:.4f}")

    # 5. Submission
    print("\n" + "=" * 40)
    print("SUBMISSION GENERATION")
    print("=" * 40)

    THRESHOLD = 3.8442371867640412
    if metric < THRESHOLD:
        print(
            f"Metric ({metric}) is below threshold ({THRESHOLD}). Generating submission..."
        )
        generate_submission(
            test_meta_path=Config.TEST_METADATA_PATH,
            model_weights_path=os.path.join(Config.WORKING_DIR, "model_weights.pth"),
            output_path=os.path.join(Config.SUBMISSION_DIR, "submission.csv"),
            batch_size=Config.BATCH_SIZE,
            device=Config.DEVICE,
        )
    else:
        print(
            f"Metric ({metric}) is NOT below threshold ({THRESHOLD}). Skipping submission generation."
        )


if __name__ == "__main__":
    run_pipeline()
