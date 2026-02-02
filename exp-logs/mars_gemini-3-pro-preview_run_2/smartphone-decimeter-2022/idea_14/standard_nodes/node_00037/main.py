import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from scipy.stats import pearsonr

from library.config import Config
from library.utils import set_seed, haversine_distance, meters_to_latlon
from library.data_loader import load_data, GNSSWindowDataset
from library.model import SkyStateTransformer
from library.train import run_training
from library.inference import generate_predictions


def calculate_competition_metric(df_results):
    """
    Calculates the competition metric: mean of the 50th and 95th percentile distance errors,
    averaged for each phone, then averaged across all phones.
    """
    # Group by phone
    phone_groups = df_results.groupby("phone_name")

    phone_scores = []
    for _, group in phone_groups:
        errors = group["ErrorMeters"].values
        p50 = np.percentile(errors, 50)
        p95 = np.percentile(errors, 95)
        phone_score = (p50 + p95) / 2
        phone_scores.append(phone_score)

    final_score = np.mean(phone_scores)
    return final_score


def run_validation_analysis(model, val_loader, val_meta, device):
    """
    Runs inference on validation set, computes metrics, and performs failure analysis.
    """
    model.eval()
    all_preds = []

    # Run inference
    with torch.no_grad():
        for batch_seq, batch_sky, _ in val_loader:
            batch_seq = batch_seq.to(device)
            batch_sky = batch_sky.to(device)

            outputs = model(batch_seq, batch_sky)
            all_preds.append(outputs.cpu().numpy())

    predictions_meters = np.concatenate(all_preds, axis=0)

    # Load Ground Truth Metadata
    # Fix for KeyError: 'LatitudeDegrees' and missing 'phone_name'
    # Cite debug_lesson_6
    val_gt_df = pd.read_csv(Config.VAL_METADATA_PATH)

    # Merge prediction metadata with ground truth
    # val_meta contains the specific epochs we predicted for
    val_merged = pd.merge(
        val_meta,
        val_gt_df[
            [
                "tripId",
                "UnixTimeMillis",
                "LatitudeDegrees",
                "LongitudeDegrees",
                "phone_name",
            ]
        ],
        on=["tripId", "UnixTimeMillis"],
        how="left",
    )

    # Reconstruction
    wls_lat = val_merged["WlsLat"].values
    wls_lon = val_merged["WlsLon"].values
    gt_lat = val_merged["LatitudeDegrees"].values
    gt_lon = val_merged["LongitudeDegrees"].values

    delta_east = predictions_meters[:, 0]
    delta_north = predictions_meters[:, 1]

    pred_lat, pred_lon = meters_to_latlon(delta_north, delta_east, wls_lat, wls_lon)

    # Calculate errors
    errors = haversine_distance(pred_lat, pred_lon, gt_lat, gt_lon)

    # Create results dataframe
    val_results = val_merged.copy()
    val_results["ErrorMeters"] = errors

    # Calculate Metric
    metric = calculate_competition_metric(val_results)
    print(f"Final Validation Metric: {metric}")

    # Failure Analysis
    print("\nFailure Analysis (Correlation with Error Magnitude):")
    # We need to access the input features corresponding to these predictions
    # Since we used a loader, we can't easily map back to the original X array indices
    # without reloading or passing indices.
    # However, we have val_meta which aligns with the predictions if shuffle=False.
    # We can use the Sky features from the dataset directly.

    # Access the underlying dataset arrays
    X_sky = val_loader.dataset.X_sky.numpy()

    # Config.SKY_FEATURES lists the feature names
    feature_names = Config.SKY_FEATURES

    correlations = {}
    for i, feature_name in enumerate(feature_names):
        feat_values = X_sky[:, i]
        # Handle potential NaNs or constants
        if np.std(feat_values) > 0:
            corr, _ = pearsonr(feat_values, errors)
            correlations[feature_name] = corr
        else:
            correlations[feature_name] = 0.0

    # Sort and print
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    for name, corr in sorted_corr:
        print(f"  {name}: {corr:.4f}")

    return metric


def main():
    # 1. Setup
    set_seed(Config.RANDOM_STATE)
    device = torch.device(Config.DEVICE)

    # 2. Train Model
    # We limit epochs to ensure fast execution within the time limit.
    # The full config specifies 50, but we'll use 15 for this baseline run.
    print("Starting training phase...")
    run_training(epochs=15, batch_size=Config.BATCH_SIZE, load_cached=True)

    # 3. Load Validation Data
    print("\nLoading validation data for assessment...")
    (_, val_data, _) = load_data(load_cached_data=True)
    val_X_seq, val_X_sky, val_y, val_meta = val_data

    val_dataset = GNSSWindowDataset(val_X_seq, val_X_sky, val_y)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE * 2,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # 4. Load Trained Model
    model = SkyStateTransformer().to(device)
    if os.path.exists(Config.MODEL_PATH):
        model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    else:
        print(
            "Warning: Model checkpoint not found. Using untrained model (this should not happen)."
        )

    # 5. Validation & Failure Analysis
    metric = run_validation_analysis(model, val_loader, val_meta, device)

    # 6. Submission
    THRESHOLD = 4.256982128481356
    if metric < THRESHOLD:
        print(
            f"\nValidation metric ({metric}) is better than threshold ({THRESHOLD}). Generating submission..."
        )
        generate_predictions(load_cached_data=True)
    else:
        print(
            f"\nValidation metric ({metric}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
