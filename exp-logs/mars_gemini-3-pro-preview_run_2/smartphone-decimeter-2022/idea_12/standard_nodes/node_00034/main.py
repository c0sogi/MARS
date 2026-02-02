import os
import sys
import numpy as np
import pandas as pd
import torch
import scipy.stats

# Import from provided libraries
from library.config import Config
from library.utils import seed_everything, meters_to_latlon, haversine_distance
from library.data_loader import get_dataloaders
from library.model import SkyContextualizedCNN
from library import trainer, inference


def main():
    # 1. Setup and Configuration Override
    print("Initializing Fast Baseline Run...")
    seed_everything(Config.RANDOM_STATE)

    # Override Config for a fast baseline run
    # We reduce epochs to ensure it finishes quickly.
    # The provided trainer uses Config.NUM_EPOCHS directly.
    Config.NUM_EPOCHS = 5
    print(f"Overridden NUM_EPOCHS to {Config.NUM_EPOCHS} for baseline speed.")

    # 2. Train the Model
    print("\n=== Starting Training Phase ===")
    # This will train and save the best model to Config.MODEL_PATH
    trainer.train_model(load_cached_data=True)

    # 3. Validation Assessment
    print("\n=== Starting Validation Phase ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for validation: {device}")

    # Load validation data
    # We need the loaders and the meta dataframe to map back to trips
    _, val_loader, _, val_meta, _ = get_dataloaders(load_cached_data=True)

    # Load the best model
    if not os.path.exists(Config.MODEL_PATH):
        print("Error: Model file not found. Training may have failed.")
        return

    model = SkyContextualizedCNN().to(device)
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    model.eval()

    # Run Inference on Validation Set
    val_preds_list = []
    val_targets_list = []

    # We also collect features for failure analysis
    # Feature indices: 6 = mean_cn0, 7 = mean_uncertainty (from Config.TRAJECTORY_FEATURES)
    feature_cn0_list = []
    feature_unc_list = []

    with torch.no_grad():
        for traj, sky, target in val_loader:
            traj = traj.to(device)
            sky = sky.to(device)

            # Forward pass
            output = model(traj, sky)

            val_preds_list.append(output.cpu().numpy())
            val_targets_list.append(target.numpy())

            # Extract features for analysis (taking the mean over the window or center)
            # traj shape: (Batch, Channels, Window)
            # We take the center value of the window for correlation analysis
            center_idx = Config.WINDOW_SIZE // 2
            # Channel 6: mean_cn0, Channel 7: mean_uncertainty
            feature_cn0_list.append(traj[:, 6, center_idx].cpu().numpy())
            feature_unc_list.append(traj[:, 7, center_idx].cpu().numpy())

    val_preds = np.concatenate(val_preds_list, axis=0)
    val_targets = np.concatenate(val_targets_list, axis=0)
    feat_cn0 = np.concatenate(feature_cn0_list, axis=0)
    feat_unc = np.concatenate(feature_unc_list, axis=0)

    # Reconstruct Coordinates
    # val_meta has 'wls_lat', 'wls_lon' aligned with the loader
    wls_lat = val_meta["wls_lat"].values
    wls_lon = val_meta["wls_lon"].values

    # Ground Truth Lat/Lon
    # The targets in the dataset are (d_east, d_north) relative to WLS
    gt_lat, gt_lon = meters_to_latlon(
        wls_lat, wls_lon, val_targets[:, 0], val_targets[:, 1]
    )

    # Predicted Lat/Lon
    pred_lat, pred_lon = meters_to_latlon(
        wls_lat, wls_lon, val_preds[:, 0], val_preds[:, 1]
    )

    # Calculate Distance Errors
    errors = haversine_distance(gt_lat, gt_lon, pred_lat, pred_lon)

    # 4. Compute Official Metric
    # Metric: Mean of (50th + 95th percentile) averaged for each phone (tripId)

    # Add errors and tripId to a dataframe for grouping
    eval_df = pd.DataFrame({"tripId": val_meta["tripId"], "error": errors})

    def compute_trip_metric(group):
        p50 = np.percentile(group["error"], 50)
        p95 = np.percentile(group["error"], 95)
        return (p50 + p95) / 2.0

    trip_scores = eval_df.groupby("tripId").apply(compute_trip_metric)
    final_metric = trip_scores.mean()

    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Calculate correlations
    # Note: Features are scaled, but correlation is scale-invariant (Pearson) or rank-based (Spearman)

    corr_cn0 = np.corrcoef(errors, feat_cn0)[0, 1]
    corr_unc = np.corrcoef(errors, feat_unc)[0, 1]

    print(f"Correlation between Error and Signal Strength (Cn0): {corr_cn0:.4f}")
    print(f"Correlation between Error and Signal Uncertainty: {corr_unc:.4f}")

    # 6. Submission Generation
    threshold = 4.256982128481356
    if final_metric < threshold:
        print(
            f"\nMetric ({final_metric}) is better than threshold ({threshold}). Generating submission..."
        )
        inference.generate_submission(load_cached_data=True)
    else:
        print(
            f"\nMetric ({final_metric}) did not meet threshold ({threshold}). Skipping submission generation."
        )


if __name__ == "__main__":
    main()
