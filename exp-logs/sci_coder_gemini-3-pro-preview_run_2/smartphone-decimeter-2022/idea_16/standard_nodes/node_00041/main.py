import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from scipy.stats import spearmanr

# Import from library files
from library import config
from library import data_loader
from library import model as model_lib
from library import trainer
from library import utils
from library import inference


def main():
    # 1. Setup
    print("Setting up execution environment...")
    config.set_seed(config.RANDOM_STATE)

    # 2. Training
    # We use a limited number of epochs to ensure it fits within the time limit,
    # but enough to converge for this MLP architecture.
    print("\n=== Starting Training Phase ===")
    # run_training handles loading data, training loop, and saving best_model.pth
    scaler = trainer.run_training(max_epochs=15, load_cached=True)

    # 3. Validation Assessment
    print("\n=== Starting Validation Phase ===")

    # Load validation data
    print("Loading validation dataset...")
    val_dataset, _ = data_loader.load_dataset(
        mode="val", scaler=scaler, load_cached_data=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
    )

    # Load best model
    print("Loading best model for validation...")
    model = model_lib.RelativeWindowedMLP().to(config.DEVICE)
    best_model_path = os.path.join(config.CACHE_DIR, "best_model.pth")

    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {best_model_path}")

    model.load_state_dict(torch.load(best_model_path, map_location=config.DEVICE))
    model.eval()

    # Run Inference on Validation Set
    print("Running inference on validation set...")
    preds_residuals = []

    with torch.no_grad():
        for batch in val_loader:
            traj = batch["traj_feat"].to(config.DEVICE)
            sky = batch["sky_feat"].to(config.DEVICE)

            outputs = model(traj, sky)
            preds_residuals.append(outputs.cpu().numpy())

    pred_residuals_np = np.concatenate(preds_residuals, axis=0)

    # Reconstruct Absolute Coordinates
    val_meta = val_dataset.meta  # [trip_id, timestamp, wls_lat, wls_lon]
    pred_lats = []
    pred_lons = []

    for i in range(len(val_meta)):
        wls_lat = val_meta[i, 2]
        wls_lon = val_meta[i, 3]

        dx = pred_residuals_np[i, 0]
        dy = pred_residuals_np[i, 1]

        lat, lon = utils.meters_to_wgs84_relative(wls_lat, wls_lon, dx, dy)
        pred_lats.append(lat)
        pred_lons.append(lon)

    # Create Prediction DataFrame
    val_preds_df = pd.DataFrame(
        {
            "tripId": val_meta[:, 0],
            "UnixTimeMillis": val_meta[:, 1],
            "LatitudeDegrees": pred_lats,
            "LongitudeDegrees": pred_lons,
        }
    )
    # Ensure types match for merge
    val_preds_df["UnixTimeMillis"] = val_preds_df["UnixTimeMillis"].astype(np.int64)

    # Load Ground Truth
    print("Loading validation ground truth...")
    val_gt_df = pd.read_csv(config.VAL_METADATA_PATH)

    # Calculate Metric
    print("Calculating validation score...")
    score = utils.calculate_score(val_preds_df, val_gt_df)

    print(f"Final Validation Metric: {score}")

    # 4. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Merge predictions with GT to get errors
    analysis_df = pd.merge(
        val_gt_df,
        val_preds_df,
        on=["tripId", "UnixTimeMillis"],
        suffixes=("_gt", "_pred"),
    )

    # Calculate error magnitude
    analysis_df["error_m"] = utils.haversine_distance(
        analysis_df["LatitudeDegrees_gt"],
        analysis_df["LongitudeDegrees_gt"],
        analysis_df["LatitudeDegrees_pred"],
        analysis_df["LongitudeDegrees_pred"],
    )

    # To correlate with features, we need to align the dataset features with the analysis_df
    # The val_dataset is ordered same as val_meta, which was used to create val_preds_df.
    # So we can just concatenate features.

    # Extract some key features from the dataset tensors (already scaled)
    # We need to know which index corresponds to which feature.
    # From config.py:
    # TRAJ_FEATURES indices (per step): 9=mean_cn0, 10=mean_pr_unc, 11=mean_sv_time_unc
    # The input is flattened window. The center epoch features are at index:
    # (WINDOW_CENTER_IDX * NUM_TRAJ_FEATURES) + feature_offset

    center_step_idx = config.WINDOW_CENTER_IDX
    num_traj_feats = config.NUM_TRAJ_FEATURES
    base_idx = center_step_idx * num_traj_feats

    # Extract center epoch features from the validation dataset tensor
    # Note: These are scaled values. Correlation works fine with scaled values (rank order preserved for Spearman).
    traj_feats_np = val_dataset.traj_feats.numpy()
    sky_feats_np = val_dataset.sky_feats.numpy()

    # Feature indices
    idx_cn0 = base_idx + 9
    idx_pr_unc = base_idx + 10

    # Sky features (global for window)
    # SKY_FEATURES = [mean_elev, std_elev, mean_azim, std_azim, mean_cn0_sky, std_cn0_sky, sat_count_mean]
    idx_sat_count = 6

    # Add to analysis dataframe (assuming order is preserved, which it is)
    analysis_df["feat_cn0"] = traj_feats_np[:, idx_cn0]
    analysis_df["feat_pr_unc"] = traj_feats_np[:, idx_pr_unc]
    analysis_df["feat_sat_count"] = sky_feats_np[:, idx_sat_count]

    # Compute correlations
    correlations = {}
    for feat in ["feat_cn0", "feat_pr_unc", "feat_sat_count"]:
        corr, _ = spearmanr(analysis_df["error_m"], analysis_df[feat])
        correlations[feat] = corr

    print("Spearman Correlation with Error Magnitude:")
    for feat, corr in correlations.items():
        print(f"  {feat}: {corr:.4f}")

    # 5. Submission Generation
    print("\n=== Submission Generation ===")
    THRESHOLD = 4.256982128481356

    if score < THRESHOLD:
        print(
            f"Validation score ({score:.4f}) is better than threshold ({THRESHOLD:.4f}). Generating submission..."
        )
        inference.generate_predictions(scaler=scaler, load_cached=True)
    else:
        print(
            f"Validation score ({score:.4f}) is NOT better than threshold ({THRESHOLD:.4f}). Skipping submission generation."
        )


if __name__ == "__main__":
    main()
