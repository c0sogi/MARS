import torch
import numpy as np
import pandas as pd
import os
from torch.utils.data import DataLoader

# Import library modules
from library.config import Config
from library.train import set_seed
from library.dataset import GNSSWindowDataset
from library.model import BiLSTMRegressor, train_model, generate_submission
from library.utils import local_meters_to_wgs84, haversine_distance


def main():
    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    # Override Config for fast baseline execution
    Config.EPOCHS = 15
    Config.BATCH_SIZE = 512
    Config.LEARNING_RATE = 0.002

    # Set random seed for reproducibility
    set_seed(Config.RANDOM_STATE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # -------------------------------------------------------------------------
    # 2. Data Loading
    # -------------------------------------------------------------------------
    print("\nLoading Datasets...")
    # Load Training Data
    train_dataset = GNSSWindowDataset(mode="train", load_cached_data=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # Load Validation Data
    val_dataset = GNSSWindowDataset(mode="val", load_cached_data=True)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,  # Must be False to align with dataframe indices
        num_workers=4,
        pin_memory=True,
    )

    # -------------------------------------------------------------------------
    # 3. Model Initialization & Training
    # -------------------------------------------------------------------------
    print("\nInitializing Model...")
    input_dim = train_dataset.features.shape[1]
    output_dim = len(Config.TARGET_COLUMNS)

    model = BiLSTMRegressor(
        input_dim=input_dim,
        hidden_dim=Config.HIDDEN_DIM,
        num_layers=Config.NUM_LAYERS,
        dropout=Config.DROPOUT,
        output_dim=output_dim,
    )

    print("Starting Training...")
    # train_model handles the training loop, validation monitoring, and saving best weights
    trained_model = train_model(model, train_loader, val_loader)

    # -------------------------------------------------------------------------
    # 4. Validation Assessment
    # -------------------------------------------------------------------------
    print("\nPerforming Validation Assessment...")
    trained_model.eval()

    val_preds = []
    with torch.no_grad():
        for x, _ in val_loader:
            x = x.to(device)
            out = trained_model(x)
            val_preds.append(out.cpu().numpy())

    # Concatenate predictions: Shape (N_val, 2) -> [DeltaEast, DeltaNorth]
    pred_deltas = np.concatenate(val_preds, axis=0)

    # Retrieve validation dataframe for reconstruction
    df_val = val_dataset.df

    # Reconstruct WGS84 coordinates
    # pred_lat, pred_lon = Baseline + Predicted_Delta_Meters converted to Degrees
    pred_lat, pred_lon = local_meters_to_wgs84(
        df_val["WlsLat"].values,
        df_val["WlsLon"].values,
        pred_deltas[:, 0],  # DeltaEast
        pred_deltas[:, 1],  # DeltaNorth
    )

    # Calculate Haversine Distance Error
    gt_lat = df_val["LatitudeDegrees"].values
    gt_lon = df_val["LongitudeDegrees"].values

    errors = haversine_distance(gt_lat, gt_lon, pred_lat, pred_lon)

    # Add error to dataframe for grouping
    df_val["Error"] = errors

    # Calculate Metric: Mean of (50th + 95th percentile) averaged across phones
    trip_scores = []
    for trip_id, group in df_val.groupby("tripId"):
        trip_errors = group["Error"].values
        p50 = np.percentile(trip_errors, 50)
        p95 = np.percentile(trip_errors, 95)
        score = (p50 + p95) / 2.0
        trip_scores.append(score)

    final_metric = np.mean(trip_scores)
    print(f"Final Validation Metric: {final_metric}")

    # -------------------------------------------------------------------------
    # 5. Failure Analysis
    # -------------------------------------------------------------------------
    print("\nFailure Analysis (Spearman Correlation with Error):")
    # Calculate correlation between input features and the error magnitude
    # Note: df_val features are already scaled, but correlation is invariant to scaling
    feature_cols = Config.INPUT_FEATURES
    correlations = df_val[feature_cols].corrwith(df_val["Error"], method="spearman")
    print(correlations.sort_values(ascending=False))

    # -------------------------------------------------------------------------
    # 6. Submission Generation
    # -------------------------------------------------------------------------
    THRESHOLD = 4.256982128481356

    if final_metric < THRESHOLD:
        print(
            f"\nMetric {final_metric} meets threshold ({THRESHOLD}). Generating submission..."
        )

        # Load Test Data
        test_dataset = GNSSWindowDataset(mode="test", load_cached_data=True)
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        # Generate and Save Submission
        generate_submission(trained_model, test_loader)

    else:
        print(
            f"\nMetric {final_metric} does NOT meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
