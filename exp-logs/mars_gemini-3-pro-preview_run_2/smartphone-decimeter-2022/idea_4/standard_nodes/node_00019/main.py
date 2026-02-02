import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from scipy.stats import spearmanr

# Import library modules
from library import config
from library import utils
from library import dataset
from library import model
from library import engine


def main():
    # -------------------------------------------------------------------------
    # 1. Setup and Configuration
    # -------------------------------------------------------------------------
    # Set random seeds for reproducibility
    torch.manual_seed(config.RANDOM_STATE)
    np.random.seed(config.RANDOM_STATE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Override training parameters for a fast baseline
    # Limit epochs to ensure runtime < 2h and increase patience slightly for stability
    train_params = config.TRAIN_PARAMS.copy()
    train_params["epochs"] = 15
    train_params["patience"] = 3

    # -------------------------------------------------------------------------
    # 2. Data Loading and Preprocessing
    # -------------------------------------------------------------------------
    print("Loading and preprocessing data...")

    # Load Train Data
    # This function loads metadata, loads raw GNSS, aggregates, calculates targets,
    # and saves/loads from cache.
    train_df = dataset.preprocess_data(config.TRAIN_METADATA_PATH, mode="train")

    # Compute Scaler Stats from Train Data
    scaler_stats = dataset.get_scaler_stats(
        train_df, config.INPUT_FEATURES, config.CACHE_FILES["scaler"]
    )

    # Load Validation Data
    val_df = dataset.preprocess_data(config.VAL_METADATA_PATH, mode="val")

    # Initialize Datasets
    # The dataset class handles windowing and padding internally
    train_dataset = dataset.GNSSWindowDataset(
        train_df, config.WINDOW_SIZE, scaler_stats, mode="train"
    )

    val_dataset = dataset.GNSSWindowDataset(
        val_df, config.WINDOW_SIZE, scaler_stats, mode="val"
    )

    # Initialize DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_params["batch_size"],
        shuffle=True,
        num_workers=train_params["num_workers"],
        pin_memory=train_params["pin_memory"],
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_params["batch_size"],
        shuffle=False,
        num_workers=train_params["num_workers"],
        pin_memory=train_params["pin_memory"],
    )

    # -------------------------------------------------------------------------
    # 3. Model Initialization
    # -------------------------------------------------------------------------
    print("Initializing model...")
    net = model.WindowedMLP(
        input_dim=config.MODEL_PARAMS["input_dim"],
        window_size=config.WINDOW_SIZE,
        fc_hidden=config.MODEL_PARAMS["fc_hidden"],
        dropout=config.MODEL_PARAMS["dropout"],
        output_dim=config.MODEL_PARAMS["output_dim"],
    ).to(device)

    # -------------------------------------------------------------------------
    # 4. Training
    # -------------------------------------------------------------------------
    # The engine handles the training loop, validation per epoch, and saving the best model
    net = engine.train_model(net, train_loader, val_loader, device, train_params)

    # -------------------------------------------------------------------------
    # 5. Validation Assessment & Metric Calculation
    # -------------------------------------------------------------------------
    print("Performing final validation assessment...")
    net.eval()

    val_predictions = []

    # Run inference on validation set to get predictions
    with torch.no_grad():
        for window, context, _ in val_loader:
            window = window.to(device)
            context = context.to(device)
            output = net(window, context)
            val_predictions.append(output.cpu().numpy())

    val_predictions = np.concatenate(val_predictions, axis=0)

    # The dataset iterates by tripId then time. We need to align predictions with the dataframe.
    # GNSSWindowDataset iterates groups sorted by tripId (default groupby behavior).
    # Within each trip, we sorted by UnixTimeMillis in the Dataset class.
    # So we must sort our val_df by tripId then UnixTimeMillis to match the predictions order.
    val_df_sorted = val_df.sort_values(by=["tripId", "UnixTimeMillis"]).reset_index(
        drop=True
    )

    # Check lengths
    if len(val_predictions) != len(val_df_sorted):
        print(
            f"Warning: Prediction length {len(val_predictions)} != DataFrame length {len(val_df_sorted)}"
        )

    # Extract Baseline WLS coordinates
    wls_lat = val_df_sorted["WlsLat"].values
    wls_lon = val_df_sorted["WlsLon"].values
    wls_alt = val_df_sorted["WlsAlt"].values

    # Predicted Residuals
    pred_east = val_predictions[:, 0]
    pred_north = val_predictions[:, 1]
    pred_up = np.zeros_like(pred_east)  # We assume 0 vertical correction for 2D task

    # Reconstruct Final Coordinates
    pred_lat, pred_lon, _ = utils.enu_to_lla(
        pred_east, pred_north, pred_up, wls_lat, wls_lon, wls_alt
    )

    # Calculate Distance Error
    gt_lat = val_df_sorted["LatitudeDegrees"].values
    gt_lon = val_df_sorted["LongitudeDegrees"].values

    errors = utils.haversine_distance(pred_lat, pred_lon, gt_lat, gt_lon)
    val_df_sorted["Error"] = errors

    # Calculate Metric: Mean of (50th + 95th) / 2 per phone
    # We group by phone_name to aggregate all trips for a specific device
    phone_scores = []
    for phone, group in val_df_sorted.groupby("phone_name"):
        p50 = np.percentile(group["Error"], 50)
        p95 = np.percentile(group["Error"], 95)
        score = (p50 + p95) / 2
        phone_scores.append(score)

    final_metric = np.mean(phone_scores)

    print(f"Final Validation Metric: {final_metric}")

    # -------------------------------------------------------------------------
    # 6. Failure Analysis
    # -------------------------------------------------------------------------
    print("\nFailure Analysis:")
    # Correlate Error with Input Features
    # We use the raw feature values from the dataframe for correlation
    analysis_features = config.INPUT_FEATURES
    correlations = {}

    for feat in analysis_features:
        if feat in val_df_sorted.columns:
            # Drop NaNs for correlation calculation just in case
            valid_mask = ~np.isnan(val_df_sorted[feat]) & ~np.isnan(
                val_df_sorted["Error"]
            )
            if np.sum(valid_mask) > 0:
                corr, _ = spearmanr(
                    val_df_sorted.loc[valid_mask, feat],
                    val_df_sorted.loc[valid_mask, "Error"],
                )
                correlations[feat] = corr

    # Sort by absolute correlation
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

    print("Spearman Correlation with Error Magnitude:")
    for feat, corr in sorted_corr:
        print(f"  {feat}: {corr:.4f}")

    # -------------------------------------------------------------------------
    # 7. Submission Generation
    # -------------------------------------------------------------------------
    threshold = 4.256982128481356

    if final_metric < threshold:
        print(
            f"\nValidation metric ({final_metric}) is better than threshold ({threshold}). Generating submission..."
        )

        # Load Test Data
        test_df = dataset.preprocess_data(config.TEST_METADATA_PATH, mode="test")

        # Initialize Test Dataset
        test_dataset = dataset.GNSSWindowDataset(
            test_df, config.WINDOW_SIZE, scaler_stats, mode="test"
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=train_params["batch_size"],
            shuffle=False,
            num_workers=train_params["num_workers"],
            pin_memory=train_params["pin_memory"],
        )

        # Generate Submission
        engine.generate_submission(net, test_loader, test_df, device)

    else:
        print(
            f"\nValidation metric ({final_metric}) is NOT better than threshold ({threshold}). Skipping submission."
        )


if __name__ == "__main__":
    main()
