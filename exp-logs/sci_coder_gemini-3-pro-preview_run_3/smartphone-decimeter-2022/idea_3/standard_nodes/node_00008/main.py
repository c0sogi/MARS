import os
import torch
import numpy as np
import pandas as pd
from library import config, dataset, model, trainer, utils


def set_seed(seed=42):
    """Sets the seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def run_failure_analysis(model, dataloader, device):
    """
    Runs inference on validation set to compute metric and perform failure analysis.
    """
    model.eval()

    all_preds = []
    all_gts = []

    # Containers for failure analysis
    feature_data = []
    error_data = []

    print("Running Validation and Failure Analysis...")

    with torch.no_grad():
        for batch in dataloader:
            features = batch["features"].to(device)
            targets = batch["targets"].to(device)
            mask = batch["mask"].to(device)
            lengths = batch["lengths"]

            trip_ids = batch["tripIds"]
            timestamps_list = batch["timestamps"]
            wls_pos_list = batch["wls_pos"]

            # Forward pass
            outputs = model(features, lengths)

            # Move to CPU
            outputs_np = outputs.cpu().numpy()
            targets_np = targets.cpu().numpy()
            features_np = features.cpu().numpy()
            mask_np = mask.cpu().numpy()

            # Process batch
            for i in range(len(trip_ids)):
                length = lengths[i]
                trip_id = trip_ids[i]

                # Extract valid sequence data
                valid_mask = mask_np[i, :length]

                # Residuals (dLat, dLon)
                pred_res = outputs_np[i, :length, :]
                target_res = targets_np[i, :length, :]

                # Features for analysis
                feats = features_np[i, :length, :]

                # Metadata
                ts = timestamps_list[i]
                wls = wls_pos_list[i]

                # Reconstruct Absolute Coordinates
                # Pred = WLS + Predicted Residual
                lat_pred = wls[:, 0] + pred_res[:, 0]
                lon_pred = wls[:, 1] + pred_res[:, 1]

                # GT = WLS + Target Residual
                lat_gt = wls[:, 0] + target_res[:, 0]
                lon_gt = wls[:, 1] + target_res[:, 1]

                # Calculate Distance Error for this sequence
                dists = utils.haversine_loss(lat_pred, lon_pred, lat_gt, lon_gt)

                # Store for Metric Calculation
                df_p = pd.DataFrame(
                    {
                        "tripId": trip_id,
                        "UnixTimeMillis": ts,
                        "LatitudeDegrees": lat_pred,
                        "LongitudeDegrees": lon_pred,
                    }
                )
                all_preds.append(df_p)

                df_g = pd.DataFrame(
                    {
                        "tripId": trip_id,
                        "UnixTimeMillis": ts,
                        "LatitudeDegrees": lat_gt,
                        "LongitudeDegrees": lon_gt,
                    }
                )
                all_gts.append(df_g)

                # Store for Failure Analysis
                feature_data.append(feats)
                error_data.append(dists)

    # 1. Calculate Metric
    if all_preds:
        df_pred_all = pd.concat(all_preds, ignore_index=True)
        df_gt_all = pd.concat(all_gts, ignore_index=True)
        val_score = utils.calc_score(df_pred_all, df_gt_all)
        print(f"Final Validation Metric: {val_score}")
    else:
        val_score = float("inf")
        print("Final Validation Metric: inf")

    # 2. Failure Analysis
    if feature_data and error_data:
        # Concatenate all sequences
        X_all = np.concatenate(feature_data, axis=0)
        y_err = np.concatenate(error_data, axis=0)

        # Create DataFrame
        analysis_df = pd.DataFrame(X_all, columns=config.FEATURE_NAMES)
        analysis_df["Error_Magnitude"] = y_err

        # Compute Correlation
        correlations = analysis_df.corr()["Error_Magnitude"].sort_values(
            ascending=False
        )

        print("\n--- Failure Analysis: Feature Correlations with Error Magnitude ---")
        print(correlations.drop("Error_Magnitude"))
        print("-------------------------------------------------------------------")

    return val_score


def main():
    # 1. Setup
    set_seed(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # 2. Data Loading
    # load_cached_data=True allows using pre-processed parquet files if they exist
    print("Initializing DataLoaders...")
    train_loader, val_loader, test_loader, scaler = dataset.get_dataloaders(
        batch_size=config.BATCH_SIZE, num_workers=2  # Reduced workers for safety
    )

    # 3. Training
    # We use a limited number of epochs for a fast baseline
    print("Starting Training...")
    trainer.run_training(
        train_loader,
        val_loader,
        epochs=15,
        learning_rate=config.LEARNING_RATE,
        device_name=str(device),
    )

    # 4. Load Best Model for Analysis
    best_model_path = os.path.join(config.WORKING_DIR, "model_best.pth")
    if not os.path.exists(best_model_path):
        print("Error: Best model not found. Training might have failed.")
        return

    best_model = model.ResidualBiLSTM(
        input_size=config.INPUT_SIZE,
        hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LAYERS,
        output_size=config.OUTPUT_SIZE,
        dropout=config.DROPOUT,
        bidirectional=config.BIDIRECTIONAL,
    ).to(device)

    best_model.load_state_dict(torch.load(best_model_path, map_location=device))
    print(f"Loaded best model from {best_model_path}")

    # 5. Validation & Failure Analysis
    val_score = run_failure_analysis(best_model, val_loader, device)

    # 6. Submission
    threshold = 4.32379283550646
    if val_score < threshold:
        print(
            f"Validation score {val_score} meets threshold ({threshold}). Generating submission..."
        )
        trainer.generate_submission(
            test_loader,
            best_model_path,
            config.SUBMISSION_PATH,
            device_name=str(device),
        )
    else:
        print(
            f"Validation score {val_score} does not meet threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    main()
