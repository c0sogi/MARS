import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import provided library functions
from library.utils import haversine_distance
from library.data_loader import GnssWindowedDataset
from library.model import TemporalConvNet, generate_submission
from library.train import run_training

# Configuration for Fast Baseline
CONFIG = {
    "window_size": 64,
    "batch_size": 256,
    "lr": 0.0005,
    "epochs": 20,
    "patience": 5,
    "hidden_dim": 64,  # Smaller model
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "seed": 42,
}


def main():
    print("=== Starting End-to-End Pipeline ===")

    # 1. Run Training
    # This handles:
    # - Loading train/val metadata
    # - Training the model
    # - Saving model weights to ./working/model_weights.pth
    # - Generating submission file at ./submission/submission.csv
    print("\n[Step 1] Running Training and Submission Generation...")
    history = run_training(
        input_dir="./input",
        metadata_dir="./metadata",
        working_dir="./working",
        submission_dir="./submission",
        config=CONFIG,
    )

    # 2. Validation Assessment & Failure Analysis
    print("\n[Step 2] Performing Validation Assessment...")

    # We need to reconstruct the environment to run inference on validation set manually
    # to compute the specific competition metric and correlations.

    # A. Re-fit Scaler on Training Data
    # (Required because run_training doesn't return the scaler object)
    print("Re-initializing datasets to restore scaler...")
    train_meta_path = "./metadata/train_metadata.csv"
    if not os.path.exists(train_meta_path):
        raise FileNotFoundError("Train metadata not found.")

    df_train_meta = pd.read_csv(train_meta_path)

    # Initialize train dataset solely to fit the scaler
    train_dataset = GnssWindowedDataset(
        metadata_df=df_train_meta,
        input_dir="./input",
        window_size=CONFIG["window_size"],
        mode="train",
        scaler=None,
    )
    scaler = train_dataset.scaler

    # B. Load Validation Data
    val_meta_path = "./metadata/val_metadata.csv"
    df_val_meta = pd.read_csv(val_meta_path)

    val_dataset = GnssWindowedDataset(
        metadata_df=df_val_meta,
        input_dir="./input",
        window_size=CONFIG["window_size"],
        mode="train",  # train mode ensures we load ground truth for comparison
        scaler=scaler,
    )

    val_loader = DataLoader(
        val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=2
    )

    # C. Load Model
    print("Loading trained model...")
    model = TemporalConvNet(
        input_channels=3,  # Fixed based on data_loader features
        window_size=CONFIG["window_size"],
        hidden_dim=CONFIG["hidden_dim"],
        output_dim=2,
    )

    weights_path = "./working/model_weights.pth"
    model.load_state_dict(torch.load(weights_path, map_location=CONFIG["device"]))
    model.to(CONFIG["device"])
    model.eval()

    # D. Run Inference
    print("Running inference on validation set...")
    predictions_list = []
    ground_truths_list = []
    features_list = []

    with torch.no_grad():
        for batch in val_loader:
            # Move data to device
            features = batch["features"].to(CONFIG["device"])
            baseline = batch["baseline"].numpy()  # CPU numpy
            target_residuals = batch["target"].numpy()  # CPU numpy

            # Forward pass
            predicted_residuals = model(features).cpu().numpy()

            # Reconstruct coordinates: Prediction = Baseline + Residual
            preds = baseline + predicted_residuals
            gts = baseline + target_residuals

            predictions_list.append(preds)
            ground_truths_list.append(gts)

            # Extract center features for analysis (Batch, Window, Channels) -> (Batch, Channels)
            # Center index of window
            center_idx = CONFIG["window_size"] // 2
            # features is (Batch, Window, Channels).
            # We take the center time step features.
            # Move back to cpu
            batch_feats = features[:, center_idx, :].cpu().numpy()
            features_list.append(batch_feats)

    # Concatenate
    all_preds = np.concatenate(predictions_list, axis=0)
    all_gts = np.concatenate(ground_truths_list, axis=0)
    all_feats = np.concatenate(features_list, axis=0)

    # E. Compute Metric
    # Assign back to dataframe to group by trace
    df_val_meta["pred_lat"] = all_preds[:, 0]
    df_val_meta["pred_lon"] = all_preds[:, 1]
    df_val_meta["gt_lat"] = all_gts[:, 0]
    df_val_meta["gt_lon"] = all_gts[:, 1]

    # Calculate Haversine Distance
    df_val_meta["error_meters"] = haversine_distance(
        df_val_meta["pred_lat"],
        df_val_meta["pred_lon"],
        df_val_meta["gt_lat"],
        df_val_meta["gt_lon"],
    )

    # Metric: Mean of (50th + 95th percentile) per phone trace
    trace_scores = []
    # Group by drive_id and phone_name to identify unique traces
    for _, group in df_val_meta.groupby(["drive_id", "phone_name"]):
        p50 = np.percentile(group["error_meters"], 50)
        p95 = np.percentile(group["error_meters"], 95)
        score = (p50 + p95) / 2
        trace_scores.append(score)

    final_metric = np.mean(trace_scores)
    print(f"Final Validation Metric: {final_metric}")

    # F. Failure Analysis
    print("\n[Step 3] Failure Analysis")
    print("Correlation between Error Magnitude and Input Features:")

    # Feature names from data_loader
    feature_names = ["WlsAlt", "Cn0DbHz", "SvElevationDegrees"]

    # Calculate Pearson correlation
    errors = df_val_meta["error_meters"].values

    for i, fname in enumerate(feature_names):
        # all_feats is normalized, but correlation is invariant to linear scaling
        feat_values = all_feats[:, i]
        corr = np.corrcoef(feat_values, errors)[0, 1]
        print(f"  {fname}: {corr:.4f}")

    # G. Conditional Submission Generation
    if final_metric < 17.661822205737813:
        print(
            f"\n[Step 4] Validation metric {final_metric:.4f} < 17.66. Generating submission..."
        )
        test_meta_path = "./metadata/test_metadata.csv"
        output_file = "./submission/submission.csv"

        generate_submission(
            model=model,
            test_metadata_path=test_meta_path,
            input_dir="./input",
            output_file=output_file,
            config=CONFIG,
            scaler=scaler,
        )
    else:
        print(
            f"\n[Step 4] Validation metric {final_metric:.4f} >= 17.66. Skipping submission."
        )

    print("\nPipeline Complete.")


if __name__ == "__main__":
    main()
