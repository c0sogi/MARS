import os
import sys
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Ensure library modules can be imported
sys.path.append(".")

from library.config import Config
from library.utils import seed_everything, TargetScaler
from library.dataset import VolcanoDataset
from library.model import AttentionPooledHybridEfficientNet
from library.train import run_training
from library.predict import generate_predictions


def main():
    # ---------------------------------------------------------
    # 1. Setup
    # ---------------------------------------------------------
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Orchestration started. Device: {device}")

    # ---------------------------------------------------------
    # 2. Training
    # ---------------------------------------------------------
    # We use 6 epochs to ensure the process completes comfortably within the 2-hour limit
    # while providing enough steps for the pre-trained backbone to converge.
    print("\n=== Starting Training Phase ===")
    run_training(num_epochs=6, batch_size=Config.BATCH_SIZE, debug=False)

    # ---------------------------------------------------------
    # 3. Validation & Evaluation
    # ---------------------------------------------------------
    print("\n=== Starting Validation Phase ===")

    # Initialize Scaler (will load parameters saved during training)
    target_scaler = TargetScaler()

    # Load Validation Dataset
    val_dataset = VolcanoDataset(
        metadata_path=Config.VAL_METADATA, mode="val", target_scaler=target_scaler
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Load Best Model
    model = AttentionPooledHybridEfficientNet()
    if not os.path.exists(Config.MODEL_SAVE_PATH):
        raise FileNotFoundError(
            f"Model checkpoint not found at {Config.MODEL_SAVE_PATH}"
        )

    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    model.to(device)
    model.eval()

    # Inference Loop
    all_preds = []
    all_targets = []
    all_features = []

    print(f"Evaluating on {len(val_dataset)} validation samples...")

    with torch.no_grad():
        for batch in val_loader:
            spectrogram = batch["spectrogram"].to(device)
            features = batch["features"].to(device)
            target = batch["target"].to(device).unsqueeze(1)

            # Forward pass
            outputs = model(spectrogram, features)

            # Inverse Transform to original scale
            preds_original = target_scaler.inverse_transform(outputs)
            targets_original = target_scaler.inverse_transform(target)

            # Store results
            all_preds.append(preds_original.cpu().numpy().flatten())
            all_targets.append(targets_original.cpu().numpy().flatten())

            # Store features for failure analysis (keep on CPU)
            all_features.append(features.cpu().numpy())

    # Concatenate results
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)
    X_val = np.vstack(all_features)

    # Compute Final Metric
    mae = np.mean(np.abs(y_pred - y_true))
    print(f"Final Validation Metric: {mae}")

    # ---------------------------------------------------------
    # 4. Failure Analysis
    # ---------------------------------------------------------
    print("\n=== Failure Analysis ===")
    errors = np.abs(y_pred - y_true)

    # Create DataFrame for correlation analysis
    # val_dataset.feature_cols contains the names of the statistical features
    df_analysis = pd.DataFrame(X_val, columns=val_dataset.feature_cols)
    df_analysis["error_magnitude"] = errors

    # Compute correlation between features and error magnitude
    correlations = df_analysis.corr()["error_magnitude"].drop("error_magnitude")

    # Display top correlations
    print("Top 10 Features Correlated with Error Magnitude:")
    print(correlations.abs().sort_values(ascending=False).head(10))

    # ---------------------------------------------------------
    # 5. Submission
    # ---------------------------------------------------------
    print("\n=== Submission Check ===")
    threshold = 1492505.6322055138

    if mae < threshold:
        print(f"Validation MAE ({mae:.4f}) meets the threshold ({threshold:.4f}).")
        print("Generating submission file...")
        generate_predictions(batch_size=Config.BATCH_SIZE, device=Config.DEVICE)
    else:
        print(
            f"Validation MAE ({mae:.4f}) does NOT meet the threshold ({threshold:.4f})."
        )
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
