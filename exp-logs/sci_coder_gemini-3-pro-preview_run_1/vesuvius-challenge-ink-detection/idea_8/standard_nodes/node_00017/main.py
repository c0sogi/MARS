import os
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.dataset import InkDataset
from library.model import FRUNet
from library.train import train_one_epoch, validate, set_seed
from library.inference import predict_and_encode


def main():
    # --- 1. Setup ---
    # Ensure reproducibility and setup directories
    set_seed(Config.SEED)
    Config.setup_directories()
    device = torch.device(Config.DEVICE)

    print(f"Running FR-UNet Pipeline on {device}")

    # --- 2. Data Loading ---
    # Load full datasets (size is small enough for rapid training)
    # Using cached data if available for speed
    train_dataset = InkDataset(mode="train", load_cached_data=True)
    val_dataset = InkDataset(mode="val", load_cached_data=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    # --- 3. Model Initialization ---
    model = FRUNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    # --- 4. Training Loop ---
    best_f05 = 0.0
    best_threshold = 0.5

    # Paths for saving artifacts
    model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    thresh_path = os.path.join(Config.CHECKPOINT_DIR, "best_threshold.txt")

    print(f"Starting training for {Config.NUM_EPOCHS} epochs...")

    for epoch in range(Config.NUM_EPOCHS):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, Config.POS_WEIGHT
        )

        # Validate
        val_loss, val_f05, val_thresh = validate(
            model, val_loader, device, Config.POS_WEIGHT
        )

        # Save Best Model
        if val_f05 > best_f05:
            best_f05 = val_f05
            best_threshold = val_thresh

            torch.save(model.state_dict(), model_path)
            with open(thresh_path, "w") as f:
                f.write(str(best_threshold))

    # Required Output: Final Validation Metric
    print(f"Final Validation Metric: {best_f05}")

    # --- 5. Failure Analysis ---
    print("Performing failure analysis on validation set...")

    # Load best model for analysis
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    analysis_data = []

    with torch.no_grad():
        for volumes, labels, masks, sample_ids in val_loader:
            volumes = volumes.to(device)
            labels = labels.to(device)
            masks = masks.to(device)

            # Inference
            outputs = model(volumes)
            preds = (outputs >= best_threshold).float()

            # Calculate Error: Mean Absolute Error on valid pixels per sample
            # |Prediction - Truth| * Mask
            errors = torch.abs(preds - labels) * masks

            # Aggregate per sample
            # Sum error over (C, H, W)
            sum_errors = errors.sum(dim=(1, 2, 3))
            # Count valid pixels
            valid_pixels = masks.sum(dim=(1, 2, 3)) + 1e-7
            # Mean error
            mean_errors = (sum_errors / valid_pixels).cpu().numpy()

            # Calculate Mean Intensity of input volume per sample for correlation
            vol_means = volumes.mean(dim=(1, 2, 3)).cpu().numpy()

            # Collect metadata
            for i, sid in enumerate(sample_ids):
                # Retrieve metadata from dataframe
                meta_row = val_dataset.df[val_dataset.df["sample_id"] == sid].iloc[0]

                analysis_data.append(
                    {
                        "sample_id": sid,
                        "error": mean_errors[i],
                        "x": meta_row["x"],
                        "y": meta_row["y"],
                        "mean_intensity": vol_means[i],
                    }
                )

    # Compute Correlations
    if analysis_data:
        df_analysis = pd.DataFrame(analysis_data)
        print("Correlation between Error Magnitude and Input Features:")

        # Calculate correlation matrix
        correlations = df_analysis[["error", "x", "y", "mean_intensity"]].corr()[
            "error"
        ]

        for feature in ["x", "y", "mean_intensity"]:
            print(f"{feature}: {correlations.get(feature, 0.0)}")
    else:
        print("No validation data available for failure analysis.")

    # --- 6. Submission Generation ---
    # Benchmark Threshold
    BENCHMARK_SCORE = 0.41758

    if best_f05 > BENCHMARK_SCORE:
        print(
            f"Validation score {best_f05} exceeds benchmark {BENCHMARK_SCORE}. Generating submission..."
        )
        predict_and_encode(
            checkpoint_path=model_path,
            threshold_path=thresh_path,
            output_file=Config.SUBMISSION_FILE,
            load_cached_data=True,
        )
    else:
        print(
            f"Validation score {best_f05} does not exceed benchmark {BENCHMARK_SCORE}. Skipping submission."
        )


if __name__ == "__main__":
    main()
