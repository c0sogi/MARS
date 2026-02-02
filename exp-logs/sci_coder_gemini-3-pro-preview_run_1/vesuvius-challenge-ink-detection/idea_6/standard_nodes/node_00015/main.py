import os
import sys
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Ensure library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything
from library.dataset import InkDataset
from library.model import FFDCNet
from library.train import train_model, validate
from library.inference import generate_submission


def main():
    # 1. Setup
    Config.setup()
    seed_everything(Config.SEED)

    # Optimize configuration for a fast baseline execution
    # The dataset is small (455 samples), so 15 epochs is sufficient and very fast.
    Config.EPOCHS = 15

    # 2. Train Model
    # This will train, validate per epoch, and save the best model/threshold.
    print("Starting training pipeline...")
    _, _ = train_model(load_cached_data=True)

    # 3. Load Best Model for Final Evaluation
    device = Config.DEVICE
    model = FFDCNet().to(device)
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        # Fallback if training failed to produce a checkpoint (unlikely)
        print("Warning: Best model checkpoint not found. Using current model state.")

    # 4. Final Validation & Metric Reporting
    # We re-run validation on the best loaded model to get the exact final metric.
    val_dataset = InkDataset(Config.VAL_METADATA, load_cached_data=True)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Define loss for validation (same as training)
    pos_weight = torch.tensor([Config.POS_WEIGHT]).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Run validation
    val_loss, val_f05, val_thresh = validate(model, val_loader, criterion, device)

    # Required output format
    print(f"Final Validation Metric: {val_f05}")

    # 5. Failure Analysis
    print("Performing failure analysis...")
    model.eval()

    errors = []
    meta_features = []
    val_df = val_dataset.df

    with torch.no_grad():
        for i, (volumes, labels) in enumerate(val_loader):
            volumes = volumes.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(volumes)
            probs = torch.sigmoid(outputs)

            # Calculate Mean Absolute Error (MAE) per patch
            # Shape: (Batch, 1, H, W) -> (Batch,)
            abs_diff = torch.abs(probs - labels)
            batch_errors = abs_diff.view(volumes.size(0), -1).mean(dim=1).cpu().numpy()

            # Calculate mean intensity of input volume for correlation analysis
            batch_intensities = (
                volumes.view(volumes.size(0), -1).mean(dim=1).cpu().numpy()
            )

            # Map back to metadata
            start_idx = i * Config.BATCH_SIZE
            end_idx = start_idx + volumes.size(0)
            batch_meta = val_df.iloc[start_idx:end_idx]

            for j, error in enumerate(batch_errors):
                row = batch_meta.iloc[j]
                errors.append(error)
                meta_features.append(
                    {
                        "x": row["x"],
                        "y": row["y"],
                        "mean_intensity": batch_intensities[j],
                    }
                )

    # Calculate and print correlations
    if len(errors) > 0:
        analysis_df = pd.DataFrame(meta_features)
        analysis_df["error"] = errors

        print("Correlation between Error Magnitude and Features:")
        # Compute correlation with error for each feature
        correlations = analysis_df.corr()["error"].drop("error")
        for feat, corr in correlations.items():
            print(f"{feat}: {corr}")

    # 6. Submission Generation
    # Condition: Metric must be > 0.41758
    submission_threshold = 0.41758

    if val_f05 > submission_threshold:
        print(
            f"Validation metric {val_f05} exceeds threshold {submission_threshold}. Generating submission..."
        )
        generate_submission(load_cached_data=True)
    else:
        print(
            f"Validation metric {val_f05} does not exceed threshold {submission_threshold}. Skipping submission."
        )


if __name__ == "__main__":
    main()
