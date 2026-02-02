import sys
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Ensure the current directory is in the path for library imports
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything, RSNALoss
from library.dataset import RSNADataset
from library.model import FractureModel
from library.engine import fit_model, inference_and_submit


def main():
    # 1. Configuration & Setup
    # Set seed for reproducibility
    seed_everything(Config.SEED)

    # Adjust Config for a fast baseline execution
    # We use 5 epochs which is sufficient for the small dataset (161 samples)
    # to demonstrate convergence without exceeding the time limit.
    Config.EPOCHS = 5
    Config.BATCH_SIZE = 2

    print("=== Starting Fast Baseline Pipeline ===")
    Config.print_config()

    # 2. Training
    print("\n[Step 1/4] Training Model...")
    fit_model()

    # 3. Validation & Metric Calculation
    print("\n[Step 2/4] Validation Assessment...")

    device = Config.DEVICE

    # Load Validation Dataset
    val_dataset = RSNADataset(subset="val")
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Initialize Model and Load Best Weights
    model = FractureModel(
        backbone_name=Config.BACKBONE,
        pretrained=False,  # Weights will be loaded
        num_classes=Config.NUM_CLASSES,
    )

    if os.path.exists(Config.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
        print(f"Loaded best model from {Config.MODEL_SAVE_PATH}")
    else:
        print("Error: Model checkpoint not found. Training may have failed.")
        return

    model.to(device)
    model.eval()

    # Metric Calculation
    # We calculate the accumulated loss over the entire validation set
    # using the custom RSNALoss logic (Weighted Log Loss).
    total_loss = 0.0
    total_samples = 0

    # Storage for Failure Analysis
    all_targets = []
    all_sample_losses = []

    # Criterion for element-wise analysis
    # We replicate the logic of RSNALoss but keep reduction='none' initially
    bce_criterion = torch.nn.BCEWithLogitsLoss(reduction="none")

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device, dtype=torch.float32)
            targets = targets.to(device, dtype=torch.float32)
            batch_size = images.size(0)

            # Inference
            logits = model(images)

            # Calculate Loss per sample for the metric
            # RSNALoss logic: loss = mean(C1..C7) + Patient
            raw_loss = bce_criterion(logits, targets)  # (B, 8)
            vertebrae_loss = raw_loss[:, :7].mean(dim=1)  # (B,)
            patient_loss = raw_loss[:, 7]  # (B,)
            sample_loss = vertebrae_loss + patient_loss  # (B,)

            # Accumulate
            total_loss += sample_loss.sum().item()
            total_samples += batch_size

            # Store for analysis
            all_sample_losses.append(sample_loss.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    # Final Metric: Average over all samples
    final_metric = total_loss / total_samples
    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    print("\n[Step 3/4] Failure Analysis...")
    all_sample_losses = np.concatenate(all_sample_losses)
    all_targets = np.concatenate(all_targets)

    # Construct DataFrame
    target_cols = [f"C{i}" for i in range(1, 8)] + ["patient_overall"]
    df_analysis = pd.DataFrame(all_targets, columns=target_cols)
    df_analysis["error_magnitude"] = all_sample_losses

    # Calculate correlation between Error Magnitude and Target Presence
    # This reveals if positive cases (fractures) are harder to predict.
    correlations = df_analysis.corr()["error_magnitude"].drop("error_magnitude")

    print("Correlation between Error Magnitude and Input Features (Targets):")
    print(correlations)

    # 5. Conditional Submission
    print("\n[Step 4/4] Submission Logic...")
    THRESHOLD = 0.06429807151236185

    if final_metric < THRESHOLD:
        print(
            f"Metric ({final_metric:.6f}) is lower than threshold ({THRESHOLD}). Generating submission..."
        )
        inference_and_submit()
    else:
        print(
            f"Metric ({final_metric:.6f}) is NOT lower than threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
