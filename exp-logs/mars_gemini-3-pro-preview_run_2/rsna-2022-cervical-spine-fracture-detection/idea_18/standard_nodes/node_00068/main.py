import sys
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, load_checkpoint
from library.train import run_training
from library.inference import predict
from library.dataset import RSNADataset, get_transforms, cache_image_paths
from library.model import CervicalSpineModel
from library.loss import WeightedMultiLabelLoss


def main():
    # --- 1. Configuration for Fast Baseline ---
    # Set epochs to 2 to ensure execution finishes within the 37-minute limit.
    # The dataset is small (161 training samples), so 2 epochs is very fast but sufficient for a baseline check.
    Config.epochs = 2
    Config.debug = False  # Use the full provided training set

    print(f"Starting pipeline with {Config.epochs} epochs...")

    # --- 2. Training ---
    try:
        # Execute training using the provided library function
        run_training(epochs=Config.epochs, load_cached_data=True)
    except Exception as e:
        print(f"Critical Error during training: {e}")
        return

    # --- 3. Validation & Failure Analysis ---
    print("Starting Validation & Failure Analysis...")

    # Load validation metadata
    val_df = pd.read_csv(Config.val_metadata_path)

    # Cache image paths (likely already cached during training)
    val_paths_map = cache_image_paths(val_df, "val", load_cached_data=True)

    # Initialize Validation Dataset & Loader
    val_dataset = RSNADataset(
        val_df, val_paths_map, phase="val", transform=get_transforms("val")
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    # Initialize Model
    model = CervicalSpineModel()
    model.to(Config.device)

    # Load the best checkpoint saved during training
    checkpoint_path = os.path.join(Config.working_dir, "best_model.pth")
    if os.path.exists(checkpoint_path):
        load_checkpoint(model, checkpoint_path)
    else:
        print("Warning: best_model.pth not found. Using random weights for validation.")

    model.eval()

    # Initialize Loss Function for Metric Calculation
    criterion = WeightedMultiLabelLoss()

    total_loss_sum = 0.0
    total_samples = 0
    sample_errors = []
    slice_counts = []

    # Pre-extract slice counts for failure analysis (order matches val_loader)
    for uid in val_df["StudyInstanceUID"]:
        paths = val_paths_map.get(uid, [])
        slice_counts.append(len(paths))

    # Validation Inference Loop
    with torch.no_grad():
        for i, (images, targets) in enumerate(val_loader):
            images = images.to(Config.device)
            targets = targets.to(Config.device)

            # Forward Pass
            logits = model(images)

            # --- Metric Calculation ---
            # We calculate the weighted log loss per sample manually to allow for failure analysis
            # and to ensure we compute the exact metric required.

            # 1. Compute raw BCE (Batch, 8)
            bce_loss = torch.nn.BCEWithLogitsLoss(reduction="none")(
                logits, targets.float()
            )

            # 2. Apply Class Weights (Batch, 8)
            # criterion.weights is already on the correct device
            weighted_loss = bce_loss * criterion.weights

            # 3. Average across classes to get per-sample loss (Batch, )
            # "Finally, loss is averaged across all rows." -> Mean over classes
            per_sample_loss = weighted_loss.mean(dim=1)

            # Accumulate totals
            total_loss_sum += per_sample_loss.sum().item()
            total_samples += images.size(0)

            # Store per-sample error for analysis
            sample_errors.extend(per_sample_loss.cpu().numpy())

    # Compute Final Metric
    final_metric = total_loss_sum / total_samples if total_samples > 0 else 0.0
    print(f"Final Validation Metric: {final_metric}")

    # --- Failure Analysis ---
    # Correlation between Error Magnitude and Slice Count (Z-Depth)
    if len(sample_errors) == len(slice_counts) and len(sample_errors) > 1:
        correlation = np.corrcoef(sample_errors, slice_counts)[0, 1]
        print(f"Correlation between Error and Slice Count: {correlation:.4f}")
    else:
        print("Could not calculate correlation (insufficient data or mismatch).")

    # --- 4. Conditional Submission ---
    threshold = 0.1241588886

    if final_metric < threshold:
        print(
            f"Metric {final_metric:.6f} is lower than threshold {threshold}. Generating submission..."
        )

        # Optimize batch size for inference
        # Since we don't need gradients, we can increase batch size to speed up the large test set inference
        Config.batch_size = 16

        predict(load_cached_data=True)
    else:
        print(
            f"Metric {final_metric:.6f} >= {threshold}. Submission generation skipped."
        )


if __name__ == "__main__":
    main()
