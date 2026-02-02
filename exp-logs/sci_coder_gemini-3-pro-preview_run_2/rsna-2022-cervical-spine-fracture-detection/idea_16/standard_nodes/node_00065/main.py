import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# Import provided library modules
from library.config import Config, seed_everything
from library.data import get_loaders
from library.engine import RSNAEngine


def main():
    # --- 1. Configuration & Setup ---
    # Set fixed random seeds
    seed_everything(Config.SEED)

    # Limit training epochs for a fast baseline execution
    Config.EPOCHS = 5

    print(f"Starting run with Device: {Config.DEVICE}, Epochs: {Config.EPOCHS}")

    # --- 2. Data Loading ---
    # Load data using cached paths for speed
    print("Loading datasets...")
    train_loader, val_loader, test_loader = get_loaders(load_cached_data=True)

    # --- 3. Model Training ---
    print("Initializing training engine...")
    # Initialize engine (Scheduler uses Config.EPOCHS, so we modified it before init)
    engine = RSNAEngine(device=Config.DEVICE)

    print("Starting training loop...")
    engine.fit(train_loader, val_loader)

    # --- 4. Validation & Metric Calculation ---
    print("Performing final validation assessment...")

    # Load the best model weights saved during training
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    if os.path.exists(best_model_path):
        engine.model.load_state_dict(
            torch.load(best_model_path, map_location=Config.DEVICE)
        )
        print("Best model loaded successfully.")
    else:
        print("Warning: Best model checkpoint not found. Using current model state.")

    engine.model.eval()

    val_logits_list = []
    val_targets_list = []

    # Inference on Validation Set
    # Disable gradients for speed and memory efficiency
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(Config.DEVICE)

            # Use mixed precision for faster inference
            with torch.cuda.amp.autocast():
                logits = engine.model(images)

            # Move results to CPU to avoid GPU OOM during accumulation
            val_logits_list.append(logits.cpu())
            val_targets_list.append(targets.cpu())

    # Concatenate all batches
    val_logits = torch.cat(val_logits_list)
    val_targets = torch.cat(val_targets_list)

    # Calculate Weighted Multi-Label Log Loss
    # Weights: [C1..C7, Overall]
    weights = torch.tensor(Config.CLASS_WEIGHTS, dtype=torch.float32)

    # Compute BCE loss per element (N, C)
    # Target needs to be float for BCE
    element_loss = F.binary_cross_entropy_with_logits(
        val_logits, val_targets.float(), weight=weights, reduction="none"
    )

    # Sum over classes to get per-patient loss (N,)
    patient_loss = element_loss.sum(dim=1)

    # Average over patients to get the final metric
    final_metric = patient_loss.mean().item()

    # Print the required metric
    print(f"Final Validation Metric: {final_metric}")

    # --- 5. Failure Analysis ---
    print("Performing failure analysis...")

    # We correlate error magnitude (patient_loss) with input features (Slice Count)
    # Retrieve metadata to get slice counts
    val_dataset = val_loader.dataset
    val_metadata = val_dataset.metadata
    path_dict = val_dataset.path_dict

    # Extract slice counts in the same order as the validation set
    slice_counts = []
    for uid in val_metadata["StudyInstanceUID"]:
        # Get number of slices from the path dictionary
        num_slices = len(path_dict.get(uid, []))
        slice_counts.append(num_slices)

    slice_counts = np.array(slice_counts)
    errors = patient_loss.numpy()

    # Calculate Pearson Correlation
    if len(errors) > 1 and np.std(slice_counts) > 0 and np.std(errors) > 0:
        corr, _ = pearsonr(errors, slice_counts)
        print(
            f"Correlation between Error Magnitude and Input Features (Slice Count): {corr:.4f}"
        )
    else:
        print(
            "Correlation between Error Magnitude and Input Features (Slice Count): N/A (Insufficient variance)"
        )

    # --- 6. Submission Generation ---
    # Generate submission only if metric is below the specified threshold
    THRESHOLD = 0.1241588886

    if final_metric < THRESHOLD:
        print(
            f"Validation metric ({final_metric}) is lower than threshold ({THRESHOLD}). Generating submission..."
        )
        engine.predict_and_submit(test_loader)
    else:
        print(
            f"Validation metric ({final_metric}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
