import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

# Import provided library modules
from library.config import Config
from library.trainer import Trainer
from library.utils import seed_everything, load_checkpoint
from library.dataset import load_and_process_df


def run():
    # 1. Configuration & Setup
    # ---------------------------------------------------------
    # Set random seed for reproducibility
    seed_everything(Config.seed)

    # Use the full training duration defined in Config (10 epochs) to ensure convergence.
    # Cite solution_lesson_node_00008: Prioritize convergence over complexity.

    print("Configuration:")
    print(f"  Epochs: {Config.epochs}")
    print(f"  Debug Mode: {Config.debug}")
    print(f"  Device: {Config.device}")

    # 2. Training
    # ---------------------------------------------------------
    trainer = Trainer()
    trainer.fit()

    # 3. Validation & Failure Analysis
    # ---------------------------------------------------------
    print("\nStarting Validation and Failure Analysis...")

    # Load the best model saved during training to ensure we evaluate the peak performance
    if os.path.exists(Config.model_save_path):
        print(f"Loading best model from {Config.model_save_path}")
        load_checkpoint(trainer.model, Config.model_save_path, device=Config.device)
    else:
        print("Warning: Best model checkpoint not found. Using current model state.")

    trainer.model.eval()
    val_loader = trainer.get_dataloader("val")

    all_probs = []
    all_targets = []

    # Efficient Inference Loop
    # We disable gradients to save memory and computation speed
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(Config.device, non_blocking=True)

            # Forward pass
            logits = trainer.model(images)
            probs = torch.sigmoid(logits)

            # Store results on CPU to avoid OOM
            all_probs.append(probs.cpu().numpy())
            all_targets.append(targets.numpy())

    # Concatenate all batches
    all_probs = np.concatenate(all_probs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Determine Best Threshold on Validation Set
    # We scan the range [0.01, 0.99] to find the global optimum
    best_val_f1 = 0.0
    best_thresh = 0.5
    thresholds = np.arange(0.01, 1.00, 0.01)

    # Loop through thresholds to find the best F1
    for t in thresholds:
        # Binarize
        preds_bin = (all_probs > t).astype(np.int8)
        # Calculate Micro F1
        score = f1_score(all_targets, preds_bin, average="micro", zero_division=0)

        if score > best_val_f1:
            best_val_f1 = score
            best_thresh = t

    # REQUIRED OUTPUT: Print Final Validation Metric
    print(f"Final Validation Metric: {best_val_f1}")
    print(f"Optimal Threshold: {best_thresh:.2f}")

    # Failure Analysis
    # ---------------------------------------------------------
    print("\nPerforming Failure Analysis...")

    # Calculate Sample-wise Error Magnitude
    # We define error as (1.0 - Sample-wise F1 Score) using the optimal threshold
    final_preds = (all_probs > best_thresh).astype(np.float32)

    # Calculate Intersection (True Positives) per sample
    intersection = np.sum(final_preds * all_targets, axis=1)

    # Calculate Sum of Preds and Targets per sample
    sum_preds = np.sum(final_preds, axis=1)
    sum_targets = np.sum(all_targets, axis=1)

    denominator = sum_preds + sum_targets

    # Calculate F1 per sample.
    # Handle division by zero: if denom is 0, it means target=0 and pred=0, which is a perfect match (F1=1)
    sample_f1 = np.divide(
        2 * intersection,
        denominator,
        out=np.ones_like(intersection),
        where=denominator != 0,
    )

    error_magnitude = 1.0 - sample_f1

    # Feature 1: Number of Labels (Ground Truth Complexity)
    num_labels = sum_targets

    # Feature 2: Image File Size (Proxy for visual complexity/texture)
    # We need to load the validation metadata to get file paths
    # We use load_cached_data=True to speed up loading
    val_df = load_and_process_df(Config.val_metadata_path, "val", load_cached_data=True)

    # Get file sizes
    file_sizes = []
    if "full_path" in val_df.columns:
        for path in val_df["full_path"]:
            try:
                file_sizes.append(os.path.getsize(path))
            except:
                file_sizes.append(0)
    else:
        file_sizes = [0] * len(val_df)

    file_sizes = np.array(file_sizes)

    # Calculate Correlations using numpy

    # Correlation: Error vs Number of Labels
    if len(np.unique(num_labels)) > 1:
        corr_labels = np.corrcoef(error_magnitude, num_labels)[0, 1]
    else:
        corr_labels = 0.0

    # Correlation: Error vs File Size
    if len(np.unique(file_sizes)) > 1:
        corr_size = np.corrcoef(error_magnitude, file_sizes)[0, 1]
    else:
        corr_size = 0.0

    print(f"Correlation (Error Magnitude vs Num Labels): {corr_labels:.4f}")
    print(f"Correlation (Error Magnitude vs File Size): {corr_size:.4f}")

    if abs(corr_labels) > 0.1:
        print("-> Observation: Error is correlated with the number of attributes.")

    # 4. Submission Generation
    # ---------------------------------------------------------
    THRESHOLD_SCORE = 0.606834287443573

    if best_val_f1 > THRESHOLD_SCORE:
        print(
            f"\nValidation score ({best_val_f1:.4f}) exceeds threshold ({THRESHOLD_SCORE})."
        )
        print("Generating submission for test set...")

        # Update the trainer's threshold to the optimized one
        trainer.best_threshold = best_thresh

        # Run prediction
        trainer.predict()

        print("Submission generated successfully.")
    else:
        print(
            f"\nValidation score ({best_val_f1:.4f}) does not meet threshold ({THRESHOLD_SCORE})."
        )
        print("Skipping submission generation.")


if __name__ == "__main__":
    run()
