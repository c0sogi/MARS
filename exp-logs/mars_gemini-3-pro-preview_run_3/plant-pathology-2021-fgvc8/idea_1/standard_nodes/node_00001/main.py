import os
import torch
import pandas as pd
import numpy as np
import sys

# Import from the provided library files
from library.config import Config
from library.utils import set_seed, calculate_f1_score
from library.engine import run_training, run_inference
from library.dataset import get_loaders


def main():
    # ==========================================
    # 1. Configuration and Setup
    # ==========================================
    # Ensure reproducibility
    set_seed(Config.SEED)

    # Detect device (cuda is expected given the A100 environment)
    device = Config.DEVICE
    print(f"Orchestrating workflow on device: {device}")

    # ==========================================
    # 2. Training
    # ==========================================
    # We run training for 5 epochs. This is sufficient for a ResNet18
    # to converge to a strong baseline on this dataset size using an A100,
    # and ensures the script completes well within the time limit.
    print("\n--- Starting Training ---")
    model = run_training(epochs=5, device=device)

    # ==========================================
    # 3. Validation & Metric Calculation
    # ==========================================
    print("\n--- Starting Validation & Failure Analysis ---")

    # Load the validation loader. We use debug_sample_size=None to ensure
    # we validate on the entire hold-out set as required.
    _, val_loader, _ = get_loaders(debug_sample_size=None)

    model.eval()

    all_logits = []
    all_targets = []

    # Lists to store image features for failure analysis
    # We will compute these on the fly from the tensors
    brightness_stats = []
    contrast_stats = []

    with torch.no_grad():
        for images, targets, _ in val_loader:
            images = images.to(device)
            targets = targets.to(device)

            # Perform Inference
            # Using AMP for consistency, though less critical for inference
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                logits = model(images)

            # Store outputs for metric calculation
            all_logits.append(logits.cpu())
            all_targets.append(targets.cpu())

            # --- Feature Extraction for Failure Analysis ---
            # images shape: (Batch, Channels, Height, Width)
            # We calculate stats on the normalized tensor.
            # Brightness: Mean pixel value across all channels and pixels
            # Contrast: Standard deviation across all channels and pixels

            # Calculate mean/std per image in the batch
            batch_means = images.mean(dim=(1, 2, 3)).cpu().numpy()
            batch_stds = images.std(dim=(1, 2, 3)).cpu().numpy()

            brightness_stats.extend(batch_means)
            contrast_stats.extend(batch_stds)

    # Concatenate all batches
    all_logits = torch.cat(all_logits)
    all_targets = torch.cat(all_targets)

    # Calculate and Print the Required Metric
    val_f1 = calculate_f1_score(all_logits, all_targets)
    # Printing full precision as requested
    print(f"Final Validation Metric: {val_f1}")

    # ==========================================
    # 4. Failure Analysis
    # ==========================================
    # Calculate Error Magnitude: Mean Absolute Error (MAE) per sample
    # We compare the predicted probabilities (sigmoid of logits) with the binary targets
    probs = torch.sigmoid(all_logits).numpy()
    targets_np = all_targets.numpy()

    # MAE per image (averaged across classes)
    errors = np.mean(np.abs(probs - targets_np), axis=1)

    # Create a DataFrame to analyze correlations
    df_analysis = pd.DataFrame(
        {"error": errors, "brightness": brightness_stats, "contrast": contrast_stats}
    )

    # Calculate Pearson Correlation
    corr_brightness = df_analysis["error"].corr(df_analysis["brightness"])
    corr_contrast = df_analysis["error"].corr(df_analysis["contrast"])

    print("\n--- Failure Analysis Report ---")
    print(
        f"Correlation between Error Magnitude and Image Brightness: {corr_brightness}"
    )
    print(f"Correlation between Error Magnitude and Image Contrast: {corr_contrast}")

    # ==========================================
    # 5. Submission
    # ==========================================
    print("\n--- Generating Submission ---")
    # run_inference handles loading the test set, predicting, and saving to CSV
    run_inference(model, device=device)

    print("Workflow completed successfully.")


if __name__ == "__main__":
    main()
