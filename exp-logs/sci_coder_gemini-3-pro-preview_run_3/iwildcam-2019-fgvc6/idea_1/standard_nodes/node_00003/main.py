import os
import sys
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score

# Ensure library is in path
sys.path.append(os.getcwd())

from library.config import Config
from library.trainer import Trainer
from library.inference import predict_and_submit


def main():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    # Optimize for A100 GPU and speed requirements
    # A100 40GB allows for larger batch sizes, speeding up training
    Config.BATCH_SIZE = 256
    Config.NUM_WORKERS = 8
    # Increase epochs for better convergence with EfficientNet and OneCycleLR
    Config.EPOCHS = 8

    # Ensure reproducibility
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)

    print("Configuration updated for fast baseline:")
    print(f"  Batch Size: {Config.BATCH_SIZE}")
    print(f"  Epochs: {Config.EPOCHS}")
    print(f"  Device: {Config.DEVICE}")

    # ==========================================
    # 2. Training
    # ==========================================
    print("\nInitializing Trainer...")
    trainer = Trainer()

    print("Starting Training...")
    trainer.fit()

    # ==========================================
    # 3. Validation & Failure Analysis
    # ==========================================
    print("\nStarting Validation and Failure Analysis...")

    # Use the trainer's model and val_loader
    model = trainer.model
    val_loader = trainer.val_loader
    device = Config.DEVICE

    # Load best weights if available (Trainer saves best model to Config.MODEL_SAVE_PATH)
    if os.path.exists(Config.MODEL_SAVE_PATH):
        print("Loading best model for validation analysis...")
        model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))

    model.eval()

    all_preds = []
    all_labels = []

    # Lists for failure analysis
    error_flags = []
    pixel_means = []
    pixel_stds = []

    with torch.no_grad():
        for images, labels, _ in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Inference
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            # Store for Metric
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # --- Failure Analysis Data Collection ---
            # Calculate simple features from the normalized image tensor
            # images shape: [Batch, Channels, Height, Width]

            # Mean intensity per image (across channels and spatial dims)
            b_means = images.mean(dim=(1, 2, 3)).cpu().numpy()
            # Std deviation per image
            b_stds = images.std(dim=(1, 2, 3)).cpu().numpy()

            # Error: 1 if prediction is wrong, 0 if correct
            b_errors = (preds != labels).cpu().numpy().astype(int)

            pixel_means.extend(b_means)
            pixel_stds.extend(b_stds)
            error_flags.extend(b_errors)

    # --- Metric Calculation ---
    # Print the full precision of the validation metric
    final_f1 = f1_score(all_labels, all_preds, average="macro")
    print(f"Final Validation Metric: {final_f1}")

    # --- Failure Analysis Correlation ---
    analysis_df = pd.DataFrame(
        {"Error": error_flags, "PixelMean": pixel_means, "PixelStd": pixel_stds}
    )

    # Calculate correlations
    corr_mean = analysis_df["Error"].corr(analysis_df["PixelMean"])
    corr_std = analysis_df["Error"].corr(analysis_df["PixelStd"])

    print("\nFailure Analysis - Feature Correlations with Error:")
    print(f"Correlation (Error vs PixelMean): {corr_mean}")
    print(f"Correlation (Error vs PixelStd): {corr_std}")

    # ==========================================
    # 4. Submission
    # ==========================================
    print("\nGenerating Submission...")
    # predict_and_submit handles loading the best model and generating the CSV
    predict_and_submit()


if __name__ == "__main__":
    main()
