import sys
import os
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr

# Import from the provided library
from library.config import Config
from library.utils import set_seed, calculate_roc_auc, load_checkpoint
from library.model import SpatiotemporalResNet
from library.train import train_model
from library.inference import predict
from library.data import get_dataloaders


def run():
    # 1. Setup and Configuration
    # Set random seed for reproducibility
    set_seed(Config.SEED)

    # Configure for a fast baseline execution
    # We use a subset of data and a small number of epochs to ensure the script completes within 2 hours.
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = (
        5000  # Use 5000 samples to get a stable metric while remaining fast
    )
    Config.EPOCHS = 2  # Limit to 2 epochs for speed

    print("Execution Configuration:")
    print(f"  Debug Mode: {Config.DEBUG}")
    print(f"  Subset Size: {Config.DEBUG_SUBSET_SIZE}")
    print(f"  Epochs: {Config.EPOCHS}")
    print(f"  Device: {Config.DEVICE}")

    # 2. Model Training
    print("\n=== Starting Training Phase ===")
    # Train the model and retrieve the best validation score achieved
    _ = train_model(debug=Config.DEBUG, epochs=Config.EPOCHS)

    # 3. Validation Assessment & Failure Analysis
    print("\n=== Starting Validation & Failure Analysis Phase ===")

    # Load the validation data loader
    # We use the same debug settings to evaluate on the validation subset used during training logic
    _, val_loader, _ = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        debug=Config.DEBUG,
        debug_subset_size=Config.DEBUG_SUBSET_SIZE,
    )

    # Initialize model and load the best checkpoint
    device = torch.device(Config.DEVICE)
    model = SpatiotemporalResNet(pretrained=False).to(device)

    checkpoint_path = Config.MODEL_SAVE_PATH
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    checkpoint = load_checkpoint(model, filename=checkpoint_path, device=device)
    model.eval()

    # Containers for analysis
    all_preds = []
    all_targets = []
    meta_features = []
    errors = []

    # Inference Loop on Validation Set
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets_np = targets.numpy()

            # Forward pass
            logits = model(inputs)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            all_preds.extend(probs)
            all_targets.extend(targets_np)

            # Calculate absolute errors
            batch_errors = np.abs(targets_np - probs)
            errors.extend(batch_errors)

            # Extract meta-features for failure analysis
            # inputs shape: (B, 1, 6, 273, 256) -> Move to CPU
            imgs = inputs.cpu().numpy()

            for i in range(imgs.shape[0]):
                # Select the single channel: (6, 273, 256)
                img = imgs[i, 0]

                # Calculate basic statistics
                mean_val = np.mean(img)
                std_val = np.std(img)
                max_val = np.max(img)

                # Calculate Contrast (ON vs OFF panels)
                # ON panels: 0, 2, 4; OFF panels: 1, 3, 5
                on_panels = img[[0, 2, 4], :, :]
                off_panels = img[[1, 3, 5], :, :]

                mean_on = np.mean(on_panels)
                mean_off = np.mean(off_panels)
                contrast = mean_on - mean_off

                meta_features.append(
                    {
                        "mean": mean_val,
                        "std": std_val,
                        "max": max_val,
                        "contrast": contrast,
                    }
                )

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Calculate and Print Final Validation Metric
    final_metric = calculate_roc_auc(all_targets, all_preds)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis: Correlation between Error and Features
    print("\nFailure Analysis (Correlation with Prediction Error):")
    df_features = pd.DataFrame(meta_features)
    df_features["error"] = errors

    correlations = {}
    feature_cols = ["mean", "std", "max", "contrast"]

    for col in feature_cols:
        # Check for constant values to avoid division by zero in correlation
        if df_features[col].std() > 1e-9:
            corr, _ = pearsonr(df_features[col], df_features["error"])
            correlations[col] = corr
        else:
            correlations[col] = 0.0

    for feat, corr in correlations.items():
        print(f"  {feat}: {corr:.4f}")

    # 4. Submission Generation
    # Threshold defined in the task
    threshold = 0.5095077440787745

    if final_metric > threshold:
        print(f"\nValidation metric ({final_metric}) exceeds threshold ({threshold}).")
        print("Generating submission for the test set...")
        # Generate predictions for the full test set
        predict(debug=False)
    else:
        print(
            f"\nValidation metric ({final_metric}) does not exceed threshold ({threshold})."
        )
        print("Skipping submission generation.")


if __name__ == "__main__":
    run()
