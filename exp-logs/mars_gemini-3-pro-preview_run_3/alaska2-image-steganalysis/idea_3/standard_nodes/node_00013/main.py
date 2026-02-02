import os
import torch
import pandas as pd
import numpy as np
import sys

# Import provided library functions
from library.utils import seed_everything, weighted_auc_score
from library.engine import train_model, generate_submission
from library.model import MonoResidualEfficientNet
from library.dataset import get_dataloaders


def main():
    # 1. Setup
    seed_everything(42)
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    print(f"Running on device: {device}")

    # 2. Training
    # We use 5 epochs and batch size 32 to allow the larger B2 model to converge (Cite solution_lesson_node_00012).
    print("\n=== Starting Training ===")
    best_model_path = train_model(
        device_name=device_name,
        epochs=5,
        batch_size=32,
        learning_rate=1e-3,
        patience=3,
        num_workers=4,
        seed=42,
    )

    # 3. Validation & Metric Calculation
    print("\n=== Starting Validation ===")

    # Re-initialize model and load best weights
    # pretrained=False because we are loading our own trained weights
    model = MonoResidualEfficientNet(
        model_name="efficientnet_b2", pretrained=False, num_classes=1
    )
    state_dict = torch.load(best_model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Get Validation DataLoader
    # We don't need the train loader here
    _, val_loader = get_dataloaders(
        input_dir="./input",
        metadata_dir="./metadata",
        batch_size=32,
        num_workers=4,
        seed=42,
    )

    y_true = []
    y_pred = []

    # Lists to store image stats for failure analysis
    pixel_means = []
    pixel_stds = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)
            outputs = outputs.squeeze(1)
            probs = torch.sigmoid(outputs)

            # Store predictions and labels
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(probs.cpu().numpy())

            # Compute image statistics for failure analysis
            # inputs shape: (Batch, 1, H, W)
            # Flatten spatial dimensions to compute mean/std per image
            batch_flat = inputs.view(inputs.size(0), -1)

            # Mean and Std per image in the batch
            b_means = batch_flat.mean(dim=1).cpu().numpy()
            b_stds = batch_flat.std(dim=1).cpu().numpy()

            pixel_means.extend(b_means)
            pixel_stds.extend(b_stds)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate Final Metric
    final_metric = weighted_auc_score(y_true, y_pred)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    print("\n=== Failure Analysis ===")

    # Calculate absolute error
    errors = np.abs(y_true - y_pred)

    # Create DataFrame for analysis
    analysis_df = pd.DataFrame(
        {"error": errors, "pixel_mean": pixel_means, "pixel_std": pixel_stds}
    )

    # Calculate correlations
    corr_mean = analysis_df["error"].corr(analysis_df["pixel_mean"])
    corr_std = analysis_df["error"].corr(analysis_df["pixel_std"])

    print("Correlation between Error Magnitude and Input Features:")
    print(f"  Pixel Mean (Brightness): {corr_mean:.6f}")
    print(f"  Pixel Std (Contrast/Texture): {corr_std:.6f}")

    # 5. Submission
    print("\n=== Submission Generation ===")
    THRESHOLD = 0.8303656056

    if final_metric > THRESHOLD:
        print(f"Validation metric ({final_metric}) exceeds threshold ({THRESHOLD}).")
        print("Generating submission file...")
        generate_submission(
            model_path=best_model_path, batch_size=32, device_name=device_name
        )
    else:
        print(
            f"Validation metric ({final_metric}) does not exceed threshold ({THRESHOLD})."
        )
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
