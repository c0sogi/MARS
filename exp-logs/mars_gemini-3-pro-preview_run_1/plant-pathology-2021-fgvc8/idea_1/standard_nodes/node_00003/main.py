import os
import sys
import numpy as np
import pandas as pd
import torch
from PIL import Image

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, calculate_score
from library.data_loader import get_dataloaders
from library.network import AppleDiseaseModel
from library.trainer import Trainer


def main():
    # 1. Setup & Reproducibility
    Config.setup_reproducibility()
    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # 2. Data Loading
    # Load cached data if available to speed up the process
    # We use the full dataset as 12k images is manageable on A100 within the time limit
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=True, debug=Config.DEBUG
    )

    # 3. Model Initialization
    model = AppleDiseaseModel(pretrained=True)
    model.to(device)

    # 4. Training
    # Initialize Trainer with the model and dataloaders
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=Config.EPOCHS,
    )

    # Execute training loop
    trainer.fit()

    # 5. Validation Assessment
    print("\n=== Validation Assessment ===")

    # Load the best model weights saved during training
    if os.path.exists(trainer.best_model_path):
        model.load_state_dict(torch.load(trainer.best_model_path, map_location=device))
    else:
        print("Warning: Best model not found. Using current weights.")

    model.eval()

    val_preds = []
    val_targets = []

    # Run inference on the validation set
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            # Forward pass
            logits = model(images)
            probs = torch.sigmoid(logits)

            val_preds.append(probs.cpu().numpy())
            val_targets.append(targets.numpy())

    val_preds = np.concatenate(val_preds, axis=0)
    val_targets = np.concatenate(val_targets, axis=0)

    # Compute and print the required metric
    final_score = calculate_score(
        val_targets, val_preds, threshold=0.5, average="macro"
    )
    print(f"Final Validation Metric: {final_score}")

    # 6. Failure Analysis
    print("\n=== Failure Analysis ===")

    # Calculate error magnitude (Hamming distance per sample)
    # Convert probabilities to binary predictions
    val_preds_bin = (val_preds > 0.5).astype(int)
    # Sum of absolute differences between prediction and target vectors
    error_magnitude = np.sum(np.abs(val_preds_bin - val_targets), axis=1)

    # Retrieve metadata from the dataset
    val_df = val_loader.dataset.df.copy()

    # Extract image dimensions for correlation analysis
    # This reads headers of validation images
    widths = []
    heights = []

    for full_path in val_df["full_path"]:
        try:
            with Image.open(full_path) as img:
                w, h = img.size
                widths.append(w)
                heights.append(h)
        except Exception:
            # Fallback for any corrupt images
            widths.append(0)
            heights.append(0)

    val_df["width"] = widths
    val_df["height"] = heights
    # Avoid division by zero
    val_df["aspect_ratio"] = np.array(widths) / (np.array(heights) + 1e-6)
    val_df["error"] = error_magnitude

    # Calculate and print correlations
    print("Correlation between Model Error and Input Features:")
    for feature in ["width", "height", "aspect_ratio"]:
        feat_values = val_df[feature].values
        if np.std(feat_values) > 0:
            # Calculate Pearson correlation coefficient
            corr = np.corrcoef(val_df["error"].values, feat_values)[0, 1]
            print(f"Correlation with {feature}: {corr:.6f}")
        else:
            print(f"Correlation with {feature}: N/A (no variance)")

    # 7. Submission
    print("\n=== Generating Submission ===")

    TARGET_SCORE = 0.918587190946815
    if final_score > TARGET_SCORE:
        print(
            f"Validation score ({final_score:.6f}) exceeds target ({TARGET_SCORE:.6f}). Generating submission..."
        )
        # The trainer's predict method handles test inference and file saving
        trainer.predict(test_loader)
    else:
        print(
            f"Validation score ({final_score:.6f}) did not exceed target ({TARGET_SCORE:.6f}). Skipping submission."
        )

    print("Runfile execution completed.")


if __name__ == "__main__":
    main()
