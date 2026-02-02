import sys
import os
import cv2
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

# Import provided library modules
from library.config import Config, seed_everything
from library.trainer import Trainer


def main():
    # --- 1. Configuration & Setup ---
    # Set random seeds for reproducibility
    seed_everything(Config.SEED)

    # Ensure necessary directories exist (handled by Config, but double checking)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # --- 2. Model Training ---
    # Initialize the Trainer which sets up the ConvNeXt model, optimizer, and loaders
    trainer = Trainer()

    # Run the training loop
    trainer.fit()

    # --- 3. Validation Assessment ---
    # Load the best model checkpoint saved during training
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    if not os.path.exists(best_model_path):
        print(f"Error: Checkpoint not found at {best_model_path}")
        return

    # Load weights
    checkpoint = torch.load(best_model_path, map_location=Config.DEVICE)
    trainer.model.load_state_dict(checkpoint["state_dict"])
    trainer.model.eval()

    # Run inference on the full validation set
    val_loader = trainer.val_loader
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(Config.DEVICE)

            # Use AMP for faster inference
            with torch.cuda.amp.autocast(enabled=Config.USE_AMP):
                outputs = trainer.model(images)
                preds = torch.sigmoid(outputs)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.numpy())

    # Concatenate results
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)

    # Binarize predictions using the defined threshold
    binary_preds = (all_preds > Config.THRESHOLD).astype(int)

    # Calculate and print the final metric
    final_f1 = f1_score(all_targets, binary_preds, average="macro")
    print(f"Final Validation Metric: {final_f1}")

    # --- 4. Failure Analysis ---
    # Calculate error magnitude (sum of absolute differences per sample)
    # This represents how many labels were incorrectly predicted for each image
    error_magnitude = np.sum(np.abs(all_targets - binary_preds), axis=1)

    # Load validation metadata to retrieve file paths
    df_val = pd.read_csv(Config.VAL_METADATA)

    # Extract input features for correlation analysis
    widths = []
    heights = []
    file_sizes = []

    # Iterate through validation samples to gather image metadata
    # (Data loader preserves order when shuffle=False)
    for idx, row in df_val.iterrows():
        file_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        # Get file size
        try:
            fsize = os.path.getsize(file_path)
        except OSError:
            fsize = 0
        file_sizes.append(fsize)

        # Get image dimensions
        try:
            img = cv2.imread(file_path)
            if img is not None:
                h, w, _ = img.shape
                widths.append(w)
                heights.append(h)
            else:
                widths.append(0)
                heights.append(0)
        except Exception:
            widths.append(0)
            heights.append(0)

    # Create a DataFrame for analysis
    analysis_df = pd.DataFrame(
        {
            "error": error_magnitude,
            "width": widths,
            "height": heights,
            "file_size": file_sizes,
        }
    )

    # Calculate and print correlations
    print("\nFailure Analysis - Correlation with Error Magnitude:")
    correlations = analysis_df.corr()["error"].drop("error")
    print(correlations)

    # --- 5. Submission Generation ---
    # Check if the model meets the performance requirement
    THRESHOLD_SCORE = 0.918587190946815

    if final_f1 > THRESHOLD_SCORE:
        # Generate predictions on the test set
        trainer.predict()
    else:
        print(
            f"Validation metric {final_f1} is not higher than {THRESHOLD_SCORE}. Submission skipped."
        )


if __name__ == "__main__":
    main()
