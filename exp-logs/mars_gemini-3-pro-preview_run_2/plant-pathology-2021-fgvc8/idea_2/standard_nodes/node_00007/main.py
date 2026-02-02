import os
import sys
import numpy as np
import pandas as pd
import torch
import cv2
from tqdm.auto import tqdm

# Import library components
from library.config import Config
from library.utils import seed_everything, get_score
from library.dataset import get_loaders
from library.train import train_loop
from library.inference import predict_fn
from library.model import AppleDiseaseModel


def main():
    # 1. Setup and Configuration
    seed_everything(Config.SEED)

    # Adjust Config for the task requirements
    # We use 10 epochs which is sufficient for convergence on this dataset
    # while fitting well within the 2-hour limit on an A100.
    Config.EPOCHS = 10
    Config.DEBUG = False  # Ensure we use the full dataset for best performance

    # 2. Train the Model
    # train_loop handles the training process, saving the best model,
    # and returns the model with the best weights loaded.
    print("--- Starting Training ---")
    model = train_loop()

    # 3. Final Validation Assessment
    print("\n--- Starting Final Validation ---")
    # We need the validation loader. get_loaders returns (train, val, test)
    _, val_loader, _ = get_loaders()

    device = torch.device(Config.DEVICE)
    model.to(device)
    model.eval()

    all_preds = []
    all_targets = []

    # Run inference on validation set
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="Validating"):
            images = images.to(device)

            # Forward pass
            logits = model(images)
            probs = torch.sigmoid(logits)

            all_preds.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Compute Final Metric
    final_f1 = get_score(all_targets, all_preds, threshold=Config.THRESHOLD)
    print(f"Final Validation Metric: {final_f1}")

    # 4. Failure Analysis
    print("\n--- Performing Failure Analysis ---")
    # Calculate error magnitude per sample (Mean Absolute Error across classes)
    # shape: (N_samples,)
    errors = np.mean(np.abs(all_preds - all_targets), axis=1)

    # Extract input features for correlation analysis
    # We iterate through the validation dataframe to get file paths
    val_df = val_loader.dataset.df

    file_sizes = []
    widths = []
    heights = []

    print("Extracting image features...")
    for _, row in tqdm(val_df.iterrows(), total=len(val_df), desc="Analyzing Images"):
        file_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        # File Size
        try:
            f_size = os.path.getsize(file_path)
        except OSError:
            f_size = 0
        file_sizes.append(f_size)

        # Image Dimensions
        # We read the image to get actual dimensions
        img = cv2.imread(file_path)
        if img is not None:
            h, w, _ = img.shape
            widths.append(w)
            heights.append(h)
        else:
            widths.append(0)
            heights.append(0)

    # Calculate Correlations
    def calculate_correlation(x, y):
        if len(x) != len(y) or len(x) < 2:
            return 0.0
        return np.corrcoef(x, y)[0, 1]

    corr_size = calculate_correlation(errors, file_sizes)
    corr_width = calculate_correlation(errors, widths)
    corr_height = calculate_correlation(errors, heights)

    print(f"Correlation (Error vs File Size): {corr_size:.4f}")
    print(f"Correlation (Error vs Width): {corr_width:.4f}")
    print(f"Correlation (Error vs Height): {corr_height:.4f}")

    # 5. Submission Generation
    THRESHOLD = 0.9153778856820253

    if final_f1 > THRESHOLD:
        print(f"\nValidation metric ({final_f1}) exceeds threshold ({THRESHOLD}).")
        print("Generating submission...")
        # predict_fn loads the best model from disk and generates submission.csv
        predict_fn()
    else:
        print(
            f"\nValidation metric ({final_f1}) does not exceed threshold ({THRESHOLD})."
        )
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
