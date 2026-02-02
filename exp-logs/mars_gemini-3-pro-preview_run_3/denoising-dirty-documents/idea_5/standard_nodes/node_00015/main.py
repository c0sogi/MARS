import sys
import os
import torch
import pandas as pd
import numpy as np
import warnings

# Add current directory to path to ensure library imports work correctly
sys.path.append(os.getcwd())

# Import from provided library
from library.config import Config
from library.utils import seed_everything, load_image
from library.train import run_training
from library.inference import generate_submission, predict_full_image
from library.model import RDN

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    print(f"Running orchestration script on {device}...")

    # 2. Training
    # We limit epochs to 5 to ensure the script completes quickly as a baseline.
    # We use cached data if available to save preprocessing time.
    print("\n--- Starting Training ---")
    FAST_EPOCHS = 5
    run_training(
        epochs=FAST_EPOCHS, batch_size=Config.BATCH_SIZE, load_cached_data=True
    )

    # 3. Validation (Full Image RMSE)
    print("\n--- Starting Validation ---")

    # Load the best model saved during training
    model = RDN().to(device)
    if os.path.exists(Config.MODEL_PATH):
        model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
        print("Loaded best model checkpoint.")
    else:
        print("Warning: Model checkpoint not found. Using random weights.")

    model.eval()

    # Load Validation Metadata
    if not os.path.exists(Config.VAL_METADATA):
        raise FileNotFoundError(
            f"Validation metadata not found at {Config.VAL_METADATA}"
        )

    val_df = pd.read_csv(Config.VAL_METADATA)

    total_squared_error = 0.0
    total_pixels = 0

    # Lists for failure analysis
    img_rmses = []
    img_means = []
    img_stds = []

    with torch.no_grad():
        for _, row in val_df.iterrows():
            input_path = os.path.join(Config.INPUT_DIR, row["input_path"])
            target_path = os.path.join(Config.INPUT_DIR, row["target_path"])

            # Load images
            img_in = load_image(input_path)
            img_tar = load_image(target_path)

            # Predict (Full image inference)
            # predict_full_image handles tensor conversion and device movement
            img_pred = predict_full_image(model, input_path, device)

            # Clip predictions to valid range [0, 1]
            img_pred = np.clip(img_pred, 0, 1)

            # Calculate Squared Error for this image
            diff = img_pred - img_tar
            squared_error = np.sum(diff**2)
            num_pixels = img_tar.size

            total_squared_error += squared_error
            total_pixels += num_pixels

            # Record stats for failure analysis
            mse_img = squared_error / num_pixels
            rmse_img = np.sqrt(mse_img)

            img_rmses.append(rmse_img)
            img_means.append(np.mean(img_in))
            img_stds.append(np.std(img_in))

    # Calculate Global RMSE
    final_mse = total_squared_error / total_pixels
    final_rmse = np.sqrt(final_mse)

    # REQUIRED OUTPUT
    print(f"Final Validation Metric: {final_rmse}")

    # 4. Failure Analysis
    print("\n--- Failure Analysis ---")
    if len(img_rmses) > 1:
        # Calculate correlations using numpy
        # Correlation between Error and Image Brightness
        corr_mean = np.corrcoef(img_rmses, img_means)[0, 1]
        # Correlation between Error and Image Contrast/Complexity
        corr_std = np.corrcoef(img_rmses, img_stds)[0, 1]

        print(f"Correlation (RMSE vs Input Mean Intensity): {corr_mean}")
        print(f"Correlation (RMSE vs Input Std Dev): {corr_std}")

        if abs(corr_mean) > 0.3:
            print("Observation: Model performance is sensitive to image brightness.")
        if abs(corr_std) > 0.3:
            print("Observation: Model performance is sensitive to image complexity.")
    else:
        print("Insufficient validation samples for correlation analysis.")

    # 5. Submission
    # Threshold defined in task description
    THRESHOLD = 0.011577641381826402

    if final_rmse < THRESHOLD:
        print(f"\nValidation metric {final_rmse} meets threshold {THRESHOLD}.")
        print("Generating submission for test set...")
        generate_submission(
            model_path=Config.MODEL_PATH, submission_output=Config.SUBMISSION_PATH
        )
    else:
        print(f"\nValidation metric {final_rmse} does not meet threshold {THRESHOLD}.")
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
