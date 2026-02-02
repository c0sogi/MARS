import os
import sys
import torch
import numpy as np
from scipy.stats import pearsonr

# Import library components
from library.config import Config
from library.train import run_training
from library.inference import run_inference
from library.model import CSKResUNet
from library.dataset import DenoisingDataset
from library.utils import set_seed, get_device


def main():
    # 1. Configuration Overrides for Fast Baseline
    # Limit training duration and sampling density
    Config.NUM_EPOCHS = 10
    Config.PATCHES_PER_IMAGE = 50
    Config.BATCH_SIZE = 32

    # Set submission path to required location
    submission_dir = "./submission"
    os.makedirs(submission_dir, exist_ok=True)
    Config.SUBMISSION_PATH = os.path.join(submission_dir, "submission.csv")

    # Set seed for reproducibility
    set_seed(Config.SEED)

    # 2. Run Training
    # Trains the model and saves the best checkpoint to Config.MODEL_SAVE_PATH
    run_training()

    # 3. Validation & Failure Analysis
    device = get_device()
    model = CSKResUNet().to(device)

    # Load the best model
    if not os.path.exists(Config.MODEL_SAVE_PATH):
        print(f"Error: Model checkpoint not found at {Config.MODEL_SAVE_PATH}")
        return

    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    model.eval()

    # Initialize Validation Dataset
    val_dataset = DenoisingDataset(
        metadata_path=Config.VAL_METADATA,
        root_dir=Config.INPUT_DIR,
        mode="val",
        load_cached_data=True,
    )

    # Containers for analysis
    all_clean_true = []
    all_clean_pred = []

    img_rmses = []
    img_means = []
    img_stds = []

    # Validation Loop
    with torch.no_grad():
        for i in range(len(val_dataset)):
            # Get sample (noisy, clean, id)
            noisy, clean, _ = val_dataset[i]

            # Move to device (Add batch dim: 1, 1, H, W)
            noisy_dev = noisy.unsqueeze(0).to(device)

            # Predict Noise Residual
            noise_pred = model(noisy_dev)

            # Reconstruct Clean Image: Clean = Noisy - Noise
            clean_pred_dev = noisy_dev - noise_pred
            clean_pred_dev = torch.clamp(clean_pred_dev, 0.0, 1.0)

            # Move to CPU
            clean_pred = clean_pred_dev.squeeze(0).cpu()

            # Flatten for metric calculation
            clean_flat = clean.numpy().flatten()
            pred_flat = clean_pred.numpy().flatten()

            # Store for Global RMSE
            all_clean_true.append(clean_flat)
            all_clean_pred.append(pred_flat)

            # Store for Failure Analysis
            # Error magnitude (RMSE per image)
            mse_img = np.mean((clean_flat - pred_flat) ** 2)
            rmse_img = np.sqrt(mse_img)
            img_rmses.append(rmse_img)

            # Input features
            noisy_flat = noisy.numpy().flatten()
            img_means.append(np.mean(noisy_flat))
            img_stds.append(np.std(noisy_flat))

    # Calculate Final Validation Metric
    y_true_total = np.concatenate(all_clean_true)
    y_pred_total = np.concatenate(all_clean_pred)

    global_mse = np.mean((y_true_total - y_pred_total) ** 2)
    final_rmse = np.sqrt(global_mse)

    print(f"Final Validation Metric: {final_rmse}")

    # Failure Analysis
    if len(img_rmses) > 1:
        corr_mean, _ = pearsonr(img_rmses, img_means)
        corr_std, _ = pearsonr(img_rmses, img_stds)

        print(f"Correlation (Error vs Input Mean): {corr_mean}")
        print(f"Correlation (Error vs Input Std): {corr_std}")

    # 4. Submission
    THRESHOLD = 0.0076658159
    if final_rmse < THRESHOLD:
        run_inference()


if __name__ == "__main__":
    main()
