import os
import sys
import numpy as np
import torch
import pandas as pd
from scipy.stats import pearsonr
from torch.utils.data import DataLoader

# Ensure library is in path
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything, calculate_rmse, load_checkpoint
from library.train import train_model
from library.dataset import TextDenoisingDataset, get_transforms
from library.model import CoSPResUNet
from library.predict import predict_tiled, generate_submission


def run():
    # 1. Setup and Configuration
    Config.initialize()
    seed_everything(Config.SEED)

    # --- Fast Baseline Overrides ---
    # We reduce epochs and sampling density to ensure the run completes quickly
    # while maintaining enough capacity to learn the denoising task.
    # 92 images * 40 patches = 3680 samples per epoch.
    Config.EPOCHS = 15
    Config.PATCHES_PER_EPOCH = 40

    print("Starting Fast Baseline Run...")
    print(
        f"Configuration: Epochs={Config.EPOCHS}, Patches/Img={Config.PATCHES_PER_EPOCH}, Batch={Config.BATCH_SIZE}"
    )

    # 2. Training
    # Explicitly pass modified Config values as arguments
    train_model(
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE,
        learning_rate=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
        patience=5,  # Reduced patience for fast baseline
        num_workers=Config.NUM_WORKERS,
    )

    # 3. Validation & Failure Analysis
    print("\n--- Starting Validation & Failure Analysis ---")

    device = Config.DEVICE
    model = CoSPResUNet().to(device)

    # Load the best model saved during training
    epoch, loss = load_checkpoint(model, filename=Config.MODEL_SAVE_PATH, device=device)
    print(f"Loaded best model from Epoch {epoch} (Val Loss: {loss:.6f})")

    model.eval()

    # Initialize Validation Dataset
    val_dataset = TextDenoisingDataset(
        metadata_path=Config.VAL_METADATA, mode="val", transform=get_transforms("val")
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,  # Process one image at a time for accurate per-image RMSE
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    image_rmses = []
    input_means = []
    input_stds = []

    total_rmse = 0.0
    count = 0

    print(f"Validating on {len(val_dataset)} images...")

    with torch.no_grad():
        for i, (noisy_img, clean_img, img_id) in enumerate(val_loader):
            noisy_img = noisy_img.to(device)
            clean_img = clean_img.to(device)

            # Use tiled inference for robust handling of varying image sizes
            # This matches the inference strategy used for submission
            pred_clean = predict_tiled(
                model,
                noisy_img,
                patch_size=Config.PATCH_SIZE,
                overlap=Config.TILE_OVERLAP,
                device=device,
            )

            # Calculate RMSE for this specific image
            val_rmse = calculate_rmse(pred_clean, clean_img)

            # Store metrics for failure analysis
            image_rmses.append(val_rmse)

            # Calculate Input Features (on CPU)
            noisy_np = noisy_img.cpu().numpy().squeeze()
            input_means.append(np.mean(noisy_np))
            input_stds.append(np.std(noisy_np))

            total_rmse += val_rmse
            count += 1

    # Compute final aggregate metric
    final_metric = total_rmse / count

    # REQUIRED OUTPUT: Print the validation metric
    print(f"Final Validation Metric: {final_metric}")

    # Perform Failure Analysis
    if len(image_rmses) > 1:
        # Correlation with Mean Intensity (Brightness)
        corr_mean, _ = pearsonr(image_rmses, input_means)
        # Correlation with Standard Deviation (Contrast)
        corr_std, _ = pearsonr(image_rmses, input_stds)

        print("\nFailure Analysis:")
        print(f"Correlation (Error vs Input Mean Intensity): {corr_mean:.6f}")
        print(f"Correlation (Error vs Input Contrast/Std): {corr_std:.6f}")

    # 4. Submission Generation
    TARGET_THRESHOLD = 0.0076658159

    if final_metric < TARGET_THRESHOLD:
        print(
            f"\nValidation metric {final_metric:.8f} is better than threshold {TARGET_THRESHOLD}."
        )
        print("Generating submission file...")
        generate_submission(
            model_path=Config.MODEL_SAVE_PATH,
            output_path=Config.SUBMISSION_PATH,
            batch_size=1,
            device=device,
        )
    else:
        print(
            f"\nValidation metric {final_metric:.8f} did not meet threshold {TARGET_THRESHOLD}."
        )
        print("Skipping submission generation.")


if __name__ == "__main__":
    run()
