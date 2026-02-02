import os
import sys
import torch
import torch.optim as optim
import numpy as np
import warnings

# Filter warnings to keep output clean
warnings.filterwarnings("ignore")

# Import library modules
from library.config import Config
from library.dataset import DenoisingDataset
from library.model import DS_AG_CAC_ResUNet
from library.loss import MultiScaleMSELoss
from library.train import train_one_epoch, validate
from library.utils import seed_everything, get_device, calculate_rmse
from library.inference import generate_submission, predict_tiled


def run_pipeline():
    # -------------------------------------------------------------------------
    # 1. Configuration Adjustments for Fast Baseline
    # -------------------------------------------------------------------------
    # Adjust hyperparameters to ensure execution finishes within 2 hours
    Config.NUM_EPOCHS = 15
    Config.PATCHES_PER_IMAGE = 50  # Reduced sampling density for speed
    Config.BATCH_SIZE = 48  # Safe batch size for A100

    # Disable TTA during training validation to speed up epoch feedback.
    # We will enable it for the final rigorous evaluation.
    Config.USE_TTA = False

    # -------------------------------------------------------------------------
    # 2. Setup
    # -------------------------------------------------------------------------
    seed_everything(Config.SEED)
    device = get_device()
    print(f"Running on device: {device}")

    # -------------------------------------------------------------------------
    # 3. Data Loading
    # -------------------------------------------------------------------------
    print("Loading datasets...")
    # Load cached data is True as requested to utilize pre-processed .npy files
    train_dataset = DenoisingDataset(
        Config.TRAIN_METADATA_PATH, mode="train", load_cached_data=True
    )
    val_dataset = DenoisingDataset(
        Config.VAL_METADATA_PATH, mode="val", load_cached_data=True
    )

    # Train loader with shuffling and pinning memory
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Validation loader (Batch size 1 is required for full-image inference)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=Config.NUM_WORKERS
    )

    # -------------------------------------------------------------------------
    # 4. Model Initialization
    # -------------------------------------------------------------------------
    print(f"Initializing model: {Config.MODEL_NAME}...")
    model = DS_AG_CAC_ResUNet().to(device)

    # -------------------------------------------------------------------------
    # 5. Optimization Setup
    # -------------------------------------------------------------------------
    # Define weights for Deep Supervision (decaying weights for aux heads)
    loss_weights = [1.0, 0.5, 0.5, 0.5] if Config.USE_DEEP_SUPERVISION else [1.0]
    criterion = MultiScaleMSELoss(weights=loss_weights).to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.NUM_EPOCHS, eta_min=Config.MIN_LEARNING_RATE
    )

    # -------------------------------------------------------------------------
    # 6. Training Loop
    # -------------------------------------------------------------------------
    print(f"Starting training for {Config.NUM_EPOCHS} epochs...")
    best_rmse = float("inf")

    for epoch in range(Config.NUM_EPOCHS):
        # Train Step
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validation Step (Fast mode, no TTA)
        val_rmse = validate(model, val_loader, device)

        # Scheduler Step
        scheduler.step()

        print(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | Train Loss: {train_loss:.6f} | Val RMSE (No TTA): {val_rmse:.6f}"
        )

        # Checkpoint Best Model
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            torch.save(model.state_dict(), Config.MODEL_CHECKPOINT_PATH)
            print("  -> New Best Model Saved!")

    print(f"Training complete. Best Val RMSE (No TTA): {best_rmse:.6f}")

    # -------------------------------------------------------------------------
    # 7. Final Validation & Failure Analysis
    # -------------------------------------------------------------------------
    print("\n--- Starting Final Validation & Failure Analysis ---")

    # Load the best model saved during training
    if os.path.exists(Config.MODEL_CHECKPOINT_PATH):
        model.load_state_dict(
            torch.load(Config.MODEL_CHECKPOINT_PATH, map_location=device)
        )
    else:
        print("Warning: Checkpoint not found, using current model state.")

    model.eval()

    # Enable TTA for the final metric calculation to maximize performance
    Config.USE_TTA = True
    print("TTA Enabled for final evaluation.")

    # Data containers for failure analysis
    image_rmses = []
    input_means = []
    input_stds = []

    total_rmse_accum = 0.0

    with torch.no_grad():
        for i, (noisy, clean, img_id) in enumerate(val_loader):
            noisy = noisy.to(device)
            # clean is ground truth (1, 1, H, W)

            # Predict Clean Image using Tiled Inference + TTA
            pred_clean = predict_tiled(model, noisy, device)

            # Calculate RMSE for this specific image
            rmse = calculate_rmse(clean, pred_clean)

            image_rmses.append(rmse)
            total_rmse_accum += rmse

            # Calculate Input Statistics (from noisy image) for correlation analysis
            noisy_np = noisy.cpu().numpy().flatten()
            input_means.append(np.mean(noisy_np))
            input_stds.append(np.std(noisy_np))

    # Calculate Final Metric (Mean of Image RMSEs)
    final_metric = total_rmse_accum / len(val_loader)

    # Print Final Validation Metric (Full Precision)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis: Calculate Correlations
    # We use numpy's corrcoef to avoid extra dependencies like scipy if not present
    if len(image_rmses) > 1:
        corr_mean = np.corrcoef(image_rmses, input_means)[0, 1]
        corr_std = np.corrcoef(image_rmses, input_stds)[0, 1]
    else:
        corr_mean, corr_std = 0.0, 0.0

    print("\nFailure Analysis - Error Correlations:")
    print(f"Correlation (RMSE vs Input Mean Intensity): {corr_mean:.6f}")
    print(f"Correlation (RMSE vs Input Std Dev): {corr_std:.6f}")

    # -------------------------------------------------------------------------
    # 8. Submission Generation
    # -------------------------------------------------------------------------
    threshold = 0.0076658159

    if final_metric < threshold:
        print(
            f"\nValidation metric {final_metric} < {threshold}. Generating submission..."
        )
        generate_submission(Config.MODEL_CHECKPOINT_PATH, Config.SUBMISSION_PATH)
    else:
        print(f"\nValidation metric {final_metric} >= {threshold}. Submission skipped.")


if __name__ == "__main__":
    run_pipeline()
