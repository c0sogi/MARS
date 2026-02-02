import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from scipy.stats import pearsonr

# Import library modules
import library.config as config
from library.utils import seed_everything, save_checkpoint
from library.model import ICResUNet
from library.dataset import DenoisingDataset
from library.train import train_one_epoch, validate, predict_tiled
import library.inference  # Import module to patch TTA if needed

# =============================================================================
# Configuration Overrides for Fast Execution
# =============================================================================
# Time limit is critical (~6 mins). We reduce workload significantly.
config.NUM_EPOCHS = 1
config.SAMPLES_PER_IMAGE = 15  # Reduced from 100 to 15 to ensure < 1 min training
config.BATCH_SIZE = 32  # Increased batch size for throughput


# Monkey-patch TTA to speed up submission inference if triggered
# (Replaces 4x inference with 1x inference)
def fast_predict_no_tta(model, noisy_tensor):
    return predict_tiled(model, noisy_tensor)


library.inference.predict_with_tta = fast_predict_no_tta


def run():
    # 1. Setup
    seed_everything(config.SEED)
    device = config.DEVICE
    print(f"Initializing run on {device} with {config.NUM_EPOCHS} epoch(s)...")

    # 2. Data Loading
    # Load cached data to save time
    train_dataset = DenoisingDataset(
        mode="train", load_cached_data=True, samples_per_image=config.SAMPLES_PER_IMAGE
    )
    val_dataset = DenoisingDataset(mode="val", load_cached_data=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    # Validation loader (Batch size 1 for varying image sizes)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)

    # 3. Model Initialization
    model = ICResUNet().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    # 4. Training Loop
    best_rmse = float("inf")

    for epoch in range(1, config.NUM_EPOCHS + 1):
        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_rmse = validate(model, val_loader, device)

        print(f"Epoch {epoch}: Train Loss {train_loss:.6f} | Val RMSE {val_rmse:.6f}")

        # Save Checkpoint
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            save_checkpoint(
                model, optimizer, epoch, train_loss, filename="best_model.pth"
            )

    # 5. Final Validation Metric
    # Reload best model to ensure we evaluate the optimal state
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print("Loaded best model for analysis.")

    # Re-calculate to print the exact required format
    final_rmse = validate(model, val_loader, device)
    print(f"Final Validation Metric: {final_rmse}")

    # 6. Failure Analysis
    print("\n--- Failure Analysis ---")
    model.eval()

    sampled_errors = []
    sampled_inputs = []

    # We analyze the validation set.
    # To keep it fast, we sample 10% of pixels from each validation image.
    with torch.no_grad():
        for noisy, clean, _ in val_loader:
            noisy = noisy.to(device)
            clean = clean.to(device)

            # Inference
            # noisy shape: (1, C, H, W) -> predict_tiled expects (C, H, W)
            pred = predict_tiled(model, noisy[0])

            # Calculate Absolute Error
            # clean shape: (1, C, H, W)
            target = clean[0]
            error_map = torch.abs(pred - target).cpu().numpy().flatten()
            input_map = noisy[0].cpu().numpy().flatten()

            # Random subsample to manage memory/time
            n_pixels = len(error_map)
            idx = np.random.choice(n_pixels, size=max(1, n_pixels // 10), replace=False)

            sampled_errors.append(error_map[idx])
            sampled_inputs.append(input_map[idx])

    # Concatenate all samples
    if len(sampled_errors) > 0:
        all_errors = np.concatenate(sampled_errors)
        all_inputs = np.concatenate(sampled_inputs)

        # Calculate Pearson Correlation
        corr, _ = pearsonr(all_inputs, all_errors)
        print(f"Correlation between Input Intensity and Error Magnitude: {corr:.8f}")
    else:
        print("Insufficient data for failure analysis.")

    # 7. Submission Logic
    THRESHOLD = 0.0076658159
    if final_rmse < THRESHOLD:
        print(f"\nMetric {final_rmse} < {THRESHOLD}. Generating submission...")
        # generate_submission uses the library machinery, which we've patched to be faster
        library.inference.generate_submission(checkpoint_name="best_model.pth")
    else:
        print(f"\nMetric {final_rmse} >= {THRESHOLD}. Submission skipped.")


if __name__ == "__main__":
    run()
