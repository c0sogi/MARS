import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from torch.utils.data import DataLoader

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, get_device, calculate_rmse
from library.dataset import DenoisingDataset
from library.model import AG_CAC_ResUNet
from library.engine import train_model, validate
from library.inference import generate_submission_inference


def run_failure_analysis(model, val_loader, device):
    """
    Performs failure analysis on the validation set.
    Calculates per-image RMSE and correlates it with input image statistics.
    """
    print("\n--- Failure Analysis ---")
    model.eval()

    errors = []
    means = []
    stds = []

    with torch.no_grad():
        for inputs, targets, _ in val_loader:
            inputs = inputs.to(device)
            # Targets are kept on CPU/converted inside calculate_rmse if needed,
            # but calculate_rmse handles tensors. For consistency, move to device or keep logic simple.
            # calculate_rmse expects (y_true, y_pred).

            # Inference
            outputs = model(inputs)
            outputs = torch.clamp(outputs, 0, 1)

            # Calculate RMSE for this single image (Batch size 1 expected for Val)
            # Move to CPU for metric calculation to ensure consistency with utils
            rmse = calculate_rmse(targets, outputs)
            errors.append(rmse)

            # Calculate Input Features (Noisy Image)
            # inputs is (B, C, H, W). B=1.
            inp_np = inputs.cpu().numpy()
            means.append(np.mean(inp_np))
            stds.append(np.std(inp_np))

    # Calculate Correlations
    if len(errors) > 1:
        corr_mean, _ = pearsonr(errors, means)
        corr_std, _ = pearsonr(errors, stds)

        print(f"Correlation (Error vs Input Mean Intensity): {corr_mean:.4f}")
        print(f"Correlation (Error vs Input Std Dev):        {corr_std:.4f}")
    else:
        print("Not enough validation samples for correlation analysis.")


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = get_device()
    print(f"Using device: {device}")

    # 2. Data Loading
    # Train: Uses patching and augmentation
    train_dataset = DenoisingDataset(
        metadata_path=Config.TRAIN_METADATA_PATH, mode="train", load_cached_data=True
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Val: Uses full images
    val_dataset = DenoisingDataset(
        metadata_path=Config.VAL_METADATA_PATH, mode="val", load_cached_data=True
    )
    # Batch size 1 for validation to handle potentially varying image sizes (though dataset is uniform-ish)
    # and to simplify per-image failure analysis.
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Test: Uses full images
    test_dataset = DenoisingDataset(
        metadata_path=Config.TEST_METADATA_PATH, mode="test", load_cached_data=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val samples: {len(val_dataset)}")

    # 3. Model Initialization
    model = AG_CAC_ResUNet().to(device)

    # 4. Optimization
    # Using AdamW with weight decay as specified in Idea
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Cosine Annealing Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.NUM_EPOCHS, eta_min=1e-6
    )

    # 5. Training
    # We limit epochs to ensure runtime compliance, though Config.NUM_EPOCHS is 100.
    # Given the A100 and dataset size, 100 epochs is feasible within 2 hours.
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=Config.NUM_EPOCHS,
        patience=Config.EARLY_STOPPING_PATIENCE,
    )

    # 6. Evaluation
    # Load best model
    print(f"Loading best model from {Config.MODEL_SAVE_PATH}")
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))

    # Calculate Final Validation Metric
    final_val_rmse = validate(model, val_loader, device)
    print(f"Final Validation Metric: {final_val_rmse:.10f}")

    # 7. Failure Analysis
    run_failure_analysis(model, val_loader, device)

    # 8. Submission
    # Threshold from instructions
    THRESHOLD = 0.0076658159

    if final_val_rmse < THRESHOLD:
        print(
            f"Validation metric ({final_val_rmse:.10f}) meets threshold ({THRESHOLD}). Generating submission..."
        )
        generate_submission_inference(
            model=model,
            dataloader=test_loader,
            device=device,
            output_path=Config.SUBMISSION_FILE_PATH,
        )
    else:
        print(
            f"Validation metric ({final_val_rmse:.10f}) does not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
