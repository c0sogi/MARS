import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from scipy.stats import pearsonr

# Import from library
from library.config import Config
from library.utils import seed_everything, save_checkpoint, calculate_auc
from library.dataset import create_datasets, PathologyDataset, get_transforms
from library.model import PathologyModel
from library.engine import train_epoch, valid_epoch, inference_fn


def main():
    # 1. Setup
    Config.setup()
    seed_everything(Config.SEED)

    # Removed Epoch override to allow full training (Cite Lesson 00014)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 2. Data Loading
    print("Loading datasets...")
    # Load raw datasets directly
    train_ds_raw, val_ds_raw, test_ds = create_datasets(load_cached_data=True)

    print(f"Training samples: {len(train_ds_raw)}")
    print(f"Validation samples: {len(val_ds_raw)}")

    # 3. Single Fold Training (Cite Lesson 00014)
    print(f"\n{'='*20} Training Start {'='*20}")

    # Create Loaders
    train_loader = DataLoader(
        train_ds_raw,  # Already has transforms applied in dataset creation? No, dataset object needs to be created
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds_raw,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Initialize Model
    model = PathologyModel(config=Config)
    model.to(device)

    # Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.MIN_LR
    )

    # Training Loop
    best_auc = 0.0

    for epoch in range(Config.EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch)
        val_loss, val_auc = valid_epoch(model, val_loader, device)
        scheduler.step()

        is_best = val_auc > best_auc
        if is_best:
            best_auc = val_auc

        save_checkpoint(
            {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "best_auc": best_auc,
                "optimizer": optimizer.state_dict(),
            },
            is_best,
            0,  # Fold 0
        )

    print(f"Best Validation AUC: {best_auc:.6f}")

    # Load best model for Final Validation & Analysis
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model_fold_0.pth")
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    # Generate predictions for validation set
    val_preds = []
    val_targets = []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            val_preds.extend(probs.cpu().numpy().flatten())
            val_targets.extend(labels.numpy().flatten())

    val_preds = np.array(val_preds)
    val_targets = np.array(val_targets)

    # 4. Global Validation Metric
    final_auc = calculate_auc(val_targets, val_preds)
    print(f"\nFinal Validation Metric: {final_auc}")

    # 5. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Calculate error magnitude
    errors = np.abs(val_targets - val_preds)

    # Compute image stats for the validation dataset
    print("Computing image statistics for correlation analysis...")

    # Use raw validation images for stats
    imgs_float = val_ds_raw.images.astype(np.float32) / 255.0

    # Mean across H,W (spatial)
    spatial_means = np.mean(imgs_float, axis=(1, 2))  # Shape (N, 3)
    # Std across H,W
    spatial_stds = np.std(imgs_float, axis=(1, 2))  # Shape (N, 3)

    # Brightness: mean of channels
    brightness = np.mean(spatial_means, axis=1)

    # Contrast: mean of channel stds (approx)
    contrast = np.mean(spatial_stds, axis=1)

    # Red mean (Channel 0)
    red_mean = spatial_means[:, 0]

    # Calculate Correlations
    corr_brightness, _ = pearsonr(errors, brightness)
    corr_contrast, _ = pearsonr(errors, contrast)
    corr_red, _ = pearsonr(errors, red_mean)

    print(f"Correlation (Error vs Brightness): {corr_brightness:.4f}")
    print(f"Correlation (Error vs Contrast):   {corr_contrast:.4f}")
    print(f"Correlation (Error vs Red Mean):   {corr_red:.4f}")

    # 6. Submission Generation
    THRESHOLD = 0.9889066475479729

    if final_auc > THRESHOLD:
        print(
            f"\nValidation Metric ({final_auc}) > Threshold ({THRESHOLD}). Generating submission..."
        )

        test_loader = DataLoader(
            test_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        print(f"Inference with best model...")
        model = PathologyModel(config=Config)
        model.to(device)

        # Load checkpoint
        ckpt_path = os.path.join(Config.CHECKPOINT_DIR, "best_model_fold_0.pth")
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])

        # Inference with TTA
        preds = inference_fn(model, test_loader, device)

        # Create Submission DataFrame
        df_test = pd.read_csv(Config.TEST_META_PATH)
        submission = pd.DataFrame({"id": df_test["id"], "label": preds})

        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"\nValidation Metric ({final_auc}) <= Threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
