import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from scipy.stats import pearsonr

# Import from library
from library.config import Config
from library.utils import seed_everything, save_checkpoint, calculate_auc
from library.dataset import create_datasets, PathologyDataset, get_transforms
from library.model import ConvNeXtGeM
from library.engine import train_epoch, valid_epoch, inference_fn


def main():
    # 1. Setup
    Config.setup()
    seed_everything(Config.SEED)

    # Use default Config.EPOCHS = 20 to ensure convergence (Cite solution_lesson_node_00014)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 2. Data Loading
    print("Loading datasets...")
    # Load raw datasets directly using the fixed split from metadata
    train_ds_raw, val_ds_raw, test_ds = create_datasets(load_cached_data=True)

    print(f"Training samples: {len(train_ds_raw)}")
    print(f"Validation samples: {len(val_ds_raw)}")

    # Create Datasets with appropriate transforms
    train_dataset = PathologyDataset(
        train_ds_raw.images, train_ds_raw.labels, transforms=get_transforms("train")
    )
    val_dataset = PathologyDataset(
        val_ds_raw.images, val_ds_raw.labels, transforms=get_transforms("val")
    )

    # Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Single Model Training Loop
    print(f"\n{'='*20} Training Start {'='*20}")

    # Initialize Model
    model = ConvNeXtGeM(config=Config)
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

    best_auc = 0.0
    fold = 0  # Single split treated as fold 0

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
            fold,
        )

    print(f"Best Validation AUC: {best_auc:.6f}")

    # 4. Global Validation Metric & Inference
    # Load best model
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, f"best_model_fold_{fold}.pth")
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

    final_auc = calculate_auc(val_targets, val_preds)
    print(f"\nFinal Validation Metric: {final_auc}")

    # 5. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Calculate error magnitude
    errors = np.abs(val_targets - val_preds)

    print("Computing image statistics for correlation analysis...")
    # Vectorized calculation for speed on validation set
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

        # Single model inference
        print(f"Inference with best model...")

        # Model is already loaded with best weights from above
        # Inference with TTA (Cite solution_lesson_node_00006)
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
