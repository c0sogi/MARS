import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import warnings

# Import library components
from library.config import CFG
from library.utils import seed_everything
from library.dataset import CassavaDataset
from library.model import CassavaModel
from library.engine import train_one_epoch, valid_one_epoch
from library.augmentations import get_transforms

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    # 1. Setup Environment
    seed_everything(CFG.seed)
    device = CFG.device

    # Override batch sizes for A100 efficiency
    # EfficientNet-B4 (380x380) requires smaller batches to fit in 40GB VRAM
    CFG.train_batch_size = 32
    CFG.valid_batch_size = 64

    # Ensure submission directory exists
    os.makedirs("./submission", exist_ok=True)

    # 2. Data Loading
    train_df = pd.read_csv(CFG.train_csv)
    val_df = pd.read_csv(CFG.val_csv)

    # Debug mode limit (optional, based on CFG)
    if CFG.debug:
        train_df = train_df.iloc[: CFG.debug_sample_size]
        val_df = val_df.iloc[: CFG.debug_sample_size]

    train_dataset = CassavaDataset(
        train_df, transform=get_transforms("train"), output_label=True
    )

    val_dataset = CassavaDataset(
        val_df, transform=get_transforms("valid"), output_label=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.train_batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=CFG.valid_batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
    )

    # 3. Model Initialization
    model = CassavaModel(pretrained=True)
    model.to(device)

    # 4. Training Setup
    optimizer = AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)

    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=CFG.T_0, T_mult=CFG.T_mult, eta_min=CFG.min_lr
    )

    # Loss function with Label Smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=CFG.label_smoothing)

    # 5. Training Loop
    best_acc = 0.0
    best_model_path = os.path.join(CFG.output_dir, "best_model.pth")

    for epoch in range(CFG.epochs):
        # Train
        train_loss = train_one_epoch(
            epoch, model, train_loader, optimizer, device, criterion
        )

        # Validate
        val_loss, val_acc = valid_one_epoch(epoch, model, val_loader, device, criterion)

        # Update Scheduler
        scheduler.step()

        # Save Best Model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_model_path)

    # 6. Final Evaluation & Failure Analysis
    print(f"Final Validation Metric: {best_acc}")

    # Reload best model for analysis and inference
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    val_errors = []
    val_means = []
    val_stds = []

    # Iterate validation set to collect stats and errors for failure analysis
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            # Error: 1 if wrong, 0 if correct
            batch_errors = (preds != labels).cpu().numpy().astype(int)
            val_errors.extend(batch_errors)

            # Calculate stats per image in batch (on normalized tensors)
            # Flatten spatial dims: (B, C, H, W) -> (B, -1)
            flat_images = images.view(images.size(0), -1)
            batch_means = flat_images.mean(dim=1).cpu().numpy()
            batch_stds = flat_images.std(dim=1).cpu().numpy()

            val_means.extend(batch_means)
            val_stds.extend(batch_stds)

    # Calculate correlations
    if len(val_errors) > 0:
        # Check for variance to avoid NaNs
        if np.std(val_errors) > 1e-9 and np.std(val_means) > 1e-9:
            corr_mean = np.corrcoef(val_errors, val_means)[0, 1]
        else:
            corr_mean = 0.0

        if np.std(val_errors) > 1e-9 and np.std(val_stds) > 1e-9:
            corr_std = np.corrcoef(val_errors, val_stds)[0, 1]
        else:
            corr_std = 0.0

        print(f"Failure Analysis - Correlation with Input Mean: {corr_mean}")
        print(f"Failure Analysis - Correlation with Input Std: {corr_std}")

    # 7. Submission
    THRESHOLD = 0.8461949265687584

    if best_acc > THRESHOLD:
        test_df = pd.read_csv(CFG.test_csv)
        test_dataset = CassavaDataset(
            test_df, transform=get_transforms("inference"), output_label=False
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=CFG.valid_batch_size,
            shuffle=False,
            num_workers=CFG.num_workers,
            pin_memory=True,
        )

        final_preds = []

        # Test Time Augmentation (TTA)
        # Strategy: Average logits of Original, Horizontal Flip, and Vertical Flip
        with torch.no_grad():
            for images in test_loader:
                images = images.to(device)

                # 1. Original
                logits = model(images)

                # 2. Horizontal Flip (dim 3 is width)
                logits += model(torch.flip(images, dims=[3]))

                # 3. Vertical Flip (dim 2 is height)
                logits += model(torch.flip(images, dims=[2]))

                # Average
                logits /= 3.0

                preds = torch.argmax(logits, dim=1).cpu().numpy()
                final_preds.extend(preds)

        # Create submission file
        submission_df = pd.DataFrame(
            {"image_id": test_df["image_id"], "label": final_preds}
        )

        submission_path = "./submission/submission.csv"
        submission_df.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")

    else:
        print(
            f"Validation metric {best_acc} is lower than threshold {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
