import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Import from provided library files
from library.config import Config
from library.dataset import StegoDataset, TestDataset, get_transforms
from library.model import ResV2GeM
from library.engine import train_one_epoch, validate, predict_tta
from library.utils import seed_everything, weighted_auc


def main():
    # 1. Configuration & Setup
    # Removed overrides to allow full training schedule (Cite solution_lesson_node_00005)

    seed_everything(Config.seed)
    device = Config.device
    print(f"Running on device: {device}")

    # 2. Data Loading
    print("Initializing DataLoaders...")

    # Training Set (Uses Unique Content Sampling via Grouping)
    train_dataset = StegoDataset(
        csv_path=Config.train_csv,
        root_dir=Config.input_dir,
        mode="train",
        transform=get_transforms("train"),
        load_cached_data=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Validation Set
    val_dataset = StegoDataset(
        csv_path=Config.val_csv,
        root_dir=Config.input_dir,
        mode="val",
        transform=get_transforms("val"),
        load_cached_data=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.batch_size * 2,  # Can handle larger batch size in eval
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    print(f"Train samples (unique content): {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # 3. Model Initialization
    print("Initializing Model...")
    model = ResV2GeM(config=Config, pretrained=Config.pretrained)
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.epochs, eta_min=Config.min_lr
    )

    # 4. Training Loop
    best_score = -float("inf")
    best_model_path = os.path.join(Config.checkpoint_dir, "best_model.pth")

    print("Starting Training...")
    start_train_time = time.time()

    for epoch in range(1, Config.epochs + 1):
        epoch_start = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device)

        # Validate
        val_score, val_loss = validate(model, val_loader, device)

        # Step Scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        epoch_time = time.time() - epoch_start
        print(
            f"Epoch {epoch}/{Config.epochs} | Time: {epoch_time:.1f}s | "
            f"LR: {current_lr:.2e} | Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Weighted AUC: {val_score:.6f}"
        )

        # Checkpoint
        if val_score > best_score:
            best_score = val_score
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> New best model saved (Score: {best_score:.6f})")

    total_train_time = time.time() - start_train_time
    print(f"Training finished in {total_train_time:.1f}s")

    # Required Output
    print(f"Final Validation Metric: {best_score}")

    # 5. Failure Analysis
    print("\n--- Failure Analysis ---")

    # Load best model
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print("Loaded best model for analysis.")
    else:
        print("Warning: Best model file not found. Using current model weights.")

    model.eval()

    # Collect predictions and targets manually to map to metadata
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            all_preds.extend(probs)
            all_targets.extend(targets.numpy().flatten())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Calculate absolute error
    errors = np.abs(all_targets - all_preds)

    # Get metadata for correlation analysis
    val_df = val_dataset.data

    # Ensure lengths match
    if len(val_df) != len(errors):
        print(
            f"Warning: Mismatch in validation set size ({len(val_df)}) and predictions ({len(errors)}). Skipping correlation."
        )
    else:
        # Extract file sizes as a proxy for image complexity
        print("Calculating file sizes for correlation...")
        file_sizes = []
        for rel_path in val_df["file_path"]:
            full_path = os.path.join(Config.input_dir, rel_path)
            try:
                file_sizes.append(os.path.getsize(full_path))
            except OSError:
                file_sizes.append(0)

        file_sizes = np.array(file_sizes)

        # Calculate correlation
        if np.std(file_sizes) > 0 and np.std(errors) > 0:
            correlation = np.corrcoef(errors, file_sizes)[0, 1]
            print(f"Correlation between Error and File Size: {correlation:.6f}")
        else:
            print("Correlation could not be computed (zero variance).")

    # 6. Submission
    threshold = 0.8416
    if best_score > threshold:
        print(
            f"\nValidation score ({best_score:.6f}) exceeds threshold ({threshold}). Generating submission..."
        )

        test_dataset = TestDataset(
            csv_path=Config.test_csv,
            root_dir=Config.input_dir,
            transform=get_transforms("val"),  # No augs for base, TTA handles it
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.batch_size,
            shuffle=False,
            num_workers=Config.num_workers,
            pin_memory=True,
        )

        # predict_tta handles the 5-view TTA and returns a DataFrame
        submission_df = predict_tta(model, test_loader, device)

        submission_df.to_csv(Config.submission_path, index=False)
        print(f"Submission saved to {Config.submission_path}")

    else:
        print(
            f"\nValidation score ({best_score:.6f}) did not meet threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    main()
