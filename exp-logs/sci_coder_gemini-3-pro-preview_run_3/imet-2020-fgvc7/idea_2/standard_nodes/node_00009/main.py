import os
import time
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler

# Import library modules
from library.config import Config
from library.dataset import ArtworkDataset, get_transforms, load_and_process_metadata
from library.model import ArtworkModel
from library.utils import set_seed, calculate_f1, optimize_threshold
from library.train import train_one_epoch, validate
from library.inference import predict


def run_full_training():
    # 1. Setup
    set_seed(Config.SEED)
    device = Config.DEVICE
    print(f"Running Full Training on device: {device}")

    # 2. Data Loading
    print("Loading metadata...")
    # Load raw dataframes
    train_df = load_and_process_metadata(
        Config.TRAIN_CSV, "cached_train.parquet", load_cached_data=True
    )
    val_df = load_and_process_metadata(
        Config.VAL_CSV, "cached_val.parquet", load_cached_data=True
    )

    # Create Datasets
    train_dataset = ArtworkDataset(
        train_df, transforms=get_transforms("train"), mode="train"
    )
    val_dataset = ArtworkDataset(val_df, transforms=get_transforms("val"), mode="val")

    # Create DataLoaders
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
        drop_last=False,
    )

    # 3. Model Initialization
    print(f"Initializing model: {Config.MODEL_NAME}")
    model = ArtworkModel(pretrained=True)
    model.to(device)

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        epochs=Config.EPOCHS,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1,
        anneal_strategy="cos",
        div_factor=25.0,
        final_div_factor=10000.0,
    )

    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler()

    # 4. Training Loop
    best_f1 = -1.0
    best_epoch = 0

    print(f"Starting training for {Config.EPOCHS} epochs...")
    for epoch in range(1, Config.EPOCHS + 1):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler, device, epoch
        )

        # Validate
        val_loss, val_f1, _, _ = validate(model, val_loader, criterion, device)

        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch}/{Config.EPOCHS} | Time: {elapsed:.1f}s | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}"
        )

        # Save Best
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            torch.save(model.state_dict(), Config.MODEL_PATH)

    print(f"Training finished. Best F1: {best_f1:.4f} at epoch {best_epoch}")

    # 5. Final Evaluation & Threshold Optimization
    print("Loading best model for final evaluation...")
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))

    # Get predictions on full validation set
    _, final_val_f1, val_probs, val_targets = validate(
        model, val_loader, criterion, device
    )

    # Optimize Threshold
    best_threshold, optimized_f1 = optimize_threshold(val_targets, val_probs)

    # REQUIRED OUTPUT: Final Validation Metric
    # Using the optimized F1 score as the final metric
    print(f"Final Validation Metric: {optimized_f1}")

    # 6. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Calculate Error Magnitude (Mean Absolute Error per sample)
    # val_probs: (N, C), val_targets: (N, C)
    error_magnitude = np.mean(np.abs(val_probs - val_targets), axis=1)

    # Calculate Label Cardinality (Number of active labels per sample)
    label_cardinality = np.sum(val_targets, axis=1)

    # Calculate Correlation
    if len(error_magnitude) > 1:
        correlation = np.corrcoef(error_magnitude, label_cardinality)[0, 1]
        print(
            f"Correlation between Error Magnitude and Label Cardinality: {correlation:.4f}"
        )

        # Additional insight
        print(f"Mean Error: {np.mean(error_magnitude):.4f}")
        print(f"Mean Cardinality: {np.mean(label_cardinality):.4f}")
    else:
        print("Not enough samples for correlation analysis.")

    # 7. Submission
    TARGET_METRIC = 0.5833256393684794

    if optimized_f1 > TARGET_METRIC:
        print(f"\nMetric {optimized_f1} > {TARGET_METRIC}. Generating submission...")

        # Load Test Data
        test_df = load_and_process_metadata(
            Config.TEST_CSV, "cached_test.parquet", load_cached_data=True
        )
        test_dataset = ArtworkDataset(
            test_df, transforms=get_transforms("test"), mode="test"
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
            drop_last=False,
        )

        # Predict
        ids, predictions = predict(model, test_loader, best_threshold, device)

        # Save
        submission_df = pd.DataFrame({"id": ids, "attribute_ids": predictions})
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(
            f"\nMetric {optimized_f1} <= {TARGET_METRIC}. Skipping submission generation."
        )


if __name__ == "__main__":
    run_full_training()
