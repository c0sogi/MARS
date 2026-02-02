import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

# Import from provided library files
from library.config import Config, seed_everything
from library.dataset import (
    process_data,
    HotelDataset,
    get_transforms,
    BalanceClassSampler,
)
from library.model import HotelIdModel, train_one_epoch, validate, inference
from library.utils import apk


def run_pipeline():
    # 1. Setup and Configuration
    seed_everything(Config.SEED)

    # Adjust Config for a fast but effective baseline run
    Config.EPOCHS = 5
    Config.DEBUG = False  # Use full dataset

    print(f"Device: {Config.DEVICE}")
    print(f"Epochs: {Config.EPOCHS}")

    # 2. Data Loading
    print("Processing data...")
    train_df, val_df, test_df, num_classes = process_data(load_cached_data=True)

    # Pre-calculate class frequencies for failure analysis
    # (Count occurrences of each hotel_id in the training set)
    train_class_counts = train_df["hotel_id"].value_counts().to_dict()

    # Load Label Encoder classes for decoding
    encoder_path = os.path.join(Config.WORKING_DIR, "label_encoder.npy")
    label_encoder_classes = np.load(encoder_path, allow_pickle=True)

    # Initialize Datasets
    train_dataset = HotelDataset(
        train_df, transforms=get_transforms("train"), root_dir=Config.INPUT_DIR
    )
    val_dataset = HotelDataset(
        val_df, transforms=get_transforms("val"), root_dir=Config.INPUT_DIR
    )

    # Initialize Sampler and Loaders
    classes_per_batch = min(Config.CLASSES_PER_BATCH, num_classes)
    train_sampler = BalanceClassSampler(
        train_df["label"].values,
        classes_per_batch=classes_per_batch,
        samples_per_class=Config.SAMPLES_PER_CLASS,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Initialization
    model = HotelIdModel(num_classes=num_classes).to(Config.DEVICE)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.MIN_LR
    )

    criterion = nn.CrossEntropyLoss()

    # 4. Training Loop
    best_map = 0.0
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    print("Starting training...")
    for epoch in range(1, Config.EPOCHS + 1):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, Config.DEVICE, epoch
        )

        # Validate
        val_map = validate(model, val_loader, Config.DEVICE, num_classes)

        print(
            f"Epoch {epoch} | Train Loss: {train_loss:.6f} | Val MAP@5: {val_map:.6f}"
        )

        # Save Best Model
        if val_map > best_map:
            best_map = val_map
            torch.save(model.state_dict(), best_model_path)

    print(f"Final Validation Metric: {best_map}")

    # 5. Failure Analysis
    print("\nPerforming Failure Analysis on Validation Set...")

    # Reload best model
    model.load_state_dict(torch.load(best_model_path, map_location=Config.DEVICE))
    model.eval()

    val_errors = []
    val_freqs = []
    val_chains = []

    # We need to run inference manually on val set to get per-sample predictions
    # Note: We disable TTA here to match the validation metric calculation logic
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(Config.DEVICE)
            # Forward pass without labels (inference mode)
            outputs = model(images, labels=None)
            _, topk_indices = torch.topk(outputs, Config.TOP_K, dim=1)

            all_preds.extend(topk_indices.cpu().numpy().tolist())
            all_targets.extend(labels.numpy().tolist())

    # Calculate Error and correlate with features
    # Ensure alignment with val_df
    if len(all_preds) == len(val_df):
        for i, (pred, target) in enumerate(zip(all_preds, all_targets)):
            # Calculate AP@5 for this sample
            score = apk([target], pred, k=Config.TOP_K)
            error = 1.0 - score

            # Get metadata features
            hotel_id = val_df.iloc[i]["hotel_id"]
            chain_id = val_df.iloc[i]["chain"]

            # Get frequency from training set stats
            freq = train_class_counts.get(hotel_id, 0)

            val_errors.append(error)
            val_freqs.append(freq)
            val_chains.append(chain_id)

        # Compute Correlations
        if len(val_errors) > 0:
            corr_freq = np.corrcoef(val_errors, val_freqs)[0, 1]
            corr_chain = np.corrcoef(val_errors, val_chains)[0, 1]

            print(
                f"Correlation between Error Magnitude (1-AP@5) and Class Frequency: {corr_freq}"
            )
            print(
                f"Correlation between Error Magnitude (1-AP@5) and Chain ID: {corr_chain}"
            )
    else:
        print(
            "Error: Validation predictions length mismatch. Skipping detailed analysis."
        )

    # 6. Submission Generation
    THRESHOLD = 0.14571255006929015

    if best_map > THRESHOLD:
        print(f"\nValidation metric {best_map} > {THRESHOLD}. Generating submission...")

        # Prepare Test Loader
        test_dataset = HotelDataset(
            test_df,
            transforms=get_transforms("test"),
            root_dir=Config.INPUT_DIR,
            is_test=True,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Run Inference (uses TTA internally)
        predictions = inference(
            model, test_loader, Config.DEVICE, label_encoder_classes
        )

        # Create Submission File
        submission_df = test_df[["image"]].copy()
        submission_df["hotel_id"] = predictions

        submission_df.to_csv(Config.SUBMISSION_FILE, index=False)
        print(f"Submission saved to {Config.SUBMISSION_FILE}")

    else:
        print(f"\nValidation metric {best_map} <= {THRESHOLD}. Submission skipped.")


if __name__ == "__main__":
    run_pipeline()
