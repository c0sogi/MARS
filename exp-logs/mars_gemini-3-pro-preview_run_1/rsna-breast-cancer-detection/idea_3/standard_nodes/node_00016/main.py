import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import library modules
from library import config, utils, data, model, train, predict


def main():
    # 1. Setup
    utils.seed_everything(config.SEED)
    device = config.DEVICE
    print(f"Device: {device}")

    # Fast Baseline Configuration
    NUM_EPOCHS = 5
    MAX_TRAIN_SAMPLES = 30000  # Limit samples for fast baseline

    # 2. Data Preparation
    print("Preparing data...")
    # Load metadata dataframes
    df_train, df_val, df_test = data.process_metadata(load_cached_data=True)

    # Calculate stats from full train set before subsampling (to maintain distribution stats)
    age_mean = df_train["age"].mean()
    age_std = df_train["age"].std()

    # Subsample training data for fast baseline
    if len(df_train) > MAX_TRAIN_SAMPLES:
        print(
            f"Subsampling training data from {len(df_train)} to {MAX_TRAIN_SAMPLES}..."
        )
        df_train = df_train.sample(
            n=MAX_TRAIN_SAMPLES, random_state=config.SEED
        ).reset_index(drop=True)

    # Create Datasets
    train_dataset = data.BreastCancerDataset(
        df_train,
        transforms=data.get_transforms("train"),
        age_mean=age_mean,
        age_std=age_std,
        mode="train",
    )

    val_dataset = data.BreastCancerDataset(
        df_val,
        transforms=data.get_transforms("val"),
        age_mean=age_mean,
        age_std=age_std,
        mode="val",
    )

    # Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Setup
    net = model.MetadataEfficientNet()
    net.to(device)

    # Loss, Optimizer, Scheduler
    pos_weight = torch.tensor([config.POS_WEIGHT]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.AdamW(
        net.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # 4. Training Loop
    print(f"Starting training for {NUM_EPOCHS} epochs...")
    best_pf1 = -1.0

    for epoch in range(NUM_EPOCHS):
        # Train
        train_loss = train.train_one_epoch(
            net, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_pf1 = train.validate(net, val_loader, criterion, device)

        # Step Scheduler
        scheduler.step()

        print(
            f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val pF1: {val_pf1:.6f}"
        )

        # Save Best
        if val_pf1 > best_pf1:
            best_pf1 = val_pf1
            torch.save(net.state_dict(), config.MODEL_SAVE_PATH)
            print(f"  New best model saved!")

    print(f"Final Validation Metric: {best_pf1}")

    # 5. Failure Analysis
    print("\nRunning Failure Analysis on Validation Set...")
    # Load best model
    net.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))
    net.eval()

    val_preds = []
    val_labels = []

    with torch.no_grad():
        for images, ages, implants, labels in val_loader:
            images = images.to(device)
            ages = ages.to(device)
            implants = implants.to(device)

            logits = net(images, ages, implants)
            probs = torch.sigmoid(logits)

            val_preds.append(probs.cpu().numpy())
            val_labels.append(labels.cpu().numpy())

    val_preds = np.concatenate(val_preds).flatten()
    val_labels = np.concatenate(val_labels).flatten()

    # Calculate Error
    errors = np.abs(val_preds - val_labels)

    # Create Analysis DataFrame
    df_analysis = df_val.copy()
    # Ensure alignment (loader is sequential)
    if len(df_analysis) == len(errors):
        df_analysis["error"] = errors

        # Encode categoricals for correlation
        if "view" in df_analysis.columns:
            df_analysis["view_code"] = df_analysis["view"].astype("category").cat.codes
        if "laterality" in df_analysis.columns:
            df_analysis["lat_code"] = (
                df_analysis["laterality"].astype("category").cat.codes
            )

        # Correlate
        correlations = {}
        features_to_check = ["age", "implant", "view_code", "lat_code", "machine_id"]

        print("Correlation between Error Magnitude and Features:")
        for feat in features_to_check:
            if feat in df_analysis.columns:
                corr = df_analysis["error"].corr(df_analysis[feat])
                correlations[feat] = corr
                print(f"  {feat}: {corr:.4f}")
    else:
        print("Warning: Mismatch in validation set length for analysis.")

    # 6. Submission
    THRESHOLD = 0.04437665641307831
    if best_pf1 > THRESHOLD:
        print(
            f"\nValidation score ({best_pf1}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )

        # Use predict module functions
        # Note: predict.inference_fn loads the model from config.MODEL_SAVE_PATH
        probs = predict.inference_fn(
            config.MODEL_SAVE_PATH, device, load_cached_data=True
        )
        predict.create_submission(probs, config.SUBMISSION_PATH, load_cached_data=True)
    else:
        print(
            f"\nValidation score ({best_pf1}) did not exceed threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
