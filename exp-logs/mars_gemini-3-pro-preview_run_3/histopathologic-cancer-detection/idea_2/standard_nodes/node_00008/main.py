import os
import sys
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

# Import provided library components
from library.config import Config
from library.utils import seed_everything
from library.dataset import load_metadata, get_transforms, PathologyDataset
from library.models import get_model
from library.engine import train_one_epoch, validate_with_tta, predict_with_tta
from library.inference import predict_ensemble


def analyze_failures(df_val, preds, targets):
    """
    Performs failure analysis by correlating prediction errors with image meta-features.
    """
    print("\n--- Failure Analysis ---")

    # Calculate absolute error
    errors = np.abs(targets - preds)

    # We will compute stats for a subset to save time if dataset is huge,
    # but 35k is manageable on this hardware. We'll do all for accuracy.
    # To be safe on time, we'll use a max of 5000 random samples for the correlation analysis.
    n_samples = min(len(df_val), 5000)
    indices = np.random.choice(len(df_val), n_samples, replace=False)

    subset_df = df_val.iloc[indices].copy()
    subset_errors = errors[indices]

    # Feature accumulators
    features = {
        "brightness": [],
        "contrast": [],
        "red_mean": [],
        "green_mean": [],
        "blue_mean": [],
        "sharpness": [],
    }

    print(f"Extracting features from {n_samples} validation images...")

    for idx, row in subset_df.iterrows():
        img_path = os.path.join(Config.INPUT_DIR, row["file_path"])
        img = cv2.imread(img_path)

        if img is None:
            # Fill with defaults if read fails
            for k in features:
                features[k].append(0.0)
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Meta-feature extraction
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        brightness = np.mean(gray)
        contrast = np.std(gray)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = laplacian.var()

        features["brightness"].append(brightness)
        features["contrast"].append(contrast)
        features["red_mean"].append(np.mean(img[:, :, 0]))
        features["green_mean"].append(np.mean(img[:, :, 1]))
        features["blue_mean"].append(np.mean(img[:, :, 2]))
        features["sharpness"].append(sharpness)

    # Compute correlations
    print("Correlation between Error Magnitude and Image Features:")
    for name, values in features.items():
        if len(values) != len(subset_errors):
            continue
        # Pearson correlation
        corr = np.corrcoef(subset_errors, values)[0, 1]
        print(f"  Error vs {name}: {corr:.4f}")


def main():
    # 1. Setup
    seed_everything(Config.SEED)

    # Override Config for Fast Baseline
    # We use 2 epochs to ensure convergence while staying within time limits
    Config.NUM_EPOCHS = 2

    device = Config.DEVICE
    print(f"Running on device: {device}")

    # 2. Data Loading
    print("Loading metadata...")
    df_train = load_metadata("train")
    df_val = load_metadata("val")

    # Datasets
    train_dataset = PathologyDataset(
        df_train, phase="train", transform=get_transforms("train")
    )
    val_dataset = PathologyDataset(df_val, phase="val", transform=get_transforms("val"))

    # Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(device == "cuda"),
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(device == "cuda"),
    )

    # 3. Training Loop (Heterogeneous Ensemble)
    model_weights_paths = []

    for model_name in Config.MODELS:
        print(f"\n=== Training Model: {model_name} ===")

        # Initialize Model
        model = get_model(model_name, pretrained=True)
        model.to(device)

        # Optimizer & Scheduler
        optimizer = optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=Config.NUM_EPOCHS
        )

        best_auc = 0.0
        best_weights_path = os.path.join(Config.WORKING_DIR, f"{model_name}_best.pth")

        for epoch in range(1, Config.NUM_EPOCHS + 1):
            # Train
            train_metrics = train_one_epoch(
                model, train_loader, optimizer, scheduler, device, epoch
            )

            # Step scheduler
            scheduler.step()

            # Validate (with TTA)
            val_metrics = validate_with_tta(model, val_loader, device)

            # Checkpoint
            if val_metrics["AUC"] > best_auc:
                best_auc = val_metrics["AUC"]
                torch.save(model.state_dict(), best_weights_path)
                print(f"  New Best AUC: {best_auc:.5f} -> Saved.")

        print(f"Finished training {model_name}. Best Val AUC: {best_auc:.5f}")
        model_weights_paths.append(best_weights_path)

        # Cleanup
        del model, optimizer, scheduler
        torch.cuda.empty_cache()

    # 4. Ensemble Validation
    print("\n=== Ensemble Validation ===")
    ensemble_preds = np.zeros(len(df_val))
    val_targets = df_val["label"].values

    for model_name in Config.MODELS:
        print(f"Generating predictions for {model_name}...")
        model = get_model(model_name, pretrained=False)
        weights_path = os.path.join(Config.WORKING_DIR, f"{model_name}_best.pth")
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.to(device)

        preds = predict_with_tta(model, val_loader, device)
        ensemble_preds += preds.flatten()

        del model
        torch.cuda.empty_cache()

    # Average predictions
    avg_preds = ensemble_preds / len(Config.MODELS)

    # Compute Metric
    final_val_auc = roc_auc_score(val_targets, avg_preds)
    print(f"Final Validation Metric: {final_val_auc}")

    # 5. Failure Analysis
    analyze_failures(df_val, avg_preds, val_targets)

    # 6. Submission
    THRESHOLD = 0.9892525043540494
    if final_val_auc > THRESHOLD:
        print(
            f"\nValidation metric {final_val_auc} > {THRESHOLD}. Generating submission..."
        )
        predict_ensemble()
    else:
        print(
            f"\nValidation metric {final_val_auc} <= {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
