import sys
import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset
from scipy.stats import pearsonr

# Import from provided library files
from library.config import Config
from library.data import get_loaders, get_test_loader
from library.model import PathologyModel
from library.utils import seed_everything, ModelEma
from library.engine import train_one_epoch, validate, predict_tta, save_submission


def main():
    # --- 1. Runtime Configuration Overrides for Fast Baseline ---
    # We constrain the run to a single fold and single epoch on a subset of data
    # to ensure completion within the 8-minute limit.
    Config.EPOCHS = 1
    Config.NUM_FOLDS = 1
    Config.NUM_RUNS = 1
    Config.DEBUG = False  # Must be False to load the FULL validation set as required
    Config.BATCH_SIZE = 256

    # --- 2. Setup ---
    device = Config.DEVICE
    seed_everything(Config.SEED)
    print(f"Starting Fast Baseline Run on {device}...")

    # --- 3. Data Loading ---
    # Load full data into RAM (cached). We only use Fold 0.
    print("Loading data...")
    train_loader_full, val_loader = get_loaders(
        fold=0, seed=Config.SEED, load_cached_data=True
    )

    # Subsample Training Data: Use 2000 random samples for rapid training
    n_train_samples = 2000
    if len(train_loader_full.dataset) > n_train_samples:
        indices = torch.randperm(len(train_loader_full.dataset))[:n_train_samples]
        train_ds = Subset(train_loader_full.dataset, indices)
        print(f"Subsampled training set to {len(train_ds)} images.")
    else:
        train_ds = train_loader_full.dataset

    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        drop_last=True,
    )

    # --- 4. Model Initialization ---
    print("Initializing model...")
    model = PathologyModel().to(device)

    # Initialize EMA (Exponential Moving Average)
    model_ema = None
    if Config.USE_EMA:
        model_ema = ModelEma(model, decay=Config.EMA_DECAY, device=device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )

    # --- 5. Training ---
    print("Training for 1 epoch...")
    train_one_epoch(model, model_ema, train_loader, optimizer, device, epoch=1)

    # --- 6. Validation ---
    print("Validating on entire hold-out set...")
    # Use the EMA model for evaluation if available, otherwise the standard model
    eval_model = model_ema.module if model_ema else model

    val_loss, val_auc = validate(eval_model, val_loader, device)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {val_auc}")

    # --- 7. Failure Analysis ---
    print("\nPerforming Failure Analysis...")
    eval_model.eval()
    all_preds = []
    all_labels = []

    # Run inference on validation set to get element-wise predictions
    # (validate() only returns aggregated metrics)
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = eval_model(images)
            probs = torch.sigmoid(outputs)
            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())

    all_preds = np.concatenate(all_preds).flatten()
    all_labels = np.concatenate(all_labels).flatten()

    # Calculate Error Magnitude
    errors = np.abs(all_labels - all_preds)

    # Extract Features from Validation Images
    # val_loader.dataset is a MemoryPathologyDataset, so .images is the numpy array (N, H, W, C)
    val_images = val_loader.dataset.images

    # Vectorized feature computation (Normalize 0-1)
    imgs_norm = val_images.astype(np.float32) / 255.0

    # Brightness: Mean of all pixels
    brightness = imgs_norm.mean(axis=(1, 2, 3))
    # Contrast: Std of all pixels
    contrast = imgs_norm.std(axis=(1, 2, 3))
    # Channel Means
    red_mean = imgs_norm[..., 0].mean(axis=(1, 2))
    green_mean = imgs_norm[..., 1].mean(axis=(1, 2))
    blue_mean = imgs_norm[..., 2].mean(axis=(1, 2))

    features = {
        "Brightness": brightness,
        "Contrast": contrast,
        "Red Mean": red_mean,
        "Green Mean": green_mean,
        "Blue Mean": blue_mean,
    }

    print("Correlation between Error Magnitude and Input Features:")
    for name, feat_values in features.items():
        # Compute Pearson correlation
        if len(feat_values) == len(errors):
            corr, _ = pearsonr(errors, feat_values)
            print(f"  {name}: {corr:.4f}")
        else:
            print(f"  {name}: Size mismatch ({len(feat_values)} vs {len(errors)})")

    # --- 8. Conditional Submission ---
    threshold = 0.9889066475479729

    if val_auc > threshold:
        print(f"\nValidation metric {val_auc} > {threshold}. Generating submission...")

        # Load Test Data
        test_loader, test_ids = get_test_loader(load_cached_data=True)

        # Generate Predictions using TTA (8 views)
        preds = predict_tta(eval_model, test_loader, device)

        # Save Submission
        save_submission(test_ids, preds, Config.SUBMISSION_PATH)
    else:
        print(
            f"\nValidation metric {val_auc} <= {threshold}. Skipping submission generation."
        )


if __name__ == "__main__":
    main()
