import os
import sys
import torch
import pandas as pd
import numpy as np
import cv2
from torch.utils.data import DataLoader
from scipy.stats import pearsonr

# Import from library
from library.config import Config
from library.dataset import CatheterDataset
from library.model import CatheterModel
from library.engine import train_model, evaluate, generate_submission
from library.utils import seed_everything, get_score


def main():
    # --- 1. Setup ---
    print("Initializing...")
    Config.setup()
    seed_everything(Config.SEED)

    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # --- 2. Data Loading ---
    print("Loading datasets...")
    # Train Dataset
    train_dataset = CatheterDataset(
        metadata_path=Config.TRAIN_METADATA_PATH, mode="train"
    )

    # Validation Dataset
    val_dataset = CatheterDataset(metadata_path=Config.VAL_METADATA_PATH, mode="val")

    # DataLoaders
    # Using the batch size and workers from Config
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

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # --- 3. Model Initialization ---
    print("Initializing model...")
    model = CatheterModel(pretrained=Config.PRETRAINED)
    model.to(device)

    # --- 4. Optimizer and Scheduler ---
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Calculate total steps for OneCycleLR
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * Config.EPOCHS

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        total_steps=total_steps,
        pct_start=Config.PCT_START,
        div_factor=Config.DIV_FACTOR,
        final_div_factor=Config.FINAL_DIV_FACTOR,
    )

    # --- 5. Training ---
    print("Starting training...")
    # train_model handles the loop, validation monitoring, and saving best model
    best_auc = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=Config.EPOCHS,
    )

    # --- 6. Final Validation & Failure Analysis ---
    print("\n=== Final Validation & Failure Analysis ===")

    # Load best model state
    print(f"Loading best model from {Config.MODEL_SAVE_PATH}")
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    model.to(device)
    model.eval()

    # Collect validation predictions for detailed analysis
    print("Collecting validation predictions for analysis...")
    val_preds = []
    val_targets = []

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device, dtype=torch.float)
            targets = targets.to(device, dtype=torch.float)

            outputs = model(images)
            probs = torch.sigmoid(outputs)

            val_preds.append(probs.cpu().numpy())
            val_targets.append(targets.cpu().numpy())

    val_preds = np.concatenate(val_preds, axis=0)
    val_targets = np.concatenate(val_targets, axis=0)

    # Calculate Final Metric
    print("Calculating Final Validation Metric...")
    final_metric = get_score(val_targets, val_preds)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis: Correlation between Error Magnitude and Input Features
    print("\n--- Failure Analysis ---")

    # Calculate Mean Absolute Error per sample (averaged across all classes)
    errors = np.abs(val_targets - val_preds).mean(axis=1)

    # Extract original image features (Width, Height, Intensity)
    # We iterate through the metadata which matches the order of the DataLoader (shuffle=False)
    df_val = pd.read_csv(Config.VAL_METADATA_PATH)

    print("Extracting image features for failure analysis...")
    widths = []
    heights = []
    aspect_ratios = []
    mean_intensities = []

    # Process images to get original stats
    for idx, row in df_val.iterrows():
        img_path = os.path.join(Config.INPUT_DIR, row["file_path"])
        try:
            # Read image to get original dimensions and intensity
            img = cv2.imread(img_path)
            if img is None:
                w, h, i = 0, 0, 0
            else:
                h, w, c = img.shape
                # Simple mean intensity (normalized)
                i = img.mean() / 255.0

            widths.append(w)
            heights.append(h)
            aspect_ratios.append(w / h if h > 0 else 0)
            mean_intensities.append(i)
        except Exception:
            widths.append(0)
            heights.append(0)
            aspect_ratios.append(0)
            mean_intensities.append(0)

    # Compute Correlations
    features = {
        "Width": widths,
        "Height": heights,
        "Aspect Ratio": aspect_ratios,
        "Mean Intensity": mean_intensities,
    }

    print("\nCorrelation between Error Magnitude and Input Features:")
    for name, values in features.items():
        if len(values) != len(errors):
            print(f"Shape mismatch for {name}: {len(values)} vs {len(errors)}")
            continue

        # Pearson correlation
        corr, _ = pearsonr(values, errors)
        print(f"{name}: {corr:.4f}")

    # --- 7. Submission ---
    # Threshold defined in the task
    THRESHOLD = 0.9398508707740129

    if final_metric > THRESHOLD:
        print(
            f"\nMetric ({final_metric}) > Threshold ({THRESHOLD}). Generating submission..."
        )

        # Test Dataset & Loader
        test_dataset = CatheterDataset(
            metadata_path=Config.TEST_METADATA_PATH, mode="test"
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        generate_submission(model, test_loader, device)
    else:
        print(
            f"\nMetric ({final_metric}) <= Threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
