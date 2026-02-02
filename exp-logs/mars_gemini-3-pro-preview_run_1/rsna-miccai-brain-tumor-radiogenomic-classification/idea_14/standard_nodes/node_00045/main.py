import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import pearsonr

# Import provided library modules
from library.config import Config
from library.utils import set_seed
from library.data import get_dataloader
from library.model import WITSNet
from library.train import train_one_epoch, validate
from library.inference import generate_submission


def main():
    # 1. Setup and Configuration
    set_seed(Config.SEED)

    # Override Config for Fast Baseline
    Config.NUM_EPOCHS = 10

    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # 2. Data Loading
    # Load metadata
    df_train = pd.read_csv(Config.TRAIN_METADATA_PATH)
    df_val = pd.read_csv(Config.VAL_METADATA_PATH)

    # Create DataLoaders
    # We use the full dataset as it is small (~500 subjects) and fits within the time limit.
    train_loader = get_dataloader(
        df_train,
        mode="train",
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
    )

    val_loader = get_dataloader(
        df_val, mode="val", batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS
    )

    # 3. Model Initialization
    model = WITSNet().to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    criterion = nn.BCEWithLogitsLoss()

    # 4. Training Loop
    best_auc = 0.0
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    print("Starting training...")
    for epoch in range(Config.NUM_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        # Save best model
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), best_model_path)

    # 5. Final Validation Metric
    # Requirement: Print full precision
    print(f"Final Validation Metric: {best_auc}")

    # 6. Failure Analysis
    print("Performing Failure Analysis...")

    # Reload best model
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # Generate predictions on validation set manually to get subject-level data
    all_probs = []
    all_targets = []
    all_ids = []

    with torch.no_grad():
        for images, targets, ids in val_loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            all_probs.extend(probs)
            all_targets.extend(targets.numpy().flatten())
            all_ids.extend(ids.numpy().flatten())

    # Aggregate slab predictions to subject level
    df_results = pd.DataFrame(
        {"BraTS21ID": all_ids, "prob": all_probs, "target": all_targets}
    )

    df_agg = (
        df_results.groupby("BraTS21ID")
        .agg({"prob": "mean", "target": "mean"})
        .reset_index()
    )

    # Calculate Error Magnitude
    df_agg["error"] = (df_agg["target"] - df_agg["prob"]).abs()

    # Extract Input Feature: File Count (Slice Depth) for FLAIR
    # This serves as a proxy for "amount of information" available for the subject
    file_counts = []
    for _, row in df_agg.iterrows():
        sid = row["BraTS21ID"]
        # Find path in metadata
        meta_row = df_val[df_val["BraTS21ID"] == sid]
        if not meta_row.empty:
            rel_path = meta_row.iloc[0]["flair_path"]
            full_path = os.path.join(Config.INPUT_DIR, rel_path)
            try:
                # Count dicom files
                count = len([f for f in os.listdir(full_path) if f.endswith(".dcm")])
            except (FileNotFoundError, OSError):
                count = 0
            file_counts.append(count)
        else:
            file_counts.append(0)

    df_agg["flair_count"] = file_counts

    # Calculate Correlation
    if len(df_agg) > 1:
        corr, _ = pearsonr(df_agg["error"], df_agg["flair_count"])
        print(
            f"Correlation between Error Magnitude and Input Feature (FLAIR Slice Count): {corr}"
        )
    else:
        print("Insufficient validation data for correlation analysis.")

    # 7. Submission Generation
    THRESHOLD = 0.6705454545454544

    if best_auc > THRESHOLD:
        print(
            f"Validation AUC ({best_auc}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )
        generate_submission(load_cached_data=True)
    else:
        print(
            f"Validation AUC ({best_auc}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
