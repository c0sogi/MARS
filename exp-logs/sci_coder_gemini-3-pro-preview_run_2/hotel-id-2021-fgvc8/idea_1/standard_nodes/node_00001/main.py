import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

# Import library modules
from library.config import Config
from library.utils import seed_everything, get_device
from library.data import get_dataloaders
from library.model import HotelClassifier
from library.engine import train_model, validate
from library.inference import predict_and_submit


def analyze_failures(model, val_loader, device):
    """
    Performs failure analysis on the validation set.
    Calculates MAP@5 per sample and correlates error with metadata features.
    """
    print("\n=== Failure Analysis ===")
    model.eval()

    # 1. Collect per-sample scores
    all_scores = []

    # Disable gradients for inference
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)

            # Calculate MAP@5 for each sample in the batch
            k = 5
            # Top k indices: (B, k)
            _, topk_indices = logits.topk(k, dim=1)

            # Expand targets: (B, k)
            targets_expanded = labels.view(-1, 1).expand_as(topk_indices)

            # Hits: (B, k)
            hits = topk_indices == targets_expanded

            # Ranks: 1..k
            ranks = (
                torch.arange(1, k + 1, device=device)
                .float()
                .view(1, -1)
                .expand_as(hits)
            )

            # Score = 1/rank if hit, else 0. Sum over k.
            # Since single ground truth, max score is 1.0 (if rank 1), min 0.0
            scores = torch.sum(hits.float() / ranks, dim=1)

            all_scores.extend(scores.cpu().numpy())

    # 2. Align with Metadata
    # val_loader.dataset is a HotelDataset, which has .df attribute
    df_val = val_loader.dataset.df.copy()

    # Safety check for length alignment
    if len(all_scores) != len(df_val):
        # Truncate to min length if there's a mismatch (e.g. due to drop_last)
        min_len = min(len(all_scores), len(df_val))
        all_scores = all_scores[:min_len]
        df_val = df_val.iloc[:min_len]

    df_val["score"] = all_scores
    df_val["error"] = 1.0 - df_val["score"]

    # 3. Feature Engineering for Correlation

    # A. Class Frequency (Number of samples in training set)
    # We need to load train metadata to count
    if os.path.exists(Config.TRAIN_METADATA_PATH):
        train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
        class_counts = train_df["hotel_id"].value_counts().to_dict()
        df_val["train_samples"] = df_val["hotel_id"].map(class_counts).fillna(0)
    else:
        df_val["train_samples"] = 0

    # B. Chain ID (Numeric) - already in df_val['chain']

    # C. Timestamp (Year)
    if "timestamp" in df_val.columns:
        df_val["year"] = pd.to_datetime(
            df_val["timestamp"], errors="coerce"
        ).dt.year.fillna(0)
    else:
        df_val["year"] = 0

    # 4. Compute Correlations
    features = ["train_samples", "chain", "year"]
    print("Correlation between Error Magnitude (1 - MAP@5) and Input Features:")

    for feat in features:
        if feat in df_val.columns:
            # Check if feature has variance
            if df_val[feat].nunique() > 1:
                corr = df_val["error"].corr(df_val[feat])
                print(f"  {feat}: {corr:.4f}")
            else:
                print(f"  {feat}: N/A (No variance)")


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = get_device()
    print(f"Running on device: {device}")

    # 2. Data Loading
    # Using debug=False to ensure we validate on the full set for the metric requirement.
    # Using load_cached_data=True to speed up startup.
    print("Loading datasets...")
    train_loader, val_loader, test_loader, classes = get_dataloaders(
        load_cached_data=True, debug=False
    )

    # 3. Model Initialization
    print("Initializing model...")
    model = HotelClassifier(n_classes=len(classes)).to(device)

    # 4. Training Setup
    # Using a fast baseline config: 5 epochs is sufficient for a baseline on A100
    NUM_EPOCHS = 5
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.1, patience=1
    )

    # 5. Training Loop
    print(f"Starting training for {NUM_EPOCHS} epochs...")
    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=NUM_EPOCHS,
        patience=Config.PATIENCE,
    )

    # 6. Final Validation Metric
    # Calculate on full validation set using the best model
    criterion = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)
    _, final_map = validate(model, val_loader, criterion, device)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_map}")

    # 7. Failure Analysis
    analyze_failures(model, val_loader, device)

    # 8. Submission
    predict_and_submit(model, test_loader, classes, device)


if __name__ == "__main__":
    main()
