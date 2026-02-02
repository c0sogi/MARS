import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

# Import provided library modules
import library.config as config
from library.utils import seed_everything
from library.data_loader import MelanomaDataset, get_transforms, preprocess_metadata
from library.model import EfficientNetFusion
from library.engine import train_one_epoch, evaluate, predict_tta, generate_submission


def main():
    # 1. Setup
    print("Setting up environment...")
    seed_everything(config.SEED)
    device = config.DEVICE

    # 2. Data Preparation (Using Subsets for Speed)
    print("Loading and preprocessing metadata...")
    # Load full metadata and tabular features
    train_df, val_df, test_df, train_tab, val_tab, test_tab = preprocess_metadata(
        load_cached_data=True
    )

    # Create small subsets (128 samples = 2 batches) to ensure the demo runs quickly
    SUBSET_SIZE = 128
    print(f"Creating data subsets (Size: {SUBSET_SIZE})...")

    train_subset_df = train_df.iloc[:SUBSET_SIZE].reset_index(drop=True)
    train_subset_tab = train_tab[:SUBSET_SIZE]

    val_subset_df = val_df.iloc[:SUBSET_SIZE].reset_index(drop=True)
    val_subset_tab = val_tab[:SUBSET_SIZE]

    test_subset_df = test_df.iloc[:SUBSET_SIZE].reset_index(drop=True)
    test_subset_tab = test_tab[:SUBSET_SIZE]

    # 3. Instantiate Datasets and Loaders
    print("Creating Datasets and DataLoaders...")
    train_dataset = MelanomaDataset(
        train_subset_df,
        train_subset_tab,
        transform=get_transforms("train"),
        is_test=False,
    )

    val_dataset = MelanomaDataset(
        val_subset_df, val_subset_tab, transform=get_transforms("val"), is_test=False
    )

    # We skip WeightedRandomSampler for this demo to keep logic simple and fast
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    # 4. Verification: Data Shapes
    print("Verifying data loader shapes...")
    # Fetch one batch
    images, tabular, targets = next(iter(train_loader))

    # Assert Image Shape: (Batch, 3, H, W)
    assert images.shape == (
        config.BATCH_SIZE,
        3,
        config.IMAGE_SIZE,
        config.IMAGE_SIZE,
    ), f"Image shape mismatch: {images.shape}"

    # Assert Tabular Shape: (Batch, Features)
    assert (
        tabular.ndim == 2 and tabular.shape[0] == config.BATCH_SIZE
    ), f"Tabular shape mismatch: {tabular.shape}"

    # Assert Target Shape: (Batch,)
    assert targets.shape == (
        config.BATCH_SIZE,
    ), f"Target shape mismatch: {targets.shape}"

    print("Data verification passed.")

    # 5. Model Initialization
    print("Initializing model...")
    num_tab_features = train_tab.shape[1]
    model = EfficientNetFusion(num_tabular_features=num_tab_features).to(device)

    # Verification: Forward Pass
    print("Verifying model forward pass...")
    with torch.no_grad():
        logits = model(images.to(device), tabular.to(device))

    # Assert Output Shape: (Batch, 1) - Binary Classification Logits
    assert logits.shape == (
        config.BATCH_SIZE,
        1,
    ), f"Model output shape mismatch: {logits.shape}"
    print("Model verification passed.")

    # 6. Training (1 Epoch on Subset)
    print("Starting training (1 epoch on subset)...")
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)

    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
    print(f"Train Loss: {train_loss:.4f}")

    # 7. Evaluation
    print("Evaluating on validation subset...")
    val_loss, val_auc = evaluate(model, val_loader, criterion, device)
    print(f"Validation Loss: {val_loss:.4f} | Validation AUC: {val_auc:.4f}")

    # 8. Prediction with TTA
    print("Running prediction (TTA)...")
    # Using subset of test data and 1 TTA step for speed
    preds = predict_tta(
        model,
        test_subset_df,
        test_subset_tab,
        tta_steps=1,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        device=device,
    )

    assert len(preds) == len(
        test_subset_df
    ), f"Prediction count mismatch: {len(preds)} vs {len(test_subset_df)}"

    # 9. Submission
    print("Generating submission file...")
    output_path = "./working/demo_submission.csv"
    generate_submission(test_subset_df["image_name"], preds, output_path=output_path)

    print("Demo completed successfully.")


if __name__ == "__main__":
    main()
