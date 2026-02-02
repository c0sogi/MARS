import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import from the provided library files
from library.config import (
    SEED,
    seed_everything,
    TRAIN_META_PATH,
    VAL_META_PATH,
    TEST_META_PATH,
    WORKING_DIR,
    SUBMISSION_PATH,
    BATCH_SIZE,
    LEARNING_RATE,
    NUM_EPOCHS,
    PATIENCE,
    DEVICE,
    NUM_WORKERS,
)
from library.dataset import RMSHDDataset
from library.model import RMSHDNet
from library.engine import run_training, evaluate, predict


def main():
    # ---------------------------------------------------------
    # 1. Setup & Configuration
    # ---------------------------------------------------------
    seed_everything(SEED)
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)
    best_model_path = os.path.join(WORKING_DIR, "best_model.pth")

    print(f"Device: {DEVICE}")
    print(f"Working Directory: {WORKING_DIR}")

    # ---------------------------------------------------------
    # 2. Data Loading
    # ---------------------------------------------------------
    print("Loading metadata...")
    train_df = pd.read_parquet(TRAIN_META_PATH)
    val_df = pd.read_parquet(VAL_META_PATH)
    test_df = pd.read_parquet(TEST_META_PATH)

    print("Initializing Datasets...")
    # load_cached_data=True ensures we use pre-processed numpy arrays if available
    train_dataset = RMSHDDataset(train_df, subset_name="train", load_cached_data=True)
    val_dataset = RMSHDDataset(val_df, subset_name="val", load_cached_data=True)
    test_dataset = RMSHDDataset(test_df, subset_name="test", load_cached_data=True)

    print("Initializing DataLoaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # ---------------------------------------------------------
    # 3. Model Initialization
    # ---------------------------------------------------------
    print("Initializing Model...")
    model = RMSHDNet().to(DEVICE)

    # Optimizer & Loss
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    # ---------------------------------------------------------
    # 4. Training
    # ---------------------------------------------------------
    model = run_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=DEVICE,
        num_epochs=NUM_EPOCHS,
        patience=PATIENCE,
        save_path=best_model_path,
    )

    # ---------------------------------------------------------
    # 5. Final Validation
    # ---------------------------------------------------------
    print("\nRunning Final Validation...")
    # Ensure best model is loaded (run_training returns model with best weights)
    val_loss, val_auc = evaluate(model, val_loader, criterion, DEVICE)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {val_auc}")

    # ---------------------------------------------------------
    # 6. Failure Analysis
    # ---------------------------------------------------------
    print("\nRunning Failure Analysis...")

    # Get raw probabilities for the validation set
    val_probs = predict(model, val_loader, DEVICE)
    val_ids = val_dataset.get_ids()

    # Create a DataFrame to align predictions with metadata
    preds_df = pd.DataFrame({"BraTS21ID": val_ids, "prob": val_probs})

    # Merge with original validation metadata to get features (paths, targets)
    analysis_df = pd.merge(val_df, preds_df, on="BraTS21ID", how="inner")

    # Calculate Absolute Error
    analysis_df["error"] = (analysis_df["MGMT_value"] - analysis_df["prob"]).abs()

    # Extract Meta-Features (Slice Counts) for Correlation
    modalities = ["flair", "t1w", "t1wce", "t2w"]
    for mod in modalities:
        col_name = f"{mod}_paths"
        # Handle cases where paths might be None
        analysis_df[f"{mod}_count"] = analysis_df[col_name].apply(
            lambda x: len(x) if x is not None else 0
        )

    # Calculate Correlations
    feature_cols = [f"{m}_count" for m in modalities]
    correlations = analysis_df[feature_cols + ["error"]].corr()["error"].drop("error")

    print("Correlation between Error and Modality Slice Counts:")
    print(correlations)

    # ---------------------------------------------------------
    # 7. Submission Generation
    # ---------------------------------------------------------
    THRESHOLD = 0.6978181818181817

    if val_auc > THRESHOLD:
        print(
            f"\nValidation AUC ({val_auc:.6f}) exceeds threshold ({THRESHOLD:.6f}). Generating submission..."
        )

        # Predict on Test Set
        test_probs = predict(model, test_loader, DEVICE)
        test_ids = test_dataset.get_ids()

        # Create Submission DataFrame
        submission_df = pd.DataFrame({"BraTS21ID": test_ids, "MGMT_value": test_probs})

        # Save to CSV
        submission_df.to_csv(SUBMISSION_PATH, index=False)
        print(f"Submission saved to {SUBMISSION_PATH}")

        # Verify file creation
        if os.path.exists(SUBMISSION_PATH):
            print("Submission file verified.")
            print(submission_df.head())
    else:
        print(
            f"\nValidation AUC ({val_auc:.6f}) did not exceed threshold ({THRESHOLD:.6f}). Submission skipped."
        )


if __name__ == "__main__":
    main()
