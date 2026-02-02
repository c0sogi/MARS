import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import (
    SEED,
    BATCH_SIZE,
    WEIGHT_DECAY,
    MAX_LR,
    PCT_START,
    MODEL_SAVE_PATH,
    ID_COL,
    TARGET_COL,
    EPOCHS,
)
from library.data_utils import process_data, ManufacturingDataset, set_seed
from library.model import HPFEModel
from library.train_eval import train_epoch, validate, predict


def main():
    # 1. Setup
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # 2. Data Loading & Preparation
    # Load cached data to save time
    train_df, val_df, test_df, vocab_sizes, cont_cols, cat_cols = process_data(
        load_cached_data=True
    )

    # Create Datasets
    # We use the full provided training set (640k samples) as it fits easily within memory
    # and the A100's compute capability, ensuring we meet the high AUC threshold.
    train_dataset = ManufacturingDataset(train_df, cont_cols, cat_cols)
    val_dataset = ManufacturingDataset(val_df, cont_cols, cat_cols)
    test_dataset = ManufacturingDataset(test_df, cont_cols, cat_cols, is_test=True)

    # Create DataLoaders
    # Pin memory and num_workers for faster data transfer
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # 3. Model Initialization
    model = HPFEModel(vocab_sizes=vocab_sizes, num_continuous=len(cont_cols))
    model.to(device)

    # 4. Training Configuration
    # Extended Schedule: Restore to 50 epochs (Cite solution_lesson_node_00080)
    run_epochs = EPOCHS

    optimizer = optim.AdamW(
        model.parameters(),
        lr=MAX_LR / 10,  # Initial LR (overridden by OneCycle)
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=MAX_LR,
        steps_per_epoch=len(train_loader),
        epochs=run_epochs,
        pct_start=PCT_START,
    )

    criterion = nn.BCEWithLogitsLoss()

    # 5. Training Loop
    print(f"Starting training for {run_epochs} epochs...")
    best_auc = 0.0

    for epoch in range(run_epochs):
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, device
        )
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        # print(f"Epoch {epoch+1}/{run_epochs} | Train Loss: {train_loss:.4f} | Val AUC: {val_auc:.5f}")

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

    # 6. Final Evaluation
    print("Training complete. Loading best model for evaluation...")
    if os.path.exists(MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))

    _, final_auc = validate(model, val_loader, criterion, device)

    # REQUIRED OUTPUT: Print full precision metric
    print(f"Final Validation Metric: {final_auc}")

    # 7. Failure Analysis
    print("\nPerforming Failure Analysis...")
    model.eval()
    val_probs = []
    val_targets = []

    # Inference for analysis
    with torch.no_grad():
        for batch in val_loader:
            continuous = batch["continuous"].to(device)
            categorical = batch["categorical"].to(device)
            targets = batch["target"].to(device)

            stream_outputs = model(continuous, categorical)

            # Average predictions across streams
            probs_list = [torch.sigmoid(out) for out in stream_outputs]
            avg_probs = torch.stack(probs_list).mean(dim=0)

            val_probs.append(avg_probs.cpu().numpy())
            val_targets.append(targets.cpu().numpy())

    val_probs = np.concatenate(val_probs).flatten()
    val_targets = np.concatenate(val_targets).flatten()

    # Calculate Absolute Error
    errors = np.abs(val_targets - val_probs)

    # Create analysis dataframe using the processed validation features
    # We correlate error with input features
    analysis_df = val_df[cont_cols + cat_cols].copy()
    analysis_df["error"] = errors

    # Compute correlations
    correlations = (
        analysis_df.corr()["error"].drop("error").sort_values(ascending=False)
    )

    print("Top 5 Features correlated with Error:")
    print(correlations.head(5))

    # 8. Conditional Submission
    THRESHOLD = 0.9975746465492954

    if final_auc > THRESHOLD:
        print(
            f"\nValidation metric {final_auc} > {THRESHOLD}. Generating submission..."
        )

        # Generate predictions
        sub_df = predict(model, test_loader, device)

        # Ensure directory exists
        submission_dir = "./submission"
        os.makedirs(submission_dir, exist_ok=True)
        sub_path = os.path.join(submission_dir, "submission.csv")

        # Save
        sub_df.to_csv(sub_path, index=False)
        print(f"Submission saved to {sub_path}")
    else:
        print(
            f"\nValidation metric {final_auc} did not meet threshold {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
