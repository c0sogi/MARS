import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

# Import from provided libraries
from library.config import (
    DEVICE,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    EARLY_STOPPING_PATIENCE,
    CACHE_DIR,
    NUM_WORKERS,
    SEED,
    INPUT_DIR,
    VAL_METADATA_PATH,
    SUBMISSION_PATH,
)
from library.utils import set_seed
from library.model import EarlyFusionNet
from library.data import get_datasets
from library.train import train_one_epoch, validate


def run_failure_analysis(model, val_loader, val_metadata_path, device):
    """
    Performs failure analysis on the validation set.
    Calculates error magnitude and correlates it with input features (FLAIR file count).
    """
    print("\n=== Failure Analysis ===")
    model.eval()
    all_preds = []
    all_targets = []
    all_ids = []

    # 1. Get Predictions
    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            targets = batch["target"].to(device)
            ids = batch["BraTS21ID"]

            logits = model(images)
            probs = torch.sigmoid(logits)

            all_preds.extend(probs.cpu().numpy().flatten())
            all_targets.extend(targets.cpu().numpy().flatten())
            all_ids.extend(ids.numpy().flatten())

    # 2. Create DataFrame
    df_results = pd.DataFrame(
        {"BraTS21ID": all_ids, "target": all_targets, "pred": all_preds}
    )

    # Aggregate by subject (mean pred)
    df_subject = (
        df_results.groupby("BraTS21ID")
        .agg({"target": "max", "pred": "mean"})
        .reset_index()
    )

    # Calculate Error
    df_subject["error"] = np.abs(df_subject["target"] - df_subject["pred"])

    # 3. Extract Features for Correlation
    # We will compute 'flair_count' as a proxy for brain volume/scan complexity
    # Load metadata to get paths
    if os.path.exists(val_metadata_path):
        df_meta = pd.read_csv(val_metadata_path)

        # Map ID to flair path
        id_to_path = pd.Series(
            df_meta.flair_path.values, index=df_meta.BraTS21ID
        ).to_dict()

        flair_counts = []
        for sid in df_subject["BraTS21ID"]:
            rel_path = id_to_path.get(sid)
            if rel_path:
                full_path = os.path.join(INPUT_DIR, rel_path)
                # Fast count
                try:
                    if os.path.exists(full_path):
                        cnt = len(
                            [
                                name
                                for name in os.listdir(full_path)
                                if name.endswith(".dcm")
                            ]
                        )
                    else:
                        cnt = 0
                except:
                    cnt = 0
            else:
                cnt = 0
            flair_counts.append(cnt)

        df_subject["flair_count"] = flair_counts

        # 4. Compute Correlation
        if df_subject["flair_count"].std() > 0:
            corr = df_subject["error"].corr(df_subject["flair_count"])
            print(
                f"Correlation between Error Magnitude and FLAIR Slice Count: {corr:.10f}"
            )
        else:
            print("Could not compute correlation (constant feature).")
    else:
        print("Validation metadata not found, skipping feature correlation.")

    print("Top 5 Worst Predictions:")
    print(df_subject.sort_values("error", ascending=False).head(5))


def generate_submission(model, test_loader, device, threshold_auc, val_auc):
    """
    Generates submission file if validation metric meets threshold.
    """
    print("\n=== Submission Generation ===")

    # Threshold Check
    if val_auc <= threshold_auc:
        print(
            f"Validation AUC ({val_auc:.6f}) did not meet threshold ({threshold_auc:.6f}). Skipping submission."
        )
        return

    print("Threshold met. Generating predictions for test set...")
    model.eval()
    all_preds = []
    all_ids = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            ids = batch["BraTS21ID"]

            logits = model(images)
            probs = torch.sigmoid(logits)

            all_preds.extend(probs.cpu().numpy().flatten())
            all_ids.extend(ids.numpy().flatten())

    # Aggregate
    df_test = pd.DataFrame({"BraTS21ID": all_ids, "MGMT_value": all_preds})
    # Group by ID and take mean of the expanded samples
    df_submission = (
        df_test.groupby("BraTS21ID").agg({"MGMT_value": "mean"}).reset_index()
    )

    # Ensure directory exists
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

    # Save
    df_submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")
    print(df_submission.head())


def main():
    # 1. Setup
    set_seed(SEED)
    os.makedirs(CACHE_DIR, exist_ok=True)
    model_save_path = os.path.join(CACHE_DIR, "best_model.pth")

    print(f"Running on Device: {DEVICE}")

    # 2. Data Loading
    # We use the provided get_datasets which handles caching and deterministic expansion
    train_dataset, val_dataset, test_dataset = get_datasets(load_cached_data=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    # 3. Model & Training
    model = EarlyFusionNet().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    best_auc = 0.0
    patience = 0

    print(f"Starting training for {EPOCHS} epochs...")

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_auc = validate(model, val_loader, criterion, DEVICE)

        print(
            f"Epoch {epoch}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.6f}"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            patience = 0
            torch.save(model.state_dict(), model_save_path)
        else:
            patience += 1
            if patience >= EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

    print(f"Training finished. Best AUC: {best_auc}")

    # 4. Final Evaluation & Metric Printing
    # Load best model for final verification and analysis
    if os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path, map_location=DEVICE))

    # Re-evaluate on full validation set
    _, final_val_auc = validate(model, val_loader, criterion, DEVICE)
    print(f"Final Validation Metric: {final_val_auc}")

    # 5. Failure Analysis
    run_failure_analysis(model, val_loader, VAL_METADATA_PATH, DEVICE)

    # 6. Submission
    TARGET_THRESHOLD = 0.6705454545454544
    generate_submission(model, test_loader, DEVICE, TARGET_THRESHOLD, final_val_auc)


if __name__ == "__main__":
    main()
