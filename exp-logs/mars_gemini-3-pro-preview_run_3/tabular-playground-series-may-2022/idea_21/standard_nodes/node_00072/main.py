import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import pearsonr

# Ensure library is in path
sys.path.append(os.getcwd())

from library.config import Config
from library.data_processing import set_seed, load_and_preprocess, get_dataloaders
from library.model import RSPFEModel
from library.training import train_one_epoch, evaluate


def main():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    # Override Config for Fast Baseline
    # Reducing epochs to ensure the run completes quickly while using full data
    Config.EPOCHS = 10

    print(f"Starting execution with Device: {Config.DEVICE}")
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # ==========================================
    # 2. Data Loading
    # ==========================================
    print("Loading and preprocessing data...")
    # load_cached_data=True to use existing parquet files if available
    df_train, df_val, df_test, vocab_sizes = load_and_preprocess(load_cached_data=True)

    print("Creating dataloaders...")
    train_loader, val_loader, test_loader = get_dataloaders(df_train, df_val, df_test)

    # ==========================================
    # 3. Model Initialization
    # ==========================================
    print("Initializing RSPFE Model...")
    model = RSPFEModel(vocab_sizes=vocab_sizes)
    model.to(device)

    # ==========================================
    # 4. Training Setup
    # ==========================================
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        epochs=Config.EPOCHS,
        steps_per_epoch=steps_per_epoch,
        pct_start=Config.PCT_START,
        div_factor=Config.DIV_FACTOR,
        final_div_factor=Config.FINAL_DIV_FACTOR,
    )

    criterion = nn.BCEWithLogitsLoss()

    # ==========================================
    # 5. Training Loop
    # ==========================================
    print(f"Starting training loop for {Config.EPOCHS} epochs...")
    best_auc = 0.0

    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device
        )
        val_auc, _ = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.5f} | Val AUC: {val_auc:.6f}"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)

    # ==========================================
    # 6. Final Validation & Metrics
    # ==========================================
    # Required output format
    print(f"Final Validation Metric: {best_auc}")

    # Load best model for analysis and inference
    if os.path.exists(Config.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    model.eval()

    # ==========================================
    # 7. Failure Analysis
    # ==========================================
    print("\nPerforming Failure Analysis on Validation Set...")

    # Get predictions and targets for validation set
    _, val_preds = evaluate(model, val_loader, device)
    val_targets = df_val["target"].values

    # Calculate Error Magnitude
    errors = np.abs(val_targets - val_preds)

    # Calculate correlation with continuous features
    correlations = []
    for col in Config.CONT_FEATURES:
        if col in df_val.columns:
            # df_val contains scaled values, which is fine for correlation
            feat_values = df_val[col].values

            # Handle potential NaNs if any
            if np.isnan(feat_values).any():
                continue

            corr, _ = pearsonr(errors, feat_values)
            correlations.append((col, corr))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top Feature Correlations with Error Magnitude:")
    for feat, corr in correlations[:5]:
        print(f"{feat}: {corr:.6f}")

    # ==========================================
    # 8. Submission Generation
    # ==========================================
    THRESHOLD = 0.9975746465492954

    if best_auc > THRESHOLD:
        print(f"\nMetric {best_auc} > {THRESHOLD}. Generating submission...")

        all_preds = []
        with torch.no_grad():
            for batch in test_loader:
                cat_features = batch["cat_features"].to(device)
                cont_features = batch["cont_features"].to(device)

                # Inference
                logits = model(cat_features, cont_features)
                probs = torch.sigmoid(logits)

                # Ensemble Mean (Average of 5 streams)
                mean_preds = probs.mean(dim=1)
                all_preds.extend(mean_preds.cpu().numpy())

        # Create submission dataframe
        sample_sub = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)

        # Safety check for length
        if len(all_preds) != len(sample_sub):
            print(
                f"Warning: Prediction count {len(all_preds)} != Sample submission count {len(sample_sub)}"
            )
            # Ensure we match sample submission length
            if len(all_preds) > len(sample_sub):
                all_preds = all_preds[: len(sample_sub)]
            else:
                # Pad with 0.5 if strictly necessary (should not happen)
                all_preds = all_preds + [0.5] * (len(sample_sub) - len(all_preds))

        submission = pd.DataFrame({"id": sample_sub["id"], "target": all_preds})

        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(f"\nMetric {best_auc} <= {THRESHOLD}. Skipping submission generation.")


if __name__ == "__main__":
    main()
