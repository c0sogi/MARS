import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, get_score
from library.dataset import SETIDataset
from library.model import SiameseEfficientNet
from library.engine import train_one_epoch, validate, inference


def main():
    # --- 1. Configuration & Setup ---
    Config.setup()
    seed_everything(Config.SEED)

    device = torch.device(Config.DEVICE)
    print(f"Device: {device}")

    # --- 2. Data Loading ---
    print("Loading datasets...")
    train_dataset = SETIDataset(Config.TRAIN_METADATA, mode="train")
    val_dataset = SETIDataset(Config.VAL_METADATA, mode="val")

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

    # --- 3. Model Initialization ---
    print("Initializing model...")
    model = SiameseEfficientNet()
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS  # Adjust scheduler to new epoch count
    )

    criterion = nn.BCEWithLogitsLoss()

    # --- 4. Training Loop ---
    best_auc = 0.0
    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | LR: {current_lr:.2e} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.6f}"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), Config.MODEL_PATH)
            print(f"  New best model saved! AUC: {best_auc:.6f}")

    # --- 5. Failure Analysis ---
    print("\n--- Failure Analysis ---")
    # Load best model
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    model.eval()

    # Collect stats for analysis
    all_preds = []
    all_targets = []

    # Feature accumulators
    feat_mean_on = []
    feat_mean_off = []
    feat_std_on = []
    feat_std_off = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            on_target, off_target = inputs
            on_target = on_target.to(device)
            off_target = off_target.to(device)

            # Forward pass
            outputs = model((on_target, off_target))
            probs = torch.sigmoid(outputs.squeeze(1)).cpu().numpy()

            all_preds.extend(probs)
            all_targets.extend(targets.numpy())

            # Calculate simple features on CPU for analysis
            # inputs are (Batch, 3, H, W)
            # We aggregate over (1, 2, 3) -> (Batch,)
            on_t_cpu = on_target.cpu().numpy()
            off_t_cpu = off_target.cpu().numpy()

            feat_mean_on.extend(np.mean(on_t_cpu, axis=(1, 2, 3)))
            feat_mean_off.extend(np.mean(off_t_cpu, axis=(1, 2, 3)))
            feat_std_on.extend(np.std(on_t_cpu, axis=(1, 2, 3)))
            feat_std_off.extend(np.std(off_t_cpu, axis=(1, 2, 3)))

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Calculate Error
    errors = np.abs(all_targets - all_preds)

    # Create DataFrame for correlation
    df_analysis = pd.DataFrame(
        {
            "error": errors,
            "mean_on": feat_mean_on,
            "mean_off": feat_mean_off,
            "std_on": feat_std_on,
            "std_off": feat_std_off,
            "target": all_targets,
        }
    )

    # Compute Correlations
    correlations = df_analysis.corr()["error"].drop("error")
    print("Correlation between Error Magnitude and Input Features:")
    print(correlations)

    # --- 6. Final Validation & Submission ---
    final_val_auc = get_score(all_targets, all_preds)
    print(f"\nFinal Validation Metric: {final_val_auc}")

    THRESHOLD = 0.7930069652683209

    if final_val_auc > THRESHOLD:
        print(
            f"Validation AUC ({final_val_auc}) > Threshold ({THRESHOLD}). Generating submission..."
        )

        # Load Test Data
        test_dataset = SETIDataset(Config.TEST_METADATA, mode="test")
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Run Inference with TTA
        predictions = inference(model, test_loader, device)

        # Save Submission
        df_test = pd.read_csv(Config.TEST_METADATA)
        # Ensure lengths match
        if len(predictions) != len(df_test):
            print(
                f"Warning: Prediction count ({len(predictions)}) matches test set size ({len(df_test)})?"
            )

        df_test["target"] = predictions
        submission_df = df_test[["id", "target"]]
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"Validation AUC ({final_val_auc}) <= Threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
