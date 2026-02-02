import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Add current directory to path to ensure library imports work
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything, kl_divergence_score
from library.dataset import (
    get_train_dataloader,
    get_val_dataloader,
    get_test_dataloader,
)
from library.models import PyramidFusionNet
from library.trainer import Trainer


def main():
    # 1. Setup & Configuration
    Config.setup()
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # 2. Load Metadata
    print("Loading metadata...")
    if not os.path.exists(Config.TRAIN_CSV) or not os.path.exists(Config.VAL_CSV):
        raise FileNotFoundError(
            "Metadata files not found. Ensure metadata generation was successful."
        )

    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    # 3. Prepare DataLoaders
    print("Initializing DataLoaders...")
    train_loader = get_train_dataloader(train_df)
    val_loader = get_val_dataloader(val_df)

    # 4. Initialize Model
    print("Initializing Model...")
    model = PyramidFusionNet().to(device)

    # 5. Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )

    # Calculate steps for OneCycleLR
    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LR,
        epochs=Config.EPOCHS,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1,
    )

    # 6. Training
    print("Starting Training...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        patience=Config.PATIENCE,
    )

    trainer.fit(Config.EPOCHS)

    # 7. Final Validation & Metric
    print("Loading best model for validation...")
    best_ckpt_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
    model.eval()

    val_preds = []
    val_targets = []

    print("Running validation inference...")
    with torch.no_grad():
        for eeg, spec, targets in val_loader:
            eeg = eeg.to(device, non_blocking=True)
            spec = spec.to(device, non_blocking=True)

            outputs = model(eeg, spec)

            val_preds.append(outputs.cpu().numpy())
            val_targets.append(targets.numpy())

    val_preds = np.concatenate(val_preds, axis=0)
    val_targets = np.concatenate(val_targets, axis=0)

    # Calculate Metric
    final_metric = kl_divergence_score(val_targets, val_preds)
    print(f"Final Validation Metric: {final_metric}")

    # 8. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Calculate KL per sample
    epsilon = 1e-15
    y_pred_clipped = np.clip(val_preds, epsilon, 1 - epsilon)
    y_true = val_targets.astype(np.float64)

    term1 = np.where(y_true > 0, y_true * np.log(y_true), 0.0)
    term2 = y_true * np.log(y_pred_clipped)
    kl_per_sample = np.sum(term1 - term2, axis=1)

    # Attach error to validation dataframe
    # Note: val_loader iterates sequentially, so order is preserved relative to val_df
    # However, if drop_last was True in loader (it is False for val), we'd need care.
    # val_loader has shuffle=False.
    if len(val_df) == len(kl_per_sample):
        val_df["error"] = kl_per_sample

        # Correlate with numerical metadata
        features_to_check = [
            "eeg_label_offset_seconds",
            "spectrogram_label_offset_seconds",
        ]
        print("Correlation between Error and Metadata Features:")
        for feat in features_to_check:
            if feat in val_df.columns:
                corr = val_df[feat].corr(val_df["error"])
                print(f"  {feat}: {corr}")
    else:
        print(
            f"Warning: Validation set size mismatch (DF: {len(val_df)}, Preds: {len(kl_per_sample)}). Skipping correlation analysis."
        )

    # 9. Submission Generation
    THRESHOLD = 0.6822116374969482
    if final_metric < THRESHOLD:
        print("\nMetric below threshold. Generating submission...")

        test_loader = get_test_dataloader(test_df)
        test_preds = []

        with torch.no_grad():
            for eeg, spec in test_loader:
                eeg = eeg.to(device, non_blocking=True)
                spec = spec.to(device, non_blocking=True)

                outputs = model(eeg, spec)
                test_preds.append(outputs.cpu().numpy())

        test_preds = np.concatenate(test_preds, axis=0)

        # Create submission DataFrame
        submission = pd.DataFrame(test_preds, columns=Config.TARGET_COLS)
        submission.insert(0, "eeg_id", test_df["eeg_id"])

        # Save
        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(f"\nMetric {final_metric} >= {THRESHOLD}. Submission generation skipped.")


if __name__ == "__main__":
    main()
