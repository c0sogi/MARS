import sys
import os
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

# Ensure library modules are importable
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything, kl_divergence_score
from library.data import load_data, EEGDataset
from library.model import BandAdaptiveNet
from library.train import train_one_epoch, validate_one_epoch, save_checkpoint
from library.inference import predict


def main():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    seed_everything(Config.SEED)

    # Fast Baseline Settings
    # Reducing epochs and training data size to meet time constraints
    Config.EPOCHS = 5
    TRAIN_SUBSET_SIZE = 12000

    # ==========================================
    # 2. Data Loading
    # ==========================================
    # Load Train (Cached or Raw)
    train_eeg, train_spec, train_targets = load_data(
        mode="train", load_cached_data=True
    )

    # Subsample Training Data
    total_train = len(train_eeg)
    if total_train > TRAIN_SUBSET_SIZE:
        # Use fixed seed for subsampling stability
        rng = np.random.default_rng(Config.SEED)
        indices = rng.choice(total_train, TRAIN_SUBSET_SIZE, replace=False)

        # Handle mmap slicing (creates copy in memory)
        train_eeg = train_eeg[indices]
        train_spec = train_spec[indices]
        train_targets = train_targets[indices]

    # Load Validation Data (Full)
    val_eeg, val_spec, val_targets = load_data(mode="val", load_cached_data=True)

    # Create Datasets
    train_dataset = EEGDataset(train_eeg, train_spec, train_targets, mode="train")
    val_dataset = EEGDataset(val_eeg, val_spec, val_targets, mode="val")

    # Create Loaders
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
        drop_last=False,
    )

    # ==========================================
    # 3. Model Initialization
    # ==========================================
    model = BandAdaptiveNet()
    model.to(Config.DEVICE)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=Config.EPOCHS, eta_min=Config.MIN_LR)

    best_val_score = float("inf")
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    # ==========================================
    # 4. Training Loop
    # ==========================================
    for epoch in range(1, Config.EPOCHS + 1):
        # Train
        train_loss = train_one_epoch(
            epoch, model, train_loader, optimizer, Config.DEVICE
        )

        # Validate
        val_loss, val_score = validate_one_epoch(
            epoch, model, val_loader, Config.DEVICE
        )

        # Scheduler Step
        scheduler.step()

        # Save Best
        if val_score < best_val_score:
            best_val_score = val_score
            save_checkpoint(model, best_model_path)

    # ==========================================
    # 5. Final Metrics & Failure Analysis
    # ==========================================
    print(f"Final Validation Metric: {best_val_score}")

    print("Performing Failure Analysis on Validation Set...")
    # Load best model for analysis
    model.load_state_dict(torch.load(best_model_path, map_location=Config.DEVICE))
    model.eval()

    # Generate Predictions
    val_preds_list = []
    with torch.no_grad():
        for (x_eeg, x_spec), _ in val_loader:
            x_eeg = x_eeg.to(Config.DEVICE)
            x_spec = x_spec.to(Config.DEVICE)
            with torch.amp.autocast(device_type="cuda", enabled=Config.USE_AMP):
                logits = model(x_eeg, x_spec)
                probs = torch.softmax(logits, dim=1)
            val_preds_list.append(probs.cpu().numpy())

    val_preds = np.concatenate(val_preds_list, axis=0)

    # Calculate KL per sample
    epsilon = 1e-15
    y_true = val_targets.astype(np.float64)
    y_pred = val_preds.astype(np.float64)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    # KL = sum(p * log(p/q))
    mask = y_true > 0
    kl_terms = np.zeros_like(y_true)
    kl_terms[mask] = y_true[mask] * (np.log(y_true[mask]) - np.log(y_pred[mask]))
    sample_kl = np.sum(kl_terms, axis=1)

    # Load Metadata for Correlation
    val_df = pd.read_csv(Config.VAL_CSV)
    val_df["error_kl"] = sample_kl

    features = [
        "total_votes",
        "eeg_label_offset_seconds",
        "spectogram_label_offset_seconds",
    ]
    print("Correlation between Error (KL) and Metadata features:")
    for feat in features:
        if feat in val_df.columns:
            # Drop NaNs for correlation
            tmp = val_df[[feat, "error_kl"]].dropna()
            if len(tmp) > 1:
                corr = np.corrcoef(tmp[feat], tmp["error_kl"])[0, 1]
                print(f"  {feat}: {corr:.4f}")

    # ==========================================
    # 6. Submission
    # ==========================================
    SUBMISSION_THRESHOLD = 0.8169508603799445

    if best_val_score < SUBMISSION_THRESHOLD:
        predict()


if __name__ == "__main__":
    main()
