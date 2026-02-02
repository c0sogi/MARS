import sys
import os
import warnings

# Suppress warnings and progress bars
warnings.filterwarnings("ignore")
os.environ["TQDM_DISABLE"] = "1"

# Monkey-patch tqdm to ensure silence before importing libraries that use it
import tqdm.auto


def silent_tqdm(iterable, *args, **kwargs):
    return iterable


tqdm.auto.tqdm = silent_tqdm

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, kl_divergence_score, softmax_and_normalize
from library.dataset import HMSDataset
from library.models import DeepSupervisedModel
from library.engine import Trainer, predict_and_submit


def run():
    # 1. Setup
    Config.setup()
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    print(f"Running on device: {device}")

    # 2. Data Loading
    print("Initializing Datasets...")
    # Train dataset (applies global random subsampling internally via Config)
    train_dataset = HMSDataset(mode="train", use_cache=True)
    val_dataset = HMSDataset(mode="val", use_cache=True)
    test_dataset = HMSDataset(mode="test", use_cache=True)

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

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # 3. Model Initialization
    model = DeepSupervisedModel(
        num_classes=Config.NUM_CLASSES,
        eeg_channels=Config.EEG_CHANNELS,
        spec_channels=Config.SPEC_CHANNELS,
    )
    model.to(device)

    # 4. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.MAX_LR,
        epochs=Config.EPOCHS,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1,
    )

    # 5. Training
    trainer = Trainer(
        model=model,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    trainer.fit()

    # 6. Final Validation & Failure Analysis
    print("\nRunning Final Validation & Failure Analysis...")

    # Load best model
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    model.eval()

    # Collect predictions and targets
    all_preds = []
    all_targets = []

    # We need to access metadata for failure analysis, so we'll align by index
    # The val_loader is sequential (shuffle=False), so indices match val_dataset.df

    with torch.no_grad():
        for batch in val_loader:
            X_eeg, X_spec, y = batch
            X_eeg = X_eeg.to(device)
            X_spec = X_spec.to(device)

            # Forward pass (Joint Head)
            joint_logit, _, _ = model(X_eeg, X_spec)
            probs = softmax_and_normalize(joint_logit.cpu().numpy())

            all_preds.append(probs)
            all_targets.append(y.numpy())

    y_pred = np.vstack(all_preds)
    y_true = np.vstack(all_targets)

    # Compute Final Metric
    final_score = kl_divergence_score(y_pred, y_true)
    print(f"Final Validation Metric: {final_score}")

    # --- Failure Analysis ---
    print("\n--- Failure Analysis ---")

    # Calculate KL per sample
    # Re-implement per-sample KL logic locally to get array
    epsilon = 1e-15
    y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
    mask = y_true > 0
    kl_matrix = np.zeros_like(y_true)
    kl_matrix[mask] = y_true[mask] * (
        np.log(y_true[mask]) - np.log(y_pred_clipped[mask])
    )
    sample_kl = np.sum(kl_matrix, axis=1)

    # Load Validation Metadata
    val_df = pd.read_csv(Config.VAL_CSV)

    # Ensure alignment
    if len(val_df) != len(sample_kl):
        print(
            f"Warning: Metadata length {len(val_df)} != Prediction length {len(sample_kl)}"
        )
        # Truncate to match (should not happen with correct loaders)
        min_len = min(len(val_df), len(sample_kl))
        val_df = val_df.iloc[:min_len]
        sample_kl = sample_kl[:min_len]

    val_df["error_kl"] = sample_kl

    # Correlation Analysis
    # Select numerical columns of interest
    analysis_cols = [
        "eeg_label_offset_seconds",
        "spectrogram_label_offset_seconds",
        "seizure_vote",
        "lpd_vote",
        "gpd_vote",
        "lrda_vote",
        "grda_vote",
        "other_vote",
    ]

    # Filter only columns present in dataframe
    valid_cols = [c for c in analysis_cols if c in val_df.columns]

    print("Correlation between Error (KL) and Features:")
    correlations = (
        val_df[valid_cols].corrwith(val_df["error_kl"]).sort_values(ascending=False)
    )
    print(correlations)

    # 7. Submission
    THRESHOLD = 0.6822116374969482

    if final_score < THRESHOLD:
        print(
            f"\nValidation score ({final_score}) meets threshold ({THRESHOLD}). Generating submission..."
        )
        predict_and_submit(test_loader, model, device)
    else:
        print(
            f"\nValidation score ({final_score}) does NOT meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    run()
