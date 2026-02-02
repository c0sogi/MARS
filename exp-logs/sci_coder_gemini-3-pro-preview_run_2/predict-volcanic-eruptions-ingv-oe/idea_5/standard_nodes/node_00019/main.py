import sys
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.utils import seed_everything
from library.dataset import VolcanoDataset
from library.model import HybridResNet34
from library.engine import fit, generate_submission


def main():
    # ---------------------------------------------------------
    # 1. Configuration & Setup
    # ---------------------------------------------------------
    # Override Config for fast baseline execution
    # 10 epochs is sufficient for a baseline on this dataset size (~3000 samples)
    Config.EPOCHS = 10

    # Set seeds for reproducibility
    seed_everything(Config.SEED)
    device = Config.DEVICE

    # ---------------------------------------------------------
    # 2. Data Loading
    # ---------------------------------------------------------
    # Initialize Datasets
    # load_cached_data=True ensures we use pre-computed features if available
    train_dataset = VolcanoDataset(mode="train", load_cached_data=True)
    val_dataset = VolcanoDataset(mode="val", load_cached_data=True)
    test_dataset = VolcanoDataset(mode="test", load_cached_data=True)

    # Initialize DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
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

    # ---------------------------------------------------------
    # 3. Model Initialization
    # ---------------------------------------------------------
    # Determine the input dimension for the MLP branch based on extracted features
    num_stats_features = len(train_dataset.feature_cols)

    model = HybridResNet34(num_stats_features=num_stats_features)
    model = model.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
    )

    # ---------------------------------------------------------
    # 4. Training
    # ---------------------------------------------------------
    model = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=Config.EPOCHS,
        config=Config,
    )

    # ---------------------------------------------------------
    # 5. Validation & Metric Calculation
    # ---------------------------------------------------------
    model.eval()
    val_preds = []
    val_targets = []

    # Load target scalers for inverse transformation
    # These are computed and saved by the train_dataset initialization
    if os.path.exists(Config.TARGET_MEAN_PATH) and os.path.exists(
        Config.TARGET_STD_PATH
    ):
        target_mean = np.load(Config.TARGET_MEAN_PATH)
        target_std = np.load(Config.TARGET_STD_PATH)
    else:
        # Fallback (should not happen if training ran)
        target_mean = 0.0
        target_std = 1.0

    with torch.no_grad():
        for spectrogram, features, target in val_loader:
            spectrogram = spectrogram.to(device)
            features = features.to(device)

            # Forward pass (output is scaled)
            output = model(spectrogram, features)

            # Move to CPU and flatten
            output_np = output.cpu().numpy().flatten()
            target_np = target.numpy().flatten()  # Target from loader is scaled

            # Inverse Scale: original = (scaled * std) + mean
            pred_unscaled = (output_np * target_std) + target_mean
            target_unscaled = (target_np * target_std) + target_mean

            val_preds.extend(pred_unscaled)
            val_targets.extend(target_unscaled)

    val_preds = np.array(val_preds)
    val_targets = np.array(val_targets)

    # Calculate Mean Absolute Error (MAE)
    mae = np.mean(np.abs(val_preds - val_targets))
    print(f"Final Validation Metric: {mae}")

    # ---------------------------------------------------------
    # 6. Failure Analysis
    # ---------------------------------------------------------
    errors = np.abs(val_preds - val_targets)

    # Retrieve feature values from validation dataset
    # val_dataset.df contains the merged features and metadata
    # The order is preserved because val_loader uses shuffle=False
    val_features_df = val_dataset.df[val_dataset.feature_cols].copy()

    # Add error column
    val_features_df["error"] = errors

    # Calculate correlation between features and error magnitude
    correlations = (
        val_features_df.corr()["error"].drop("error").abs().sort_values(ascending=False)
    )

    print("\nTop 5 features correlated with error magnitude:")
    print(correlations.head(5))

    # ---------------------------------------------------------
    # 7. Submission
    # ---------------------------------------------------------
    THRESHOLD = 1492505.6322055138

    if mae < THRESHOLD:
        generate_submission(model, test_loader, device, Config.SUBMISSION_PATH)
    else:
        print(
            f"Validation Metric ({mae}) is not better than threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
