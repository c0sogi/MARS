import os
import sys
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, LaplaceLogLikelihoodLoss
from library.data import get_dataloaders
from library.model import PGANet
from library.engine import run_training, eval_fn, generate_submission

# Suppress warnings
warnings.filterwarnings("ignore")


def main():
    # ==========================================
    # 1. Setup and Configuration
    # ==========================================
    # Initialize directories and seeds
    Config.setup()
    seed_everything(Config.SEED)

    # Adjust Config for Fast Baseline Execution
    # The dataset is small (~1k samples), so 15 epochs is sufficient for convergence
    # without exceeding time limits.
    Config.N_EPOCHS = 15
    Config.NUM_WORKERS = 2

    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # ==========================================
    # 2. Data Loading
    # ==========================================
    print("Loading data...")
    # load_cached_data=True ensures we use existing preprocessed .npy files if available
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE, load_cached_data=True, debug=Config.DEBUG
    )

    # ==========================================
    # 3. Model Initialization
    # ==========================================
    print("Initializing model...")
    model = PGANet().to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Cosine Annealing Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.N_EPOCHS, eta_min=1e-6
    )

    # ==========================================
    # 4. Training
    # ==========================================
    print("Starting training...")
    best_model_path = run_training(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        device,
        num_epochs=Config.N_EPOCHS,
        patience=Config.PATIENCE,
    )

    # ==========================================
    # 5. Validation Assessment
    # ==========================================
    print("Evaluating best model...")
    # Load best weights
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    # Calculate Loss on full validation set
    val_loss = eval_fn(val_loader, model, device)

    # Convert Loss to Metric
    # Loss = -Metric, so Metric = -Loss
    final_metric = -val_loss

    # Print required metric string
    print(f"Final Validation Metric: {final_metric}")

    # ==========================================
    # 6. Failure Analysis
    # ==========================================
    print("\nPerforming Failure Analysis...")
    model.eval()

    errors = []
    features = []  # Will store [Age, Percent, Sex, Smoking]

    # Collect predictions and features for analysis
    with torch.no_grad():
        for batch in val_loader:
            img_ax = batch["image_axial"].to(device)
            img_cor = batch["image_coronal"].to(device)
            tabular = batch["tabular"].to(device)
            meta = batch["meta"].to(device)
            target = batch["target"].to(device)

            # Forward
            pred_fvc, _ = model(img_ax, img_cor, tabular, meta)

            # Calculate absolute error
            batch_errors = torch.abs(target - pred_fvc).cpu().numpy()
            errors.extend(batch_errors)

            # Collect tabular features (already normalized, but correlation still valid)
            # tabular shape: (B, 4) -> Age, Percent, Sex, Smoking
            features.extend(tabular.cpu().numpy())

    errors = np.array(errors).flatten()
    features = np.array(features)

    feature_names = ["Age", "Percent", "Sex", "SmokingStatus"]

    print("Correlation between Absolute Error and Input Features:")
    for i, name in enumerate(feature_names):
        feat_vals = features[:, i]
        # Handle constant features (e.g. if batch only has one sex) to avoid warning
        if np.std(feat_vals) == 0:
            corr = 0.0
        else:
            corr, _ = pearsonr(errors, feat_vals)
        print(f"  {name}: {corr:.4f}")

    # ==========================================
    # 7. Submission
    # ==========================================
    # Threshold check
    THRESHOLD = -6.510164260864258

    if final_metric > THRESHOLD:
        print(
            f"\nMetric ({final_metric}) > Threshold ({THRESHOLD}). Generating submission..."
        )
        generate_submission(model, test_loader, device)
    else:
        print(
            f"\nMetric ({final_metric}) <= Threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
