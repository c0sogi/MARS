import sys
import os
import torch
import numpy as np
import pandas as pd
from library.config import Config
from library.trainer import Trainer
from library.utils import seed_everything


def main():
    # 1. Setup and Initialization
    # Ensure reproducibility
    seed_everything(Config.SEED)

    # Define fast baseline parameters
    FAST_EPOCHS = 5
    THRESHOLD = 0.9975746465492954

    print(f"Starting Fast Baseline Run (Epochs: {FAST_EPOCHS})...")

    # Initialize Trainer
    # load_cached_data=True ensures we use the pre-processed parquet files in ./working
    trainer = Trainer(load_cached_data=True)

    # 2. Training
    # We pass the reduced number of epochs to the train method
    trainer.train(epochs=FAST_EPOCHS)

    # 3. Validation & Metric Calculation
    print("Computing Final Validation Metric...")
    val_auc, val_loss = trainer.validate()

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {val_auc}")

    # 4. Failure Analysis
    print("\n=== Failure Analysis ===")
    print("Calculating correlation between Error Magnitude and Features...")

    trainer.model.eval()
    device = trainer.device

    all_errors = []
    all_cont_feats = []
    all_cat_feats = []

    # Collect predictions and features from validation set
    with torch.no_grad():
        for cont_x, cat_x, targets in trainer.val_loader:
            cont_x = cont_x.to(device)
            cat_x = cat_x.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = trainer.model(cont_x, cat_x)

            # Compute probabilities (mean of 5 streams)
            probs = torch.sigmoid(outputs).mean(dim=1)

            # Calculate absolute error
            # targets shape is (Batch, 1), probs is (Batch,)
            error = torch.abs(targets.squeeze() - probs).cpu().numpy()

            all_errors.append(error)
            all_cont_feats.append(cont_x.cpu().numpy())
            all_cat_feats.append(cat_x.cpu().numpy())

    # Concatenate all batches
    all_errors = np.concatenate(all_errors)
    all_cont_feats = np.concatenate(all_cont_feats, axis=0)
    all_cat_feats = np.concatenate(all_cat_feats, axis=0)

    # Compute correlations
    correlations = []

    # Continuous Features
    cont_names = Config.get_all_cont_features()
    for i, name in enumerate(cont_names):
        if i < all_cont_feats.shape[1]:
            feat_values = all_cont_feats[:, i]
            # Handle potential constant columns to avoid NaN correlation
            if np.std(feat_values) > 0 and np.std(all_errors) > 0:
                corr = np.corrcoef(feat_values, all_errors)[0, 1]
                correlations.append((name, corr))
            else:
                correlations.append((name, 0.0))

    # Categorical Features
    cat_names = Config.get_all_cat_features()
    for i, name in enumerate(cat_names):
        if i < all_cat_feats.shape[1]:
            feat_values = all_cat_feats[:, i]
            if np.std(feat_values) > 0 and np.std(all_errors) > 0:
                corr = np.corrcoef(feat_values, all_errors)[0, 1]
                correlations.append((name, corr))
            else:
                correlations.append((name, 0.0))

    # Sort by absolute correlation magnitude
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("\nTop 10 Features associated with Model Error:")
    for name, corr in correlations[:10]:
        print(f"{name}: {corr:.6f}")

    # 5. Submission Generation
    if val_auc > THRESHOLD:
        print(
            f"\nValidation AUC ({val_auc}) > Threshold ({THRESHOLD}). Generating submission..."
        )
        trainer.generate_submission()
    else:
        print(
            f"\nValidation AUC ({val_auc}) <= Threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
