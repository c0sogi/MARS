import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr
import sys
import os

# Import from provided library files
from library.config import Config
from library.utils import seed_everything
from library.dataset import get_dataloaders
from library.model import ShallowEmbeddingMLP
from library.trainer import run_training
from library.inference import run_inference


def main():
    # --------------------------------------------------------------------------
    # 1. Setup
    # --------------------------------------------------------------------------
    seed_everything(Config.SEED)
    device = Config.DEVICE
    print(f"Orchestration started on device: {device}")

    # --------------------------------------------------------------------------
    # 2. Data Loading
    # --------------------------------------------------------------------------
    print("Loading datasets...")
    # load_cached_data=True allows using preprocessed .npz files if they exist
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=True,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        debug=False,  # Use full dataset for best score, A100 is fast enough
    )

    # --------------------------------------------------------------------------
    # 3. Model Training
    # --------------------------------------------------------------------------
    print("Initializing model...")
    model = ShallowEmbeddingMLP().to(device)

    print("Starting training...")
    # Increased epochs to 25 to allow deeper model to converge (Cite solution_lesson_node_00002)
    model = run_training(
        model,
        train_loader,
        val_loader,
        device=device,
        epochs=25,
        patience=Config.EARLY_STOPPING_PATIENCE,
    )

    # --------------------------------------------------------------------------
    # 4. Validation Assessment
    # --------------------------------------------------------------------------
    print("Performing final validation assessment...")
    model.eval()

    val_preds = []
    val_targets = []
    val_cont_feats = []
    val_cat_feats = []

    # Collect predictions, targets, and features for metric and failure analysis
    with torch.no_grad():
        for batch in val_loader:
            cont = batch["continuous"].to(device)
            cat = batch["categorical"].to(device)
            targets = batch["target"].to(device)

            outputs = model(cont, cat)

            # Move to CPU for analysis
            val_preds.append(outputs.cpu().numpy())
            val_targets.append(targets.cpu().numpy())
            val_cont_feats.append(cont.cpu().numpy())
            val_cat_feats.append(cat.cpu().numpy())

    # Concatenate batches
    val_preds = np.concatenate(val_preds).flatten()
    val_targets = np.concatenate(val_targets).flatten()
    val_cont_feats = np.concatenate(val_cont_feats, axis=0)
    val_cat_feats = np.concatenate(val_cat_feats, axis=0)

    # Calculate Metric
    final_auc = roc_auc_score(val_targets, val_preds)

    # REQUIRED OUTPUT: Print full precision metric
    print(f"Final Validation Metric: {final_auc}")

    # --------------------------------------------------------------------------
    # 5. Failure Analysis
    # --------------------------------------------------------------------------
    print("\nPerforming failure analysis...")

    # Calculate error magnitude
    errors = np.abs(val_preds - val_targets)

    # Define feature names
    # Continuous features are f_00 to f_30 excluding f_27
    cont_names = Config.CONTINUOUS_FEATURES
    # Categorical feature f_27 is split into 10 positions
    cat_names = [f"f_27_pos{i}" for i in range(10)]

    correlations = {}

    # Correlate Continuous Features with Error
    for i, name in enumerate(cont_names):
        feat_values = val_cont_feats[:, i]
        # Pearsonr returns (correlation, p-value)
        corr, _ = pearsonr(feat_values, errors)
        correlations[name] = corr

    # Correlate Categorical Features (Indices) with Error
    # While these are categorical indices, high correlation might indicate
    # specific character ranges are problematic or the embedding is struggling.
    for i, name in enumerate(cat_names):
        feat_values = val_cat_feats[:, i]
        corr, _ = pearsonr(feat_values, errors)
        correlations[name] = corr

    # Sort by absolute correlation strength
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 Features correlated with Prediction Error:")
    for name, val in sorted_corr[:5]:
        print(f"  {name}: {val:.6f}")

    # --------------------------------------------------------------------------
    # 6. Submission Generation
    # --------------------------------------------------------------------------
    # Only generate submission if metric exceeds threshold
    threshold = 0.9944106468817485
    if final_auc > threshold:
        print("\nGenerating submission file...")
        # run_inference loads the 'best_model.pth' saved by run_training
        run_inference(
            device=device,
            batch_size=Config.BATCH_SIZE,
            num_workers=Config.NUM_WORKERS,
            debug=False,
        )
    else:
        print(
            f"\nFinal AUC {final_auc} did not exceed threshold {threshold}. Submission skipped."
        )


if __name__ == "__main__":
    main()
