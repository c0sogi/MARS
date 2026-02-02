import torch
import torch.optim as optim
import pandas as pd
import numpy as np
import os
import sys

# Import from the provided library files
from library.config import Config
from library.dataset import get_dataloaders
from library.architecture import LCDSModel
from library.engine import Engine
from library.utils import set_seed, compute_rmsle, save_submission


def run_failure_analysis(model, val_loader, device):
    """
    Performs failure analysis on the validation set by calculating the correlation
    between the model's absolute error and the input lattice features.
    """
    print("\n" + "=" * 60)
    print(" FAILURE ANALYSIS")
    print("=" * 60)

    model.eval()
    all_lattice_features = []
    all_targets_log = []
    all_preds_log = []

    # Collect predictions and ground truth
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            outputs = model(batch)
            targets = batch.y

            # Ensure target shape consistency
            if targets.dim() == 3:
                targets = targets.squeeze(1)

            all_preds_log.append(outputs.cpu())
            all_targets_log.append(targets.cpu())
            all_lattice_features.append(batch.lattice_features.cpu())

    # Concatenate all batches
    preds_log = torch.cat(all_preds_log, dim=0).numpy()
    targets_log = torch.cat(all_targets_log, dim=0).numpy()
    features = torch.cat(all_lattice_features, dim=0).numpy()

    # Convert from log(1+x) scale back to original scale for error analysis
    preds = np.expm1(preds_log)
    targets = np.expm1(targets_log)

    # Calculate absolute error
    errors = np.abs(targets - preds)

    # Get feature and target names from Config
    feature_names = Config.TABULAR_FEATURE_COLS
    target_names = Config.TARGET_COLS

    # Create DataFrame for correlation calculation
    df_analysis = pd.DataFrame(features, columns=feature_names)

    # Calculate and print correlations for each target
    for i, target_name in enumerate(target_names):
        error_col_name = f"Error_{target_name}"
        df_analysis[error_col_name] = errors[:, i]

        print(f"\nCorrelations with Error Magnitude for {target_name}:")
        # Compute correlation of features with the error column
        correlations = df_analysis.corr()[error_col_name].drop(
            [f"Error_{t}" for t in target_names if f"Error_{t}" in df_analysis.columns]
        )
        # Sort by absolute correlation value
        sorted_corr = correlations.abs().sort_values(ascending=False)
        for feat in sorted_corr.index:
            print(f"  {feat:<30}: {correlations[feat]:.4f}")


def main():
    # -------------------------------------------------------------------------
    # 1. Setup and Configuration
    # -------------------------------------------------------------------------
    # Override default Config parameters for a fast baseline execution
    Config.NUM_EPOCHS = 30  # Reduced epochs for speed
    Config.BATCH_SIZE = 64

    set_seed(Config.SEED)
    Config.setup()

    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # -------------------------------------------------------------------------
    # 2. Data Loading
    # -------------------------------------------------------------------------
    print("Preparing data loaders...")
    # Load data (utilizing cache if available in ./working)
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=True,
    )

    # -------------------------------------------------------------------------
    # 3. Model Initialization
    # -------------------------------------------------------------------------
    print("Initializing model...")
    model = LCDSModel().to(device)

    # -------------------------------------------------------------------------
    # 4. Optimizer and Scheduler
    # -------------------------------------------------------------------------
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler to reduce LR when validation loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    # -------------------------------------------------------------------------
    # 5. Training
    # -------------------------------------------------------------------------
    engine = Engine(model, optimizer, device, scheduler)

    engine.fit(
        train_loader,
        val_loader,
        num_epochs=Config.NUM_EPOCHS,
        early_stopping_patience=Config.EARLY_STOPPING_PATIENCE,
    )

    # -------------------------------------------------------------------------
    # 6. Validation Assessment
    # -------------------------------------------------------------------------
    print("\nComputing final validation metrics...")
    # engine.validate returns (loss, rmsle)
    _, val_rmsle = engine.validate(val_loader)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {val_rmsle}")

    # -------------------------------------------------------------------------
    # 7. Failure Analysis
    # -------------------------------------------------------------------------
    run_failure_analysis(model, val_loader, device)

    # -------------------------------------------------------------------------
    # 8. Submission Generation
    # -------------------------------------------------------------------------
    print("\nGenerating submission...")
    engine.generate_submission(test_loader, Config.SUBMISSION_PATH)

    print("Runfile execution completed.")


if __name__ == "__main__":
    main()
