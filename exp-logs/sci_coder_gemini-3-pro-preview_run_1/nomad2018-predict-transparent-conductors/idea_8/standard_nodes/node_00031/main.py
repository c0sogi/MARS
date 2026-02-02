import os
import sys
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# Ensure the library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import set_seed, get_logger
from library.data_loader import get_dataloaders
from library.model import SIRDS_SP
from library.trainer import Trainer


def main():
    # 1. Setup
    logger = get_logger("runfile")
    set_seed(Config.SEED)

    # Override Config for fast baseline
    Config.NUM_EPOCHS = 30
    Config.PATIENCE = 10

    logger.info("Configuration set for fast baseline execution.")
    logger.info(f"Device: {Config.DEVICE}")

    # 2. Data Loading
    # Load cached data if available to save time
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE, load_cached_data=True
    )

    # 3. Model Initialization
    model = SIRDS_SP()

    # 4. Training
    trainer = Trainer(model)
    trainer.fit(train_loader, val_loader)

    # 5. Validation & Metric Calculation
    logger.info("Performing final validation...")

    # Load best model
    if not os.path.exists(Config.MODEL_CHECKPOINT_PATH):
        logger.error("No checkpoint found! Training might have failed.")
        return

    trainer.model.load_state_dict(
        torch.load(Config.MODEL_CHECKPOINT_PATH, map_location=trainer.device)
    )
    trainer.model.eval()

    all_preds_log = []
    all_targets_log = []
    all_global_feats = []

    # Collect predictions, targets, and features for analysis
    with torch.no_grad():
        for batch in val_loader:
            atomic_x = batch["atomic_x"].to(trainer.device)
            atomic_mask = batch["atomic_mask"].to(trainer.device)
            global_x = batch["global_x"].to(trainer.device)
            symmetry_x = batch["symmetry_x"].to(trainer.device)
            targets = batch["y"].to(trainer.device)

            outputs = trainer.model(atomic_x, atomic_mask, global_x, symmetry_x)

            all_preds_log.append(outputs.cpu().numpy())
            all_targets_log.append(targets.cpu().numpy())
            all_global_feats.append(global_x.cpu().numpy())

    all_preds_log = np.vstack(all_preds_log)
    all_targets_log = np.vstack(all_targets_log)
    all_global_feats = np.vstack(all_global_feats)

    # Calculate Column-wise RMSLE
    # Since targets and preds are already log1p transformed:
    # RMSLE = sqrt(mean((log1p_pred - log1p_true)^2))
    mse_per_col = np.mean((all_preds_log - all_targets_log) ** 2, axis=0)
    rmsle_per_col = np.sqrt(mse_per_col)
    final_metric = np.mean(rmsle_per_col)

    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    logger.info("Running failure analysis...")

    # Calculate error magnitude per sample (mean absolute error across targets on log scale)
    # This represents the relative error magnitude
    errors = np.mean(np.abs(all_preds_log - all_targets_log), axis=1)

    # Feature names based on data_loader.py process_data function
    feature_names = [
        "lattice_vector_1_ang",
        "lattice_vector_2_ang",
        "lattice_vector_3_ang",
        "lattice_angle_alpha",
        "lattice_angle_beta",
        "lattice_angle_gamma",
        "volume",
        "density",
        "percent_atom_al",
        "percent_atom_ga",
        "percent_atom_in",
    ]

    correlations = {}
    for i, name in enumerate(feature_names):
        if i < all_global_feats.shape[1]:
            feat_values = all_global_feats[:, i]
            # Handle constant features (std=0) to avoid warnings
            if np.std(feat_values) > 1e-9:
                corr, _ = pearsonr(errors, feat_values)
                correlations[name] = corr
            else:
                correlations[name] = 0.0

    # Sort correlations by absolute magnitude
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

    print("\nCorrelation between Error Magnitude and Global Features:")
    for name, corr in sorted_corr:
        print(f"  {name:<25}: {corr:.4f}")

    # 7. Submission
    THRESHOLD = 0.05479004207787702

    if final_metric < THRESHOLD:
        logger.info(
            f"Validation metric {final_metric} < {THRESHOLD}. Generating submission..."
        )
        trainer.predict(test_loader, output_path=Config.SUBMISSION_PATH)
    else:
        logger.warning(
            f"Validation metric {final_metric} >= {THRESHOLD}. Skipping submission generation."
        )


if __name__ == "__main__":
    main()
