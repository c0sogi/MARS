import sys
import os
import numpy as np
import torch
from scipy.stats import pearsonr

# Ensure local library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything, get_logger
from library.data import get_dataloaders
from library.model import IcebergResNet18
from library.engine import IcebergTrainer, evaluate_tta
from library.utils import calculate_log_loss
from library.calibration import run_calibration_phase
from library.production import run_production_phase

# Initialize Logger
logger = get_logger("runfile")


def analyze_failures(y_true, y_pred, angles):
    """
    Performs failure analysis on the validation set results.
    Calculates correlation between error magnitude and incidence angle.
    """
    # Flatten inputs to ensure 1D arrays for statistical functions
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    # Calculate log loss contribution per sample (clipped for stability)
    epsilon = 1e-15
    y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
    errors = -(
        y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped)
    )

    # Ensure angles are numpy array
    if isinstance(angles, torch.Tensor):
        angles = angles.cpu().numpy()

    # Calculate correlation
    # Filter out any NaNs if they exist (preprocessing should have handled this, but for safety)
    valid_mask = ~np.isnan(angles)

    if np.sum(valid_mask) > 1:
        corr, _ = pearsonr(errors[valid_mask], angles[valid_mask])
        print(f"Correlation between Error and Incidence Angle: {corr:.6f}")
    else:
        print("Not enough valid angles for correlation analysis.")

    # Class-wise Error Analysis
    mean_error_ship = np.mean(errors[y_true == 0])
    mean_error_iceberg = np.mean(errors[y_true == 1])
    print(f"Mean Error (Ship): {mean_error_ship:.6f}")
    print(f"Mean Error (Iceberg): {mean_error_iceberg:.6f}")


def main():
    # 1. Setup & Configuration
    seed_everything(Config.SEED)

    # Adjust Config for Robust Execution
    # Increasing caps to ensure convergence and using full folds for accurate calibration.
    # Cite solution_lesson_node_00040: Global Epoch Selection requires accurate averaging.
    Config.P1_MAX_EPOCHS = 60  # Increased cap to prevent premature stopping
    Config.P1_PATIENCE = 8  # Relaxed patience to handle noise
    Config.N_FOLDS = 5  # Full 5-fold CV for robust calibration

    logger.info("Configuration updated for robust execution.")

    # 2. Phase 1: Calibration
    # Run CV to find optimal epoch count (e_conv)
    logger.info("--- Starting Phase 1: Calibration ---")
    e_conv = run_calibration_phase(load_cached_data=True)
    logger.info(f"Calibration complete. Optimal convergence epoch (e_conv): {e_conv}")

    # 3. Validation Run
    # We train an ensemble on the Train Split and evaluate on the Val Split
    # to generate the required "Final Validation Metric".
    # Cite solution_lesson_node_00067: Compare Like with Like (Ensemble vs Ensemble).
    logger.info("--- Starting Validation Run (Ensemble) ---")

    # Load Split DataLoaders
    train_loader, val_loader, _ = get_dataloaders(
        batch_size=Config.BATCH_SIZE, load_cached_data=True
    )

    n_val_models = 5
    val_preds_list = []
    y_true = None
    val_angles = val_loader.dataset.inc_angles

    for i in range(n_val_models):
        current_seed = Config.SEED + i
        seed_everything(current_seed)
        logger.info(
            f"Training Validation Model {i+1}/{n_val_models} (Seed {current_seed})..."
        )

        # Initialize Model & Trainer
        model = IcebergResNet18().to(Config.DEVICE)
        trainer = IcebergTrainer(model, device=Config.DEVICE)

        # Train using the Production Protocol
        val_model = trainer.fit_phase2_production(
            train_loader, num_epochs=e_conv, fold_idx=f"Val_{i}"
        )

        # Evaluate on Validation Set
        _, preds, targets = evaluate_tta(val_model, val_loader, Config.DEVICE)
        val_preds_list.append(preds)

        if y_true is None:
            y_true = targets

        # Cleanup
        del model, trainer, val_model
        torch.cuda.empty_cache()

    # Average Predictions (Ensemble)
    avg_preds = np.mean(val_preds_list, axis=0)

    # Calculate Metric
    val_loss = calculate_log_loss(y_true, avg_preds)

    # Print the Mandatory Metric
    print(f"Final Validation Metric: {val_loss}")

    # 4. Failure Analysis
    logger.info("--- Failure Analysis ---")
    analyze_failures(y_true, avg_preds, val_angles)

    # 5. Submission Decision
    THRESHOLD = 0.16918645240183008

    if val_loss < THRESHOLD:
        logger.info(
            f"Validation metric ({val_loss}) meets threshold ({THRESHOLD}). Proceeding to Production."
        )

        # Free memory before full training
        del train_loader, val_loader
        torch.cuda.empty_cache()

        # Run Production Phase
        # This trains the ensemble on the full dataset and generates submission.csv
        run_production_phase(e_conv, load_cached_data=True)

    else:
        logger.info(
            f"Validation metric ({val_loss}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
