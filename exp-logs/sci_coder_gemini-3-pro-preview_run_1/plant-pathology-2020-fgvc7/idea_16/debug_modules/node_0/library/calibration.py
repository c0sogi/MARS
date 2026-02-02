import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from library.config import Config
from library.utils import seed_everything, get_device, get_logger
from library.data import get_loaders, calculate_class_weights
from library.model import AppleResNet34, verify_initial_loss
from library.engine import train_one_epoch, evaluate

# Initialize logger for this module
logger = get_logger(name="calibration")


def run_calibration_phase(max_epochs=None, load_cached_data=True):
    """
    Executes Stage 1: Calibration.
    Runs a Stratified K-Fold Cross-Validation to determine the optimal stopping epoch (E_opt).

    Args:
        max_epochs (int, optional): Override for Config.MAX_EPOCHS. Useful for debugging.
        load_cached_data (bool): Whether to use cached data/weights.

    Returns:
        int: The optimal epoch (E_opt) based on mean validation AUC.
    """
    # Configuration
    if max_epochs is None:
        max_epochs = Config.MAX_EPOCHS

    n_folds = Config.N_FOLDS
    device = get_device()

    # Ensure reproducibility for the split logic (handled in data.py) and initialization
    seed_everything(Config.SEEDS[0])

    logger.info(f"Starting Calibration Phase: {n_folds} Folds, {max_epochs} Epochs")

    # Matrix to store validation AUCs [fold, epoch]
    auc_logs = np.zeros((n_folds, max_epochs))

    # Calculate class weights once (cached)
    class_weights = calculate_class_weights(load_cached_data=load_cached_data)
    logger.info(f"Using Class Weights: {class_weights}")

    # Loss function
    criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)

    for fold in range(n_folds):
        logger.info(f"--- Calibrating Fold {fold + 1}/{n_folds} ---")

        # 1. Data Loading
        train_loader, val_loader = get_loaders(
            mode="calibration", fold=fold, load_cached_data=load_cached_data
        )

        # 2. Model Initialization
        model = AppleResNet34(pretrained=Config.PRETRAINED)
        model.to(device)

        # 3. Safety Check
        verify_initial_loss(model, train_loader, criterion, device)

        # 4. Optimization Setup
        optimizer = optim.Adam(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Scheduler: Cosine Annealing Warm Restarts
        # T_0 is set to max_epochs to ensure one full cycle
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=Config.T_0, T_mult=Config.T_MULT, eta_min=Config.MIN_LR
        )

        # 5. Training Loop
        for epoch in range(max_epochs):
            # Train
            train_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )

            # Validate
            val_loss, val_auc = evaluate(model, val_loader, criterion, device)

            # Step Scheduler
            scheduler.step()

            # Log Metric
            auc_logs[fold, epoch] = val_auc
            logger.info(f"Fold {fold} | Epoch {epoch + 1} | Val AUC: {val_auc}")

        # Cleanup to free GPU memory
        del model, optimizer, scheduler, train_loader, val_loader
        torch.cuda.empty_cache()

    # 6. Analysis & Aggregation
    logger.info("--- Calibration Analysis ---")

    # Calculate mean AUC per epoch across all folds
    mean_aucs = np.mean(auc_logs, axis=0)
    std_aucs = np.std(auc_logs, axis=0)

    # Find the epoch with the highest mean AUC
    best_epoch_idx = np.argmax(mean_aucs)
    best_epoch = best_epoch_idx + 1
    best_auc = mean_aucs[best_epoch_idx]
    best_std = std_aucs[best_epoch_idx]

    logger.info("Mean Validation AUC per Epoch:")
    for i, auc in enumerate(mean_aucs):
        logger.info(f"Epoch {i + 1}: {auc} (std: {std_aucs[i]})")

    logger.info(f"Optimal Epoch (E_opt) identified: {best_epoch}")
    logger.info(f"Peak Mean AUC: {best_auc} +/- {best_std}")

    return best_epoch
