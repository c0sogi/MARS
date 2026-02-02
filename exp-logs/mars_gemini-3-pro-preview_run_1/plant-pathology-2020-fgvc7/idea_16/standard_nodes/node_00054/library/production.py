import os
import torch
import torch.nn as nn
import torch.optim as optim

from library.config import Config
from library.utils import seed_everything, get_device, get_logger, save_model
from library.data import get_loaders, calculate_class_weights
from library.model import AppleResNet34, verify_initial_loss
from library.engine import train_one_epoch

# Initialize logger for this module
logger = get_logger(name="production")


def run_production_phase(optimal_epoch: int, load_cached_data: bool = True) -> list:
    """
    Executes Stage 2: Production.
    Trains multiple models (Seed Ensemble) on the full dataset for the determined optimal epoch.

    Args:
        optimal_epoch (int): The stopping epoch determined by the calibration phase.
        load_cached_data (bool): Whether to use cached data/weights.

    Returns:
        list: A list of file paths to the saved model checkpoints.
    """
    device = get_device()
    saved_model_paths = []

    logger.info(
        f"Starting Production Phase: Full Data Training for {optimal_epoch} Epochs"
    )
    logger.info(f"Ensemble Seeds: {Config.SEEDS}")

    # Calculate class weights once (cached) to handle imbalance
    class_weights = calculate_class_weights(load_cached_data=load_cached_data)
    logger.info(f"Using Class Weights: {class_weights}")

    # Loss function (weighted)
    criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)

    # Iterate through seeds to build the ensemble
    for seed in Config.SEEDS:
        logger.info(f"--- Training Production Model (Seed {seed}) ---")

        # 1. Reproducibility
        seed_everything(seed)

        # 2. Data Loading (Full Data)
        # mode='production' returns (train_loader, None)
        train_loader, _ = get_loaders(
            mode="production", load_cached_data=load_cached_data
        )

        # 3. Model Initialization
        model = AppleResNet34(pretrained=Config.PRETRAINED)
        model.to(device)

        # 4. Safety Check
        # Verify loss is reasonable before starting long training
        verify_initial_loss(model, train_loader, criterion, device)

        # 5. Optimization Setup
        # Must match Calibration phase settings to ensure E_opt validity
        optimizer = optim.Adam(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Scheduler
        # We use the same T_0 as calibration to maintain the same LR curve dynamics
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=Config.T_0, T_mult=Config.T_MULT, eta_min=Config.MIN_LR
        )

        # 6. Training Loop
        # Train strictly for optimal_epoch
        for epoch in range(optimal_epoch):
            avg_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )

            # Step scheduler
            scheduler.step()

            logger.info(
                f"Seed {seed} | Epoch {epoch + 1}/{optimal_epoch} | Train Loss: {avg_loss}"
            )

        # 7. Save Model
        model_filename = f"production_seed_{seed}.pth"
        save_path = os.path.join(Config.WORKING_DIR, "models", model_filename)
        save_model(model, save_path)
        saved_model_paths.append(save_path)

        logger.info(f"Model saved to {save_path}")

        # 8. Cleanup
        # Free memory for the next seed
        del model, optimizer, scheduler, train_loader
        torch.cuda.empty_cache()

    logger.info("Production Phase Completed.")
    logger.info(f"Generated {len(saved_model_paths)} models.")

    return saved_model_paths
