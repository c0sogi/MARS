import os
import torch
import torch.nn as nn
import torch.optim as optim

from library.config import Config
from library.utils import seed_everything, get_device, get_logger, save_model
from library.data import get_loaders, calculate_class_weights
from library.model import AppleResNet34, verify_initial_loss
from library.engine import train_model

# Initialize logger for this module
logger = get_logger(name="production")


def run_production_phase(val_loader, load_cached_data: bool = True) -> list:
    """
    Executes Stage 2: Production.
    Trains multiple models (Seed Ensemble) using Early Stopping on the validation set.
    Cite solution_lesson_node_00055: Seed Averaging Ensembles for Robustness.
    Cite solution_lesson_node_00032: Effectiveness of Best-Checkpointing.

    Args:
        val_loader (DataLoader): Validation data loader for early stopping.
        load_cached_data (bool): Whether to use cached data/weights.

    Returns:
        list: A list of file paths to the saved model checkpoints.
    """
    device = get_device()
    saved_model_paths = []

    logger.info("Starting Production Phase: Seed Ensemble Training with Early Stopping")
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

        # 2. Data Loading
        # mode='production' returns (train_loader, None), but we use val_loader passed in
        train_loader, _ = get_loaders(
            mode="production", load_cached_data=load_cached_data
        )

        # 3. Model Initialization
        model = AppleResNet34(pretrained=Config.PRETRAINED)
        model.to(device)

        # 4. Safety Check
        verify_initial_loss(model, train_loader, criterion, device)

        # 5. Optimization Setup
        optimizer = optim.Adam(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=Config.T_0, T_mult=Config.T_MULT, eta_min=Config.MIN_LR
        )

        # 6. Training Loop with Early Stopping
        model_filename = f"production_seed_{seed}.pth"
        save_path = os.path.join(Config.WORKING_DIR, "models", model_filename)

        train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            num_epochs=Config.MAX_EPOCHS,
            patience=5,
            save_path=save_path,
        )

        saved_model_paths.append(save_path)

        # 7. Cleanup
        del model, optimizer, scheduler, train_loader
        torch.cuda.empty_cache()

    logger.info("Production Phase Completed.")
    logger.info(f"Generated {len(saved_model_paths)} models.")

    return saved_model_paths
