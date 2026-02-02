import numpy as np
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
import torch

from library.config import Config
from library.utils import seed_everything, get_logger
from library.data import process_json, IcebergDataset, get_transforms
from library.model import IcebergResNet18
from library.engine import IcebergTrainer

logger = get_logger("calibration")


def run_calibration_phase(load_cached_data=True):
    """
    Executes Phase 1: Adaptive Calibration using Stratified 5-Fold Cross-Validation.

    Objective:
        Determine the optimal number of epochs (E_conv) for convergence by averaging
        the best epochs found by Early Stopping across 5 folds.

    Args:
        load_cached_data (bool): Whether to load pre-processed data from cache.

    Returns:
        int: The calculated optimal epoch count (E_conv).
    """
    seed_everything(Config.SEED)
    logger.info("Starting Phase 1: Adaptive Calibration (Stratified 5-Fold CV)")

    # 1. Load Full Training Data
    # We use process_json directly to get the raw arrays, allowing us to create custom folds
    # rather than using the fixed split from get_dataloaders.
    data_dict = process_json(
        Config.TRAIN_JSON, "train_processed.npz", load_cached_data=load_cached_data
    )

    # Extract targets for stratification
    # data_dict['labels'] contains the 'is_iceberg' targets
    all_labels = data_dict["labels"]
    all_indices = np.arange(len(all_labels))

    # 2. Initialize Stratified K-Fold
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    optimal_epochs = []

    # 3. Cross-Validation Loop
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(all_indices, all_labels)):
        logger.info(f"--- Processing Fold {fold_idx} ---")

        # Create Datasets
        train_dataset = IcebergDataset(
            data_dict, train_idx, transform=get_transforms(mode="train")
        )
        val_dataset = IcebergDataset(
            data_dict, val_idx, transform=get_transforms(mode="val")
        )

        # Create DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Initialize Model and Trainer
        # We re-initialize for each fold to ensure a fresh start
        model = IcebergResNet18().to(Config.DEVICE)
        trainer = IcebergTrainer(model, device=Config.DEVICE)

        # Run Calibration Training
        # This uses ReduceLROnPlateau and Early Stopping to find the best epoch
        best_epoch = trainer.fit_phase1_calibration(train_loader, val_loader, fold_idx)

        optimal_epochs.append(best_epoch)
        logger.info(f"Fold {fold_idx} optimal epoch: {best_epoch}")

        # Cleanup to free memory
        del model, trainer, train_loader, val_loader, train_dataset, val_dataset
        torch.cuda.empty_cache()

    # 4. Aggregate Results
    avg_epoch = np.mean(optimal_epochs)
    e_conv = int(np.round(avg_epoch))

    logger.info("Phase 1 Complete.")
    logger.info(f"Optimal Epochs per fold: {optimal_epochs}")
    logger.info(f"Average Optimal Epoch: {avg_epoch}")
    logger.info(f"Selected E_conv (Rounded): {e_conv}")

    return e_conv
