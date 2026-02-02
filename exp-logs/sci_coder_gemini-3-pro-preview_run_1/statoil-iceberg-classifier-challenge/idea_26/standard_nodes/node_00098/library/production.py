import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import seed_everything, get_logger
from library.data import process_json, IcebergDataset, get_transforms
from library.model import IcebergResNet18
from library.engine import IcebergTrainer, predict_test_tta

logger = get_logger("production")


def run_production_phase(e_conv: int, load_cached_data: bool = True):
    """
    Executes Phase 2: Production Training and Submission Generation.

    Trains an ensemble of 5 independent models on the full training dataset using
    the Schedule Mapping Protocol (Cosine Annealing -> SWA), then generates
    predictions using TTA and averaging.

    Args:
        e_conv (int): The optimal convergence epoch count derived from Phase 1.
        load_cached_data (bool): Whether to load pre-processed data from cache.
    """
    logger.info(f"Starting Phase 2: Production Training (Target Epochs: {e_conv})")

    # 1. Load Data
    # Load full training data
    train_data = process_json(
        Config.TRAIN_JSON, "train_processed.npz", load_cached_data=load_cached_data
    )

    # Load test data
    test_data = process_json(
        Config.TEST_JSON, "test_processed.npz", load_cached_data=load_cached_data
    )

    # Indices for full training (all samples)
    train_indices = np.arange(len(train_data["ids"]))
    test_indices = np.arange(len(test_data["ids"]))

    # Create Test DataLoader (Created once, used for all models)
    test_dataset = IcebergDataset(
        test_data, test_indices, transform=get_transforms(mode="test")
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 2. Train Ensemble
    ensemble_models = []
    n_models = 5

    for i in range(n_models):
        # Set unique seed for independence (initialization + shuffling)
        current_seed = Config.SEED + i
        seed_everything(current_seed)
        logger.info(
            f"--- Training Ensemble Model {i+1}/{n_models} (Seed {current_seed}) ---"
        )

        # Create Full Train DataLoader
        # Re-created inside loop to ensure generator seeding affects shuffling correctly
        full_train_dataset = IcebergDataset(
            train_data, train_indices, transform=get_transforms(mode="train")
        )
        full_train_loader = DataLoader(
            full_train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )

        # Initialize Model and Trainer
        model = IcebergResNet18().to(Config.DEVICE)
        trainer = IcebergTrainer(model, device=Config.DEVICE)

        # Train using Schedule Mapping Protocol (Cosine -> SWA)
        # fit_phase2_production returns the final SWA model
        swa_model = trainer.fit_phase2_production(
            full_train_loader, num_epochs=e_conv, fold_idx=i
        )

        ensemble_models.append(swa_model)

        # Clean up trainer/loader to free memory
        del trainer, full_train_loader, full_train_dataset, model
        torch.cuda.empty_cache()

    # 3. Inference & Submission
    logger.info("Starting Ensemble Inference (TTA)...")

    avg_preds = None
    test_ids = None

    for i, model in enumerate(ensemble_models):
        logger.info(f"Generating predictions for Model {i+1}/{n_models}...")

        ids, preds = predict_test_tta(model, test_loader, Config.DEVICE)

        if avg_preds is None:
            avg_preds = preds
            test_ids = ids
        else:
            avg_preds += preds

            # Sanity check for ID alignment
            if not np.array_equal(test_ids, ids):
                raise ValueError(f"Test ID mismatch in model {i}")

    # Average predictions
    avg_preds /= n_models

    # Flatten predictions (N, 1) -> (N,)
    avg_preds = avg_preds.flatten()

    # 4. Save Submission
    logger.info("Saving submission file...")
    submission_df = pd.DataFrame({"id": test_ids, "is_iceberg": avg_preds})

    # Ensure directory exists
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    submission_df.to_csv(Config.SUBMISSION_PATH, index=False, float_format="%.6f")
    logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")

    # Print head for verification
    print(submission_df.head())
