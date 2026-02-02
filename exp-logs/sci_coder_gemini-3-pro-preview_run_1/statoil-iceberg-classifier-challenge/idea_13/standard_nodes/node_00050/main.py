import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from torch.optim.swa_utils import AveragedModel, SWALR

# Import library modules
from library import config, utils, data_loader, network, training_utils


def run_calibration(train_dataset, device):
    """
    Phase 1: 5-Fold CV to estimate convergence epoch.
    """
    print("Phase 1: Calibration (5-Fold CV)")

    labels = train_dataset.labels
    skf = StratifiedKFold(
        n_splits=config.NUM_FOLDS, shuffle=True, random_state=config.SEED
    )

    fold_best_epochs = []

    # Use config epochs
    max_epochs = config.MAX_EPOCHS_PHASE_1

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(np.zeros(len(labels)), labels)
    ):
        # Create Subsets
        train_sub = Subset(train_dataset, train_idx)
        val_sub = Subset(train_dataset, val_idx)

        # Loaders
        train_loader = DataLoader(
            train_sub,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_sub,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
        )

        # Model, Opt, Crit
        model = network.IcebergResNet18().to(device)
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )
        # Use CosineAnnealingLR to match Phase 2 and ensure smooth decay
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.MAX_EPOCHS_PHASE_1, eta_min=1e-6
        )
        criterion = nn.BCEWithLogitsLoss()

        best_loss = float("inf")
        best_epoch = 0

        # Train Loop
        for epoch in range(1, max_epochs + 1):
            # Suppress print inside loop to keep output clean, or let it print for monitoring
            # The utils function prints one line per epoch
            train_loss, train_acc = training_utils.train_one_epoch(
                model, train_loader, optimizer, criterion, device, epoch
            )
            # Use TTA=True for selection (Cite Lesson 25)
            val_loss, val_acc, _, _ = training_utils.evaluate(
                model, val_loader, criterion, device, use_tta=True
            )

            scheduler.step()

            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch = epoch

        print(f"Fold {fold+1} Best Epoch: {best_epoch} (Loss: {best_loss:.4f})")
        fold_best_epochs.append(best_epoch)

    avg_epoch = int(np.mean(fold_best_epochs))
    # Ensure at least 1 epoch
    avg_epoch = max(1, avg_epoch)
    print(f"Calibration Complete. Average Optimal Epoch: {avg_epoch}")
    return avg_epoch


def train_swa_model(model_idx, train_dataset, device, conv_epochs):
    """
    Phase 2: Train a single SWA model on the full training set.
    """
    print(f"Training SWA Model {model_idx + 1}")

    # Seed for diversity (independent models)
    utils.seed_everything(config.SEED + model_idx)

    # Loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    # Model
    model = network.IcebergResNet18().to(device)
    optimizer = optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    criterion = nn.BCEWithLogitsLoss()

    # Scheduler: Cosine Annealing for convergence phase
    # T_max matches Phase 1 for consistent trajectory (Cite Lesson 40)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.MAX_EPOCHS_PHASE_1, eta_min=1e-6
    )

    # 1. Convergence Phase
    for epoch in range(1, conv_epochs + 1):
        training_utils.train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )
        scheduler.step()

    # 2. SWA Phase
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=config.SWA_LR)

    for epoch in range(conv_epochs + 1, conv_epochs + config.SWA_EPOCHS + 1):
        training_utils.train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )
        training_utils.swa_step(model, swa_model)
        swa_scheduler.step()

    # 3. Update BN
    training_utils.update_swa_batch_norm(swa_model, train_loader, device)

    return swa_model


def main():
    # Setup
    utils.seed_everything()
    device = utils.get_device()

    # Load Data
    # train_dataset corresponds to train_metadata.csv
    train_dataset = data_loader.load_dataset("train", load_cached_data=True)
    # val_dataset corresponds to val_metadata.csv (Hold-out)
    val_dataset = data_loader.load_dataset("val", load_cached_data=True)

    # Phase 1: Calibration
    conv_epochs = run_calibration(train_dataset, device)

    # Phase 2: Ensemble Training
    ensemble_models = []
    num_models = 5

    for i in range(num_models):
        swa_model = train_swa_model(i, train_dataset, device, conv_epochs)
        ensemble_models.append(swa_model)

    # Evaluation on Hold-out Validation Set
    print("Evaluating Ensemble on Hold-out Validation Set")
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    criterion = nn.BCEWithLogitsLoss()

    ensemble_probs = []
    targets = None

    for i, model in enumerate(ensemble_models):
        # evaluate returns: loss, acc, probs, targets
        # We use TTA=True for final evaluation
        _, _, probs, t = training_utils.evaluate(
            model, val_loader, criterion, device, use_tta=True
        )
        ensemble_probs.append(probs)
        if targets is None:
            targets = t

    # Average predictions
    avg_probs = np.mean(ensemble_probs, axis=0)

    # Calculate Log Loss
    clipped_probs = np.clip(avg_probs, 1e-15, 1 - 1e-15)
    final_log_loss = log_loss(targets, clipped_probs)

    print(f"Final Validation Metric: {final_log_loss}")

    # Failure Analysis
    errors = np.abs(targets - avg_probs)
    val_angles = val_dataset.angles

    if len(errors) == len(val_angles):
        corr = np.corrcoef(errors.flatten(), val_angles)[0, 1]
        print(f"Correlation between Error and Incidence Angle: {corr:.4f}")

    # Submission
    threshold = 0.17822679498532543
    if final_log_loss < threshold:
        print("Generating Submission")
        test_dataset = data_loader.load_dataset("test", load_cached_data=True)
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
        )

        test_ensemble_probs = []
        test_ids = None

        for model in ensemble_models:
            probs, ids = training_utils.predict(
                model, test_loader, device, use_tta=True
            )
            test_ensemble_probs.append(probs)
            if test_ids is None:
                test_ids = ids

        avg_test_probs = np.mean(test_ensemble_probs, axis=0)

        training_utils.write_submission(
            test_ids, avg_test_probs, config.SUBMISSION_PATH
        )


if __name__ == "__main__":
    main()
