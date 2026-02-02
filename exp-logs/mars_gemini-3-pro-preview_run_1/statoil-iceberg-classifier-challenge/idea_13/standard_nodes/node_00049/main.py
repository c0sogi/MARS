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
    Phase 1: 5-Fold CV to estimate convergence epoch using Global Epoch Selection.
    Cite solution_lesson_node_00040
    """
    print("Phase 1: Calibration (5-Fold CV)")

    labels = train_dataset.labels
    skf = StratifiedKFold(
        n_splits=config.NUM_FOLDS, shuffle=True, random_state=config.SEED
    )

    # Use config epochs
    max_epochs = config.MAX_EPOCHS_PHASE_1

    # Store validation losses: [Fold, Epoch]
    val_loss_history = np.zeros((config.NUM_FOLDS, max_epochs))

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(np.zeros(len(labels)), labels)
    ):
        print(f"Starting Fold {fold+1}...")
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
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=config.PATIENCE
        )
        # Add Label Smoothing - Cite solution_lesson_node_00005
        criterion = nn.BCEWithLogitsLoss()

        # Train Loop - Run for full duration to capture global curve
        for epoch in range(1, max_epochs + 1):
            train_loss, train_acc = training_utils.train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
                epoch,
                label_smoothing=0.05,
            )
            val_loss, val_acc, _, _ = training_utils.evaluate(
                model, val_loader, criterion, device, use_tta=False
            )

            # Store loss (epoch is 1-based, index is 0-based)
            val_loss_history[fold, epoch - 1] = val_loss

            scheduler.step(val_loss)

    # Global Epoch Selection: Average loss across folds per epoch
    avg_val_loss_per_epoch = np.mean(val_loss_history, axis=0)
    best_epoch_idx = np.argmin(avg_val_loss_per_epoch)
    best_epoch = best_epoch_idx + 1
    best_loss = avg_val_loss_per_epoch[best_epoch_idx]

    print(
        f"Calibration Complete. Global Optimal Epoch: {best_epoch} (Avg Loss: {best_loss:.4f})"
    )
    return best_epoch


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
    # Add Label Smoothing - Cite solution_lesson_node_00005
    criterion = nn.BCEWithLogitsLoss()

    # Scheduler: Cosine Annealing for convergence phase
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=conv_epochs)

    # 1. Convergence Phase
    for epoch in range(1, conv_epochs + 1):
        training_utils.train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            epoch,
            label_smoothing=0.05,
        )
        scheduler.step()

    # 2. SWA Phase
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=config.SWA_LR)

    for epoch in range(conv_epochs + 1, conv_epochs + config.SWA_EPOCHS + 1):
        training_utils.train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            epoch,
            label_smoothing=0.05,
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
