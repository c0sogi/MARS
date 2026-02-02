import numpy as np
import torch
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from library import config, utils, data, model, engine


def run_calibration(
    load_cached_data=True, epochs=config.CALIBRATION_EPOCHS, n_folds=config.NUM_FOLDS
):
    """
    Executes Phase 1: Calibration (Global Epoch Selection).
    Runs Stratified K-Fold Cross-Validation to determine the optimal number of epochs
    and learning rate schedule milestones.

    Args:
        load_cached_data (bool): Whether to use cached numpy arrays for data.
        epochs (int): Maximum number of epochs to train per fold.
        n_folds (int): Number of folds for Cross-Validation.

    Returns:
        tuple: (best_epoch, avg_lr_milestones)
            best_epoch (int): The epoch number with the lowest average validation loss.
            avg_lr_milestones (list): List of average epochs where LR reduction occurred.
    """
    print("Starting Phase 1: Calibration (Stratified 5-Fold CV)...")

    # 1. Set Seed for Reproducibility
    utils.seed_everything(config.RANDOM_SEED)

    # 2. Load All Labeled Data
    # We use the full content of train.json for Cross-Validation
    images, angles, labels, ids = data.process_json_data(
        config.TRAIN_JSON, "train", load_cached_data=load_cached_data
    )

    # 3. Setup Stratified K-Fold
    skf = StratifiedKFold(
        n_splits=n_folds, shuffle=True, random_state=config.RANDOM_SEED
    )

    # Storage for metrics
    # Shape: (n_folds, epochs)
    fold_val_losses = np.zeros((n_folds, epochs))

    # Store list of reduction epochs for each fold
    # e.g. [[12, 24], [11, 25], ...]
    fold_lr_reductions = []

    device = config.DEVICE

    # 4. Iterate Folds
    for fold_idx, (train_index, val_index) in enumerate(skf.split(images, labels)):
        print(f"\n--- Fold {fold_idx + 1}/{n_folds} ---")

        # Split Data
        X_train, X_val = images[train_index], images[val_index]
        a_train, a_val = angles[train_index], angles[val_index]
        y_train, y_val = labels[train_index], labels[val_index]
        id_train, id_val = ids[train_index], ids[val_index]

        # Create Datasets
        train_dataset = data.IcebergDataset(
            X_train,
            a_train,
            y_train,
            id_train,
            transform=data.get_transforms(mode="train"),
        )
        val_dataset = data.IcebergDataset(
            X_val, a_val, y_val, id_val, transform=data.get_transforms(mode="val")
        )

        # Create DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
        )

        # Initialize Model
        net = model.IcebergResNet18()
        net.to(device)

        # Initialize Optimizer and Scheduler
        optimizer = optim.AdamW(
            net.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
        )

        # ReduceLROnPlateau for reactive scheduling during calibration
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=config.PATIENCE
        )

        current_fold_reductions = []

        # Training Loop
        for epoch in range(epochs):
            print(f"Fold {fold_idx + 1} Epoch {epoch + 1}/{epochs}")

            # Train
            engine.train_one_epoch(net, train_loader, optimizer, device, epoch + 1)

            # Validate
            val_loss = engine.evaluate(net, val_loader, device)
            fold_val_losses[fold_idx, epoch] = val_loss

            # Check for LR reduction
            current_lr = optimizer.param_groups[0]["lr"]
            scheduler.step(val_loss)
            new_lr = optimizer.param_groups[0]["lr"]

            if new_lr < current_lr:
                print(f"LR reduced at epoch {epoch + 1} from {current_lr} to {new_lr}")
                current_fold_reductions.append(epoch + 1)

        fold_lr_reductions.append(current_fold_reductions)

        # Clean up to save memory
        del net, optimizer, scheduler, train_loader, val_loader
        torch.cuda.empty_cache()

    # 5. Aggregate Results
    print("\n--- Calibration Results ---")

    # Calculate average validation loss per epoch
    avg_val_losses = np.mean(fold_val_losses, axis=0)

    # Find optimal epoch (1-based index)
    best_epoch_idx = np.argmin(avg_val_losses)
    best_epoch = int(best_epoch_idx + 1)
    min_loss = avg_val_losses[best_epoch_idx]

    print(f"Average Validation Losses per Epoch: {avg_val_losses}")
    print(f"Optimal Epoch (E*): {best_epoch}")
    print(f"Minimum Average Validation Loss: {min_loss}")

    # Calculate average LR milestones
    # We align reductions by their occurrence order (1st reduction, 2nd reduction, etc.)
    max_reductions = (
        max(len(r) for r in fold_lr_reductions) if fold_lr_reductions else 0
    )
    avg_milestones = []

    if max_reductions > 0:
        for i in range(max_reductions):
            # Collect the i-th reduction epoch from folds that had at least i+1 reductions
            epochs_at_i = [r[i] for r in fold_lr_reductions if len(r) > i]

            if epochs_at_i:
                avg_epoch = int(round(np.mean(epochs_at_i)))
                avg_milestones.append(avg_epoch)

    # Ensure milestones are sorted and unique
    avg_milestones = sorted(list(set(avg_milestones)))

    print(f"Raw Fold LR Reductions: {fold_lr_reductions}")
    print(f"Calculated Average Milestones: {avg_milestones}")

    return best_epoch, avg_milestones
