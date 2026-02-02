import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from collections import defaultdict

from library.config import Config
from library.dataset import get_dataset, IcebergDataset
from library.model import IcebergResNet18
from library.engine import train_one_epoch
from library.utils import seed_everything
from library.augmentation import get_training_transforms, get_validation_transforms


def evaluate_tta(model, dataloader, device):
    """
    Evaluates the model using Klein Four-Group TTA (Original, HFlip, VFlip, R180).
    Returns the Binary Cross Entropy loss of the averaged probabilities.
    """
    model.eval()
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for data in dataloader:
            images, angles, labels = data
            images = images.to(device)
            angles = angles.to(device)
            labels = labels.to(device)

            # 1. Original
            out1 = torch.sigmoid(model(images, angles))

            # 2. Horizontal Flip (dim 3)
            images_h = torch.flip(images, [3])
            out2 = torch.sigmoid(model(images_h, angles))

            # 3. Vertical Flip (dim 2)
            images_v = torch.flip(images, [2])
            out3 = torch.sigmoid(model(images_v, angles))

            # 4. Rotate 180 (H + V)
            images_r180 = torch.flip(images, [2, 3])
            out4 = torch.sigmoid(model(images_r180, angles))

            # Average Probabilities
            avg_probs = (out1 + out2 + out3 + out4) / 4.0

            all_probs.append(avg_probs)
            all_targets.append(labels.view(-1, 1))

    # Concatenate all batches
    all_probs = torch.cat(all_probs)
    all_targets = torch.cat(all_targets)

    # Compute BCE Loss
    # Clamp probabilities to avoid log(0)
    all_probs = torch.clamp(all_probs, 1e-7, 1 - 1e-7)
    loss = F.binary_cross_entropy(all_probs, all_targets)

    return loss.item()


def run_calibration():
    """
    Executes Phase 1: Calibration (Trajectory Discovery).
    Runs Stratified 5-Fold CV to determine optimal epochs and scheduler milestones.

    Returns:
        tuple: (optimal_epochs, milestones, final_lr)
    """
    print("Starting Phase 1: Calibration (Trajectory Discovery)...")
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 1. Load Data
    # We use the training set defined in metadata for CV
    full_dataset = get_dataset("train", load_cached_data=True)

    # Extract arrays for splitting
    X = full_dataset.images
    y = full_dataset.labels
    angles = full_dataset.angles

    # 2. Setup Cross-Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=Config.SEED)

    fold_best_epochs = []
    fold_milestones = []
    fold_final_lrs = []

    # 3. CV Loop
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n--- Fold {fold + 1}/5 ---")

        # Create Datasets
        train_ds = IcebergDataset(
            images=X[train_idx],
            angles=angles[train_idx],
            labels=y[train_idx],
            transform=get_training_transforms(),
        )
        val_ds = IcebergDataset(
            images=X[val_idx],
            angles=angles[val_idx],
            labels=y[val_idx],
            transform=get_validation_transforms(),
        )

        # Create Loaders
        train_loader = DataLoader(
            train_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=Config.PIN_MEMORY,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=Config.PIN_MEMORY,
        )

        # Model, Optimizer, Scheduler
        model = IcebergResNet18().to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=Config.OPTIMIZER_LR,
            weight_decay=Config.OPTIMIZER_WEIGHT_DECAY,
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=Config.PHASE1_FACTOR,
            patience=Config.PHASE1_PATIENCE,
            min_lr=Config.PHASE1_MIN_LR,
        )

        criterion = torch.nn.BCEWithLogitsLoss()

        # Training Loop Variables
        best_loss = float("inf")
        best_epoch = 0
        current_milestones = []
        last_lr = Config.OPTIMIZER_LR

        for epoch in range(1, Config.PHASE1_MAX_EPOCHS + 1):
            # Train
            train_loss = train_one_epoch(
                model, train_loader, optimizer, criterion, device, epoch
            )

            # Validate (TTA)
            val_loss = evaluate_tta(model, val_loader, device)
            print(f"Epoch {epoch} TTA Val Loss: {val_loss:.15f}")

            # Scheduler Step
            scheduler.step(val_loss)

            # Check for LR reduction (Milestone detection)
            current_lr = optimizer.param_groups[0]["lr"]
            if current_lr < last_lr:
                print(f"LR reduced from {last_lr} to {current_lr} at epoch {epoch}")
                current_milestones.append(epoch)
                last_lr = current_lr

            # Track Best
            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch = epoch

        print(f"Fold {fold + 1} Best Epoch: {best_epoch} (Loss: {best_loss:.6f})")
        fold_best_epochs.append(best_epoch)
        fold_milestones.append(current_milestones)
        fold_final_lrs.append(last_lr)

    # 4. Aggregate Results
    # Optimal Epoch: Average of best epochs
    avg_optimal_epoch = int(np.round(np.mean(fold_best_epochs)))

    # Milestones: Align milestones by index and average them
    # We assume most folds will have similar number of reductions.
    # We take the max length and average available values.
    max_reductions = max(len(m) for m in fold_milestones) if fold_milestones else 0
    avg_milestones = []

    for i in range(max_reductions):
        epoch_sum = 0
        count = 0
        for m_list in fold_milestones:
            if i < len(m_list):
                epoch_sum += m_list[i]
                count += 1
        if count > 0:
            avg_milestones.append(int(np.round(epoch_sum / count)))

    # Final LR: Geometric mean or simple mode. We use the config's min or the average final.
    # To be safe for Phase 2, we take the average final LR.
    avg_final_lr = float(np.mean(fold_final_lrs))

    print("\n--- Calibration Complete ---")
    print(f"Optimal Epoch: {avg_optimal_epoch}")
    print(f"Milestones: {avg_milestones}")
    print(f"Final LR: {avg_final_lr}")

    return avg_optimal_epoch, avg_milestones, avg_final_lr
