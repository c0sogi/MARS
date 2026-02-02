import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.model_selection import KFold
from scipy.stats import pearsonr

# Import library modules
from library.config import Config
from library.utils import seed_everything, calculate_log_loss
from library.data import load_dataset_data, IcebergDataset, get_transforms
from library.model import IcebergResNet
from library.engine import IcebergTrainer


def run_kfold_calibration():
    """
    Runs 5-Fold Cross-Validation to find the robust global optimal epoch and LR schedule.
    Cite Lesson 00040: Global Epoch Selection vs. Local Early Stopping.
    """
    print("\n--- Phase 1: 5-Fold Calibration ---")

    # Load full training data
    images, angles, labels, ids = load_dataset_data("train", load_cached_data=True)

    kf = KFold(n_splits=5, shuffle=True, random_state=Config.SEED)

    # Trackers
    fold_val_losses = np.full((5, Config.MAX_EPOCHS), np.inf)
    fold_lr_changes = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(images, labels)):
        print(f"Calibrating Fold {fold + 1}/5")

        # Split Data
        X_train, X_val = images[train_idx], images[val_idx]
        ang_train, ang_val = angles[train_idx], angles[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]

        # Datasets & Loaders
        train_ds = IcebergDataset(
            X_train, ang_train, y_train, transform=get_transforms("train")
        )
        val_ds = IcebergDataset(X_val, ang_val, y_val, transform=get_transforms("val"))

        train_loader = DataLoader(
            train_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Model & Optimization
        model = IcebergResNet(pretrained=True)
        optimizer = optim.AdamW(
            model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=Config.SCHEDULER_FACTOR,
            patience=Config.PATIENCE,
        )
        trainer = IcebergTrainer(model)

        # Fold Training Loop
        current_lr = Config.LR
        lr_changes = []

        for epoch in range(1, Config.MAX_EPOCHS + 1):
            trainer.train_one_epoch(train_loader, optimizer)
            _, val_logloss, _ = trainer.evaluate(val_loader)

            # Record Loss
            fold_val_losses[fold, epoch - 1] = val_logloss

            # Scheduler Step
            scheduler.step(val_logloss)

            # Track LR changes
            new_lr = optimizer.param_groups[0]["lr"]
            if new_lr < current_lr:
                lr_changes.append(epoch)
                current_lr = new_lr

        fold_lr_changes.append(lr_changes)

    # Global Epoch Selection
    # Average validation loss across folds for each epoch
    avg_val_losses = np.mean(fold_val_losses, axis=0)

    # Find epoch with minimum average loss
    # Add 1 because index 0 is epoch 1
    best_epoch = np.argmin(avg_val_losses) + 1
    best_loss = avg_val_losses[best_epoch - 1]

    # Calculate average LR milestones
    # Flatten list of lists and find common reduction points or average them
    # Simple heuristic: take the mean of the first reduction, second reduction, etc.
    max_reductions = max(len(changes) for changes in fold_lr_changes)
    milestones = []
    for i in range(max_reductions):
        epoch_sum = 0
        count = 0
        for changes in fold_lr_changes:
            if i < len(changes):
                epoch_sum += changes[i]
                count += 1
        if count > 0:
            milestones.append(int(epoch_sum / count))

    milestones = sorted(list(set(milestones)))

    print(
        f"Calibration Complete. Global Optimal Epoch: {best_epoch}, Avg Loss: {best_loss:.4f}"
    )
    print(f"Derived Milestones: {milestones}")

    return best_epoch, milestones


def train_ensemble(
    phase_name, train_loader, epochs, milestones, num_models, save_prefix
):
    """
    Trains an ensemble of models using the Replay + SWA strategy.
    """
    print(f"\n--- Phase {phase_name}: Training Ensemble ---")
    model_paths = []

    for i in range(num_models):
        print(f"Training Model {i+1}/{num_models}")
        seed_everything(Config.SEED + i)  # Diversity via seed

        model = IcebergResNet(pretrained=True)
        optimizer = optim.AdamW(
            model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
        )

        # Replay Scheduler: Replicate the drops found in calibration
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=Config.SCHEDULER_FACTOR
        )

        trainer = IcebergTrainer(model)

        # Fit with SWA
        # We pass val_loader=None because we are training on the full set for production
        swa_model = trainer.fit(
            train_loader=train_loader,
            val_loader=None,
            epochs=epochs + Config.SWA_EPOCHS,
            optimizer=optimizer,
            scheduler=scheduler,
            checkpoint_name=f"{save_prefix}_{i}.pth",
            use_swa=True,
            swa_start_epoch=epochs + 1,
        )

        # Save the final SWA model path
        final_path = os.path.join(Config.CHECKPOINT_DIR, f"{save_prefix}_{i}_swa.pth")
        model_paths.append(final_path)

    return model_paths


def predict_tta(model, loader, device):
    """
    Performs TTA (Original, HFlip, VFlip) inference using a single model.
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for batch in loader:
            imgs, angs = batch[0], batch[1]
            imgs = imgs.to(device)
            angs = angs.to(device)

            # 1. Original
            out_orig = model(imgs, angs)
            prob_orig = torch.sigmoid(out_orig)

            # 2. HFlip (dim 3 is width)
            imgs_h = torch.flip(imgs, dims=[3])
            out_h = model(imgs_h, angs)
            prob_h = torch.sigmoid(out_h)

            # 3. VFlip (dim 2 is height)
            imgs_v = torch.flip(imgs, dims=[2])
            out_v = model(imgs_v, angs)
            prob_v = torch.sigmoid(out_v)

            # Average
            avg_prob = (prob_orig + prob_h + prob_v) / 3.0
            all_preds.extend(avg_prob.cpu().numpy().flatten())

    return np.array(all_preds)


def evaluate_ensemble(model_paths, loader):
    """
    Evaluates the ensemble on a dataset using TTA.
    """
    print("\n--- Evaluating Ensemble ---")
    device = Config.DEVICE
    n_samples = len(loader.dataset)
    ensemble_preds = np.zeros((n_samples, len(model_paths)))

    targets = []
    # Extract targets from loader
    for batch in loader:
        if len(batch) >= 3:  # img, ang, label
            targets.extend(batch[2].numpy())
    targets = np.array(targets)

    for i, path in enumerate(model_paths):
        model = IcebergResNet(pretrained=False)
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)

        preds = predict_tta(model, loader, device)
        ensemble_preds[:, i] = preds

    # Average across ensemble
    final_preds = np.mean(ensemble_preds, axis=1)

    if len(targets) > 0:
        loss = calculate_log_loss(targets, final_preds)
        return loss, final_preds, targets
    else:
        return None, final_preds, None


def failure_analysis(preds, targets, loader):
    """
    Analyzes failure modes by correlating error with incidence angle.
    """
    print("\n--- Failure Analysis ---")

    angles = []
    # Extract angles from dataset
    ds = loader.dataset
    if hasattr(ds, "angles"):
        angles = ds.angles
    else:
        # Fallback
        for batch in loader:
            angles.extend(batch[1].numpy())

    errors = np.abs(targets - preds)

    # Correlation with Angle
    if len(angles) == len(errors):
        valid_mask = ~np.isnan(angles)
        if np.sum(valid_mask) > 0:
            corr, _ = pearsonr(np.array(angles)[valid_mask], errors[valid_mask])
            print(f"Correlation between Error and Incidence Angle: {corr:.10f}")
        else:
            print("No valid angles for correlation analysis.")
    else:
        print("Mismatch in lengths for failure analysis.")


def main():
    seed_everything(Config.SEED)

    # 1. Calibration (5-Fold CV)
    best_epoch, milestones = run_kfold_calibration()
    if best_epoch < 5:
        best_epoch = 5

    # 2. Train Ensemble (Full Data)
    # Cite Lesson 00053: Prioritize Baseline Calibration Over Semi-Supervised Expansion
    train_imgs, train_angs, train_lbls, train_ids = load_dataset_data("train")
    train_ds = IcebergDataset(
        train_imgs, train_angs, train_lbls, train_ids, transform=get_transforms("train")
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Train SWA Ensemble on full training set using calibrated schedule
    model_paths = train_ensemble(
        "Final", train_loader, best_epoch, milestones, Config.NUM_MODELS, "model"
    )

    # 3. Validation (Hold-out)
    val_imgs, val_angs, val_lbls, val_ids = load_dataset_data("val")
    val_ds = IcebergDataset(
        val_imgs, val_angs, val_lbls, val_ids, transform=get_transforms("val")
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loss, val_preds, val_targets = evaluate_ensemble(model_paths, val_loader)
    print(f"Final Validation Metric: {val_loss:.16f}")

    # 6. Failure Analysis
    failure_analysis(val_preds, val_targets, val_loader)

    # 7. Submission
    if val_loss < 0.16918645240183008:
        print("Validation score meets threshold. Generating submission...")
        test_imgs, test_angs, _, test_ids = load_dataset_data("test")
        test_ds = IcebergDataset(
            test_imgs, test_angs, ids=test_ids, transform=get_transforms("test")
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        _, test_preds, _ = evaluate_ensemble(model_paths, test_loader)

        sub_df = pd.DataFrame({"id": test_ids, "is_iceberg": test_preds})

        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print("Validation score did not meet threshold. Submission skipped.")


if __name__ == "__main__":
    main()
