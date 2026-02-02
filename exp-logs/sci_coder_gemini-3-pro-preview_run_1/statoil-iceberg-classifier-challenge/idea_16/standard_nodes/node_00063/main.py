import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.model_selection import StratifiedKFold
from scipy.stats import pearsonr

# Import library modules
from library.config import Config
from library.utils import seed_everything, calculate_log_loss
from library.data import load_dataset_data, IcebergDataset, get_transforms
from library.model import IcebergResNet
from library.engine import IcebergTrainer


def run_calibration():
    """
    Runs 5-Fold Cross-Validation to determine the global optimal epoch count.
    Cite solution_lesson_node_00040: Global Epoch Selection vs Local Early Stopping.
    Cite solution_lesson_node_00049: Calibrate Duration via K-Fold CV.
    """
    print("\n--- Phase 1: Calibration (5-Fold CV) ---")

    # Load full training data
    images, angles, labels, ids = load_dataset_data("train", load_cached_data=True)

    # Configuration
    n_folds = 5
    max_epochs = 50
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=Config.SEED)

    # Store validation losses: shape (n_folds, max_epochs)
    # Initialize with infinity
    all_val_losses = np.full((n_folds, max_epochs), np.inf)

    # Store LR decay epochs
    all_lr_changes = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(images, labels)):
        print(f"Calibrating Fold {fold + 1}/{n_folds}...")

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

        # Model & Optimizer
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
        current_lr = Config.LR

        # Training Loop
        for epoch in range(1, max_epochs + 1):
            trainer.train_one_epoch(train_loader, optimizer)
            _, val_logloss, _ = trainer.evaluate(val_loader)

            # Record Loss (0-indexed)
            all_val_losses[fold, epoch - 1] = val_logloss

            # Check LR Decay
            new_lr = optimizer.param_groups[0]["lr"]
            if new_lr < current_lr:
                all_lr_changes.append(epoch)
                current_lr = new_lr

            scheduler.step(val_logloss)

            # Simple early stopping to save time if diverged (optional)
            if epoch > 20 and val_logloss > 0.5:
                break

    # Calculate Average Validation Curve
    # Mask out inf values (in case of early break, though we run fixed epochs mostly)
    # Here we assume we ran enough epochs.
    avg_val_losses = np.mean(all_val_losses, axis=0)

    # Find Global Optimal Epoch (argmin)
    best_epoch_idx = np.argmin(avg_val_losses)
    best_epoch = best_epoch_idx + 1
    min_loss = avg_val_losses[best_epoch_idx]

    print(
        f"Calibration Complete. Global Optimal Epoch: {best_epoch}, Avg Loss: {min_loss:.4f}"
    )

    # Determine Milestones (Median of decay epochs)
    if all_lr_changes:
        median_decay = int(np.median(all_lr_changes))
        milestones = [median_decay]
        # If there are multiple decays (e.g. 1e-4 -> 1e-5 -> 1e-6), we could cluster them.
        # But usually one drop is enough for this dataset size.
        print(f"Derived LR Milestones: {milestones}")
    else:
        milestones = []
        print("No LR decay detected during calibration.")

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

    # 1. Calibration
    best_epoch, milestones = run_calibration()
    if best_epoch < 5:
        best_epoch = 5

    # 2. SWA Ensemble Training
    # Cite solution_lesson_node_00049: SWA Ensembles with Global Epoch Selection
    # Cite solution_lesson_node_00053: Prioritize Baseline Calibration Over Semi-Supervised Expansion

    # Load full train data
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

    # Train Ensemble on Full Training Set
    model_paths = train_ensemble(
        "SWA_Ensemble",
        train_loader,
        best_epoch,
        milestones,
        Config.NUM_MODELS,
        "swa_model",
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

    # 4. Failure Analysis
    failure_analysis(val_preds, val_targets, val_loader)

    # 5. Submission
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
