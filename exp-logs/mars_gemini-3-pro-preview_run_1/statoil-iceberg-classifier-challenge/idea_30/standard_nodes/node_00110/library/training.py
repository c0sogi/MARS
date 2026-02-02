import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.swa_utils import AveragedModel, SWALR
from sklearn.model_selection import StratifiedKFold
import math

from library.utils import set_seed
from library.data import IcebergDataset, get_transforms, process_and_cache_data
from library.model import IcebergResNet18
from library.inference import validate_model


def custom_update_bn(loader, model, device=None):
    """
    Custom update_bn that handles multi-input models (images, angles).
    Cite debug_lesson_8: Implement Custom update_bn for Multi-Input Models in SWA.
    """
    momenta = {}
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.running_mean = torch.zeros_like(module.running_mean)
            module.running_var = torch.ones_like(module.running_var)
            momenta[module] = module.momentum
            module.momentum = None
            module.num_batches_tracked *= 0

    model.train()
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                images, angles, _ = batch
            else:
                images, angles = batch

            if device is not None:
                images = images.to(device)
                angles = angles.to(device)

            model(images, angles)

    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.momentum = momenta[module]


class Trainer:
    def __init__(
        self,
        input_dir="./input",
        metadata_dir="./metadata",
        working_dir="./working/idea_30",
    ):
        self.input_dir = input_dir
        self.metadata_dir = metadata_dir
        self.working_dir = working_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Ensure working directory exists
        os.makedirs(self.working_dir, exist_ok=True)
        self.checkpoints_dir = os.path.join(self.working_dir, "checkpoints")
        os.makedirs(self.checkpoints_dir, exist_ok=True)

    def _load_full_labeled_data(self, debug=False):
        """
        Loads and merges train and validation data to create the full labeled dataset.
        """
        train_meta = os.path.join(self.metadata_dir, "train_metadata.csv")
        val_meta = os.path.join(self.metadata_dir, "val_metadata.csv")
        train_json = os.path.join(self.input_dir, "train.json")

        cache_train = os.path.join(self.working_dir, "train_processed.npz")
        cache_val = os.path.join(self.working_dir, "val_processed.npz")

        # Load both splits
        X_train, a_train, y_train = process_and_cache_data(
            train_meta, train_json, cache_train
        )
        X_val, a_val, y_val = process_and_cache_data(val_meta, train_json, cache_val)

        # Merge
        X = np.concatenate([X_train, X_val], axis=0)
        a = np.concatenate([a_train, a_val], axis=0)
        y = np.concatenate([y_train, y_val], axis=0)

        # Impute and Normalize Angles (Global Statistics on Full Set)
        # Note: In a strict pipeline, we might separate this, but for the task description,
        # we treat the full labeled set as available for calibration/training.
        angle_mean = np.nanmean(a)
        angle_std = np.nanstd(a)

        a = np.where(np.isnan(a), angle_mean, a)
        a = (a - angle_mean) / (angle_std + 1e-8)

        if debug:
            print("DEBUG MODE: Truncating data to 100 samples.")
            return X[:100], a[:100], y[:100]

        return X, a, y

    def train_one_epoch(
        self, model, loader, optimizer, criterion, device, label_smoothing=0.05
    ):
        model.train()
        running_loss = 0.0

        for images, angles, labels in loader:
            images = images.to(device)
            angles = angles.to(device)
            labels = labels.to(device).unsqueeze(1)

            # Apply Label Smoothing manually
            # y_smooth = y * (1 - eps) + 0.5 * eps
            targets = labels * (1.0 - label_smoothing) + 0.5 * label_smoothing

            optimizer.zero_grad()
            outputs = model(images, angles)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        return running_loss / len(loader.dataset)

    def run_calibration_phase(self, batch_size=32, max_epochs=50, debug=False):
        """
        Phase 1: Calibration.
        Runs Stratified 5-Fold CV to find the optimal total gradient steps.
        """
        print("Starting Phase 1: Calibration (Step Volume Discovery)...")
        set_seed(42)

        X, a, y = self._load_full_labeled_data(debug)

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        fold_best_steps = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            print(f"\nCalibration Fold {fold + 1}/5")

            # Prepare Data
            X_fold_train, X_fold_val = X[train_idx], X[val_idx]
            a_fold_train, a_fold_val = a[train_idx], a[val_idx]
            y_fold_train, y_fold_val = y[train_idx], y[val_idx]

            train_ds = IcebergDataset(
                X_fold_train,
                a_fold_train,
                y_fold_train,
                transform=get_transforms("train"),
            )
            val_ds = IcebergDataset(
                X_fold_val, a_fold_val, y_fold_val, transform=get_transforms("val")
            )

            train_loader = DataLoader(
                train_ds,
                batch_size=batch_size,
                shuffle=True,
                num_workers=2,
                pin_memory=True,
                drop_last=True,
            )
            val_loader = DataLoader(
                val_ds,
                batch_size=batch_size,
                shuffle=False,
                num_workers=2,
                pin_memory=True,
            )

            # Model, Optimizer, Scheduler
            model = IcebergResNet18(dropout_rate=0.5).to(self.device)
            optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
            # Cite Lesson 00070: Increase patience for noisy datasets
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=12
            )
            criterion = nn.BCEWithLogitsLoss()

            best_loss = float("inf")
            best_epoch = 0

            # Early stopping counter
            patience_counter = 0
            early_stopping_limit = 25

            for epoch in range(max_epochs):
                train_loss = self.train_one_epoch(
                    model,
                    train_loader,
                    optimizer,
                    criterion,
                    self.device,
                    label_smoothing=0.05,
                )

                # Validation with TTA
                val_loss, _, _ = validate_model(model, val_loader, self.device)

                scheduler.step(val_loss)

                if val_loss < best_loss:
                    best_loss = val_loss
                    best_epoch = epoch + 1  # 1-based
                    patience_counter = 0
                else:
                    patience_counter += 1

                print(
                    f"Epoch {epoch+1}/{max_epochs} - Train Loss: {train_loss:.6f} - Val Loss (TTA): {val_loss:.10f}"
                )

                if patience_counter >= early_stopping_limit:
                    print("Early stopping triggered.")
                    break

            # Calculate Total Gradient Steps for this fold
            # Steps = Epochs * Steps_Per_Epoch
            steps_per_epoch = len(train_loader)
            optimal_steps = best_epoch * steps_per_epoch
            fold_best_steps.append(optimal_steps)
            print(f"Fold {fold+1} Optimal Epoch: {best_epoch}, Steps: {optimal_steps}")

        avg_steps = int(np.mean(fold_best_steps))
        print(f"\nCalibration Complete. Average Optimal Gradient Steps: {avg_steps}")
        return avg_steps

    def run_production_phase(
        self, optimal_steps, batch_size=32, num_models=5, debug=False
    ):
        """
        Phase 2: Production.
        Trains 5 independent models on the full dataset using Isovariant Mapping and SWA.
        """
        print("\nStarting Phase 2: Production (Isovariant Full-Fit)...")
        set_seed(42)

        X, a, y = self._load_full_labeled_data(debug)

        # Full Dataset Loader
        train_ds = IcebergDataset(X, a, y, transform=get_transforms("train"))
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=True,
        )

        # Calculate Production Epochs (Isovariant Mapping)
        steps_per_epoch = len(train_loader)
        production_epochs = math.ceil(optimal_steps / steps_per_epoch)
        print(
            f"Isovariant Mapping: {optimal_steps} steps -> {production_epochs} epochs (Batch Size {batch_size}, Dataset Size {len(train_ds)})"
        )

        swa_epochs = 12
        total_epochs = production_epochs + swa_epochs

        saved_models = []

        for i in range(num_models):
            print(f"\nTraining Production Model {i+1}/{num_models}")
            # Reseed for independence
            set_seed(42 + i)

            model = IcebergResNet18(dropout_rate=0.5).to(self.device)
            optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)

            # Scheduler: Cosine Annealing for main phase
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=production_epochs, eta_min=1e-5
            )

            criterion = nn.BCEWithLogitsLoss()

            # SWA Setup
            swa_model = AveragedModel(model)
            swa_scheduler = SWALR(optimizer, swa_lr=1e-5)

            for epoch in range(total_epochs):
                current_epoch = epoch + 1

                train_loss = self.train_one_epoch(
                    model,
                    train_loader,
                    optimizer,
                    criterion,
                    self.device,
                    label_smoothing=0.05,
                )

                if current_epoch <= production_epochs:
                    scheduler.step()
                    phase = "Main"
                else:
                    swa_model.update_parameters(model)
                    swa_scheduler.step()
                    phase = "SWA"

                print(
                    f"Model {i+1} Epoch {current_epoch}/{total_epochs} ({phase}) - Train Loss: {train_loss:.6f}"
                )

            # Update BN statistics for SWA model
            print("Updating SWA Batch Normalization statistics...")
            custom_update_bn(train_loader, swa_model, device=self.device)

            # Save SWA model
            save_path = os.path.join(self.checkpoints_dir, f"swa_model_{i}.pth")
            torch.save(swa_model.state_dict(), save_path)
            saved_models.append(save_path)
            print(f"Saved SWA model to {save_path}")

        return saved_models
