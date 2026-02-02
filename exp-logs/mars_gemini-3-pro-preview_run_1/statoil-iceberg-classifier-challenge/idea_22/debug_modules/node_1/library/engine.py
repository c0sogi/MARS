import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel, SWALR
from library.config import Config
from library.utils import setup_logger, set_seed
from library.dataset import get_dataloaders
from library.model import AngleGatedResNet

logger = setup_logger()


class Engine:
    """
    Encapsulates training, validation, and the two-phase protocol logic.
    """

    @staticmethod
    def custom_update_bn(loader, model, device):
        """
        Custom update_bn that handles dictionary batches and multi-input models.
        Cite debug_lesson_8
        """
        momenta = {}
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.running_mean = torch.zeros_like(module.running_mean)
                module.running_var = torch.ones_like(module.running_var)
                momenta[module] = module.momentum

        if not momenta:
            return

        was_training = model.training
        model.train()
        for module in momenta.keys():
            module.momentum = None
            module.num_batches_tracked *= 0

        with torch.no_grad():
            for batch in loader:
                images = batch["image"].to(device)
                angles = batch["inc_angle"].to(device)
                model(images, angles)

        for bn_module in momenta.keys():
            bn_module.momentum = momenta[bn_module]
        model.train(was_training)

    @staticmethod
    def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
        """
        Trains the model for one epoch.
        """
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, batch in enumerate(loader):
            images = batch["image"].to(device)
            angles = batch["inc_angle"].to(device)
            labels = batch["label"].to(device).float().view(-1, 1)

            # Apply Label Smoothing to targets manually if needed,
            # or rely on criterion if it supports it.
            # Here we apply it to targets for BCEWithLogitsLoss compatibility.
            # smoothed_labels = labels * (1 - epsilon) + 0.5 * epsilon
            if Config.LABEL_SMOOTHING > 0:
                with torch.no_grad():
                    smoothed_labels = (
                        labels * (1.0 - Config.LABEL_SMOOTHING)
                        + 0.5 * Config.LABEL_SMOOTHING
                    )
            else:
                smoothed_labels = labels

            optimizer.zero_grad()

            outputs = model(images, angles)
            loss = criterion(outputs, smoothed_labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

            # Calculate accuracy using raw labels (0 or 1)
            preds = torch.sigmoid(outputs) > 0.5
            correct += (preds == (labels > 0.5)).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct / total

        return epoch_loss, epoch_acc

    @staticmethod
    def validate(model, loader, criterion, device):
        """
        Validates the model.
        """
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        # For validation, we use the raw labels without smoothing for loss calculation
        # to get a true estimate of performance, or we can use smoothed to match train.
        # Standard practice is usually raw labels for metric, but consistent loss for monitoring.
        # We will use raw labels for loss to strictly monitor convergence against ground truth.

        with torch.no_grad():
            for batch in loader:
                images = batch["image"].to(device)
                angles = batch["inc_angle"].to(device)
                labels = batch["label"].to(device).float().view(-1, 1)

                outputs = model(images, angles)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * images.size(0)

                preds = torch.sigmoid(outputs) > 0.5
                correct += (preds == (labels > 0.5)).sum().item()
                total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct / total

        return epoch_loss, epoch_acc

    @staticmethod
    def find_optimal_epoch():
        """
        Phase 1: Calibration.
        Runs Stratified 5-Fold CV to find the global optimal convergence epoch.
        """
        logger.info("Starting Phase 1: Calibration (Global Epoch Selection)")

        device = torch.device(Config.DEVICE)
        fold_val_losses = []  # Stores list of lists: [fold_idx][epoch_idx]

        for fold in range(Config.N_FOLDS):
            logger.info(f"--- Calibration Fold {fold + 1}/{Config.N_FOLDS} ---")

            # Get Dataloaders
            train_loader, val_loader = get_dataloaders(
                fold=fold, phase="calibration", load_cache=True
            )

            # Initialize Model
            model = AngleGatedResNet().to(device)

            # Optimizer & Scheduler
            optimizer = optim.AdamW(
                model.parameters(),
                lr=Config.LEARNING_RATE,
                weight_decay=Config.WEIGHT_DECAY,
            )
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=Config.FACTOR,
                patience=Config.PATIENCE,
                min_lr=Config.MIN_LR,
            )

            # Criterion (BCEWithLogitsLoss)
            # Note: We handle label smoothing in the train loop manually.
            criterion = nn.BCEWithLogitsLoss()

            fold_losses = []

            for epoch in range(Config.MAX_EPOCHS):
                train_loss, train_acc = Engine.train_one_epoch(
                    model, train_loader, optimizer, criterion, device, epoch
                )
                val_loss, val_acc = Engine.validate(
                    model, val_loader, criterion, device
                )

                scheduler.step(val_loss)
                fold_losses.append(val_loss)

                # Log occasionally
                if (epoch + 1) % 10 == 0 or epoch < 5:
                    logger.info(
                        f"Fold {fold+1} Epoch {epoch+1}/{Config.MAX_EPOCHS} - "
                        f"Train Loss: {train_loss:.6f}, Train Acc: {train_acc:.6f}, "
                        f"Val Loss: {val_loss:.6f}, Val Acc: {val_acc:.6f}"
                    )

            fold_val_losses.append(fold_losses)

            # Cleanup to save memory
            del model, optimizer, scheduler, train_loader, val_loader
            torch.cuda.empty_cache()

        # Aggregate results
        # Convert to numpy array: (n_folds, n_epochs)
        losses_arr = np.array(fold_val_losses)
        mean_val_losses = np.mean(losses_arr, axis=0)

        # Find epoch with minimum mean validation loss
        optimal_idx = np.argmin(mean_val_losses)
        optimal_epoch = optimal_idx + 1  # 1-based index
        min_loss = mean_val_losses[optimal_idx]

        logger.info(
            f"Phase 1 Complete. Optimal Convergence Epoch: {optimal_epoch} (Loss: {min_loss:.6f})"
        )
        return optimal_epoch

    @staticmethod
    def train_full_fit_swa(optimal_epoch, num_models=5):
        """
        Phase 2: Production.
        Trains independent models on the full dataset for optimal_epoch,
        followed by SWA.
        """
        logger.info(
            f"Starting Phase 2: Production (Full-Fit SWA) with {num_models} models"
        )
        logger.info(
            f"Target Epochs: {optimal_epoch} (Standard) + {Config.SWA_EPOCHS} (SWA)"
        )

        device = torch.device(Config.DEVICE)
        checkpoints_dir = os.path.join(Config.WORKING_DIR, "checkpoints")
        os.makedirs(checkpoints_dir, exist_ok=True)

        # Get Full Dataloader (100% data)
        train_loader = get_dataloaders(phase="production", load_cache=True)

        criterion = nn.BCEWithLogitsLoss()

        for i in range(num_models):
            logger.info(f"--- Training Production Model {i + 1}/{num_models} ---")

            # Re-seed for independence if needed, but usually we want diversity.
            # However, prompt says "Train 5 independent models".
            # We rely on random initialization diversity + shuffle.
            # Setting a specific seed per model ensures reproducibility of the ensemble.
            set_seed(Config.SEED + i)

            model = AngleGatedResNet().to(device)
            optimizer = optim.AdamW(
                model.parameters(),
                lr=Config.LEARNING_RATE,
                weight_decay=Config.WEIGHT_DECAY,
            )

            # --- Stage 1: Standard Training ---
            # We train for exactly optimal_epoch.
            # We do not use ReduceLROnPlateau here as we don't have a validation set to trigger it reliably.
            # We assume the trajectory learned in Phase 1 holds.

            for epoch in range(optimal_epoch):
                loss, acc = Engine.train_one_epoch(
                    model, train_loader, optimizer, criterion, device, epoch
                )
                if (epoch + 1) % 10 == 0 or epoch == optimal_epoch - 1:
                    logger.info(
                        f"Model {i+1} Std Epoch {epoch+1}/{optimal_epoch} - Loss: {loss:.6f}, Acc: {acc:.6f}"
                    )

            # --- Stage 2: SWA Training ---
            logger.info(f"Model {i+1} entering SWA phase...")

            swa_model = AveragedModel(model)
            swa_scheduler = SWALR(optimizer, swa_lr=Config.SWA_LR)

            for swa_epoch in range(Config.SWA_EPOCHS):
                loss, acc = Engine.train_one_epoch(
                    model, train_loader, optimizer, criterion, device, swa_epoch
                )

                swa_model.update_parameters(model)
                swa_scheduler.step()

                logger.info(
                    f"Model {i+1} SWA Epoch {swa_epoch+1}/{Config.SWA_EPOCHS} - Loss: {loss:.6f}, Acc: {acc:.6f}"
                )

            # Update BN Statistics
            logger.info(f"Model {i+1} updating BN statistics...")
            Engine.custom_update_bn(train_loader, swa_model, device=device)

            # Save Model
            save_path = os.path.join(checkpoints_dir, f"swa_model_{i}.pth")
            torch.save(swa_model.state_dict(), save_path)
            logger.info(f"Saved Model {i+1} to {save_path}")

            # Cleanup
            del model, swa_model, optimizer, swa_scheduler
            torch.cuda.empty_cache()

        logger.info("Phase 2 Complete.")
