import os
import torch
import torch.nn as nn
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
import numpy as np
from tqdm import tqdm

from library.config import Config
from library.utils import calculate_roc_auc


class Trainer:
    """
    Trainer class for managing the training lifecycle of the BirdResNet model,
    including standard training, validation, and SWA (Stochastic Weight Averaging).
    """

    def __init__(self, model, device, criterion, optimizer, scheduler=None):
        """
        Args:
            model (nn.Module): The PyTorch model to train.
            device (str): Device to run training on ('cuda' or 'cpu').
            criterion (nn.Module): Loss function.
            optimizer (torch.optim.Optimizer): Optimizer.
            scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Learning rate scheduler for standard phase.
        """
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.best_auc = 0.0

    def train_epoch(self, loader):
        """
        Runs one epoch of training.

        Args:
            loader (DataLoader): Training data loader.

        Returns:
            float: Average loss for the epoch.
        """
        self.model.train()
        running_loss = 0.0
        count = 0

        for batch_idx, (images, labels, _) in enumerate(loader):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            # Forward pass
            # Use mixed precision if available (optional, but good practice)
            # Here we stick to standard float32 for stability unless configured otherwise
            outputs = self.model(images)

            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            count += images.size(0)

        return running_loss / count if count > 0 else 0.0

    def validate(self, loader, model_to_validate=None):
        """
        Runs validation.

        Args:
            loader (DataLoader): Validation data loader.
            model_to_validate (nn.Module, optional): Specific model to validate.
                                                     Defaults to self.model.

        Returns:
            tuple: (average_loss, roc_auc_score)
        """
        model = model_to_validate if model_to_validate is not None else self.model
        model.eval()

        running_loss = 0.0
        count = 0

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels, _ in loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                outputs = model(images)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * images.size(0)
                count += images.size(0)

                # Apply sigmoid for probabilities
                probs = torch.sigmoid(outputs)

                all_preds.append(probs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        avg_loss = running_loss / count if count > 0 else 0.0

        if len(all_preds) > 0:
            all_preds = np.concatenate(all_preds, axis=0)
            all_labels = np.concatenate(all_labels, axis=0)
            auc_score = calculate_roc_auc(all_labels, all_preds)
        else:
            auc_score = 0.5

        return avg_loss, auc_score

    def fit_swa(
        self, train_loader, val_loader, total_epochs, swa_start_epoch, swa_lr, save_path
    ):
        """
        Runs the full training pipeline with SWA transition.

        Args:
            train_loader (DataLoader): Training data.
            val_loader (DataLoader): Validation data.
            total_epochs (int): Total number of epochs to train.
            swa_start_epoch (int): Epoch number to start SWA (1-based index).
            swa_lr (float): Learning rate for SWA phase.
            save_path (str): Path to save the best/final model.

        Returns:
            nn.Module: The final trained model (SWA model).
        """
        print(f"Starting training for {total_epochs} epochs.")
        print(f"SWA will start at epoch {swa_start_epoch} with LR {swa_lr}.")

        # Initialize SWA Model
        swa_model = AveragedModel(self.model).to(self.device)
        swa_scheduler = SWALR(self.optimizer, swa_lr=swa_lr)

        # Track best base model performance
        best_base_auc = 0.0
        best_base_path = save_path.replace(".pth", "_base_best.pth")

        for epoch in range(1, total_epochs + 1):
            # --- Training Step ---
            train_loss = self.train_epoch(train_loader)

            # --- SWA Logic ---
            if epoch >= swa_start_epoch:
                # SWA Phase
                swa_model.update_parameters(self.model)
                swa_scheduler.step()
                lr = swa_scheduler.get_last_lr()[0]
                phase = "SWA"
            else:
                # Standard Phase
                if self.scheduler:
                    self.scheduler.step()
                    lr = self.scheduler.get_last_lr()[0]
                else:
                    lr = self.optimizer.param_groups[0]["lr"]
                phase = "STD"

            # --- Validation Step (Base Model) ---
            # We validate the base model every epoch to monitor progress
            val_loss, val_auc = self.validate(val_loader, model_to_validate=self.model)

            print(
                f"Epoch {epoch}/{total_epochs} [{phase}] | LR: {lr:.6f} | "
                f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val AUC: {val_auc:.10f}"
            )

            # Save best base model (only relevant before SWA or if SWA fails)
            if val_auc > best_base_auc:
                best_base_auc = val_auc
                torch.save(self.model.state_dict(), best_base_path)

        print("Training complete. Finalizing SWA model...")

        # Update BatchNorm statistics for the SWA model
        # This is crucial as SWA averages weights but not BN stats
        update_bn(train_loader, swa_model, device=self.device)

        # Validate SWA Model
        swa_val_loss, swa_val_auc = self.validate(
            val_loader, model_to_validate=swa_model
        )
        print(
            f"SWA Final Results | Val Loss: {swa_val_loss:.6f} | Val AUC: {swa_val_auc:.10f}"
        )

        # Save SWA Model
        torch.save(swa_model.state_dict(), save_path)
        print(f"SWA model saved to {save_path}")

        return swa_model
