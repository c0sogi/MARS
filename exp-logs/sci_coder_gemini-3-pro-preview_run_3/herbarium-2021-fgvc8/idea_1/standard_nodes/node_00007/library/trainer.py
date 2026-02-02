import time
import torch
import torch.nn as nn
import numpy as np
from torch.optim.swa_utils import AveragedModel, SWALR
from library.config import Config
from library.utils import calculate_macro_f1, save_checkpoint


class Trainer:
    """
    Manages the training and validation process for the Herbarium Plant Species Classification model.
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        device=Config.DEVICE,
        patience=Config.PATIENCE,
    ):
        """
        Args:
            model (nn.Module): The neural network model.
            train_loader (DataLoader): DataLoader for training data.
            val_loader (DataLoader): DataLoader for validation data.
            optimizer (Optimizer): The optimizer.
            scheduler (LRScheduler): The learning rate scheduler.
            device (str): Device to run training on ('cuda' or 'cpu').
            patience (int): Number of epochs to wait for improvement before early stopping.
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.patience = patience

        self.criterion = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)
        # Initialize GradScaler for Automatic Mixed Precision
        self.scaler = torch.amp.GradScaler("cuda") if self.device == "cuda" else None

    def train_one_epoch(self, epoch_index):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0
        total_samples = 0

        start_time = time.time()

        for i, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            # Mixed Precision Training
            if self.scaler:
                with torch.amp.autocast("cuda"):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

        avg_loss = running_loss / total_samples
        duration = time.time() - start_time

        print(
            f"Epoch [{epoch_index+1}] Training Loss: {avg_loss} (Time: {duration:.2f}s)"
        )
        return avg_loss

    def validate(self):
        """
        Runs validation on the validation set and calculates metrics.
        """
        self.model.eval()
        running_loss = 0.0
        total_samples = 0

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                # Use mixed precision for inference as well to speed it up
                if self.device == "cuda":
                    with torch.amp.autocast("cuda"):
                        outputs = self.model(images)
                        loss = self.criterion(outputs, labels)
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                running_loss += loss.item() * images.size(0)
                total_samples += images.size(0)

                # Get predictions
                _, preds = torch.max(outputs, 1)

                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())

        avg_loss = running_loss / total_samples

        # Concatenate all batches
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        # Calculate Macro F1
        macro_f1 = calculate_macro_f1(all_labels, all_preds)

        print(f"Validation Loss: {avg_loss}")
        print(f"Validation Macro F1: {macro_f1}")

        return avg_loss, macro_f1

    def _update_bn(self, loader, model, num_batches=2000):
        """
        Updates Batch Normalization running mean and var for SWA model.
        Uses AMP to save memory and runs on a subset of data for speed.
        """
        print(f"Updating SWA Batch Norm statistics on {num_batches} batches...")
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
            for i, (input, _) in enumerate(loader):
                if i >= num_batches:
                    break
                input = input.to(self.device)
                if self.scaler:
                    with torch.amp.autocast("cuda"):
                        model(input)
                else:
                    model(input)

        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.momentum = momenta[module]

    def fit(self, num_epochs=Config.NUM_EPOCHS):
        """
        Main training loop with early stopping and SWA.
        """
        best_f1 = 0.0
        patience_counter = 0

        # SWA Initialization
        swa_model = None
        swa_scheduler = None
        if Config.USE_SWA:
            print("SWA Enabled.")
            swa_model = AveragedModel(self.model)
            swa_scheduler = SWALR(self.optimizer, swa_lr=Config.SWA_LR)

        print(f"Starting training for {num_epochs} epochs on {self.device}...")

        for epoch in range(num_epochs):
            # Train
            self.train_one_epoch(epoch)

            # SWA Update
            if Config.USE_SWA and epoch >= Config.SWA_START_EPOCH:
                swa_model.update_parameters(self.model)
                swa_scheduler.step()
                print(f"SWA update performed (Epoch {epoch+1})")
            else:
                # Step the regular scheduler
                if self.scheduler:
                    self.scheduler.step()

            # Validate (Regular model)
            val_loss, val_f1 = self.validate()

            # Checkpoint and Early Stopping (Regular model)
            if val_f1 > best_f1:
                print(f"New best model found! F1 improved from {best_f1} to {val_f1}")
                best_f1 = val_f1
                patience_counter = 0

                # Save best model
                state = {
                    "epoch": epoch + 1,
                    "state_dict": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "best_f1": best_f1,
                }
                save_checkpoint(state, is_best=True, filename=Config.MODEL_SAVE_PATH)
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{self.patience}")

            # Note: We don't stop early if SWA is active and we haven't finished SWA epochs
            if patience_counter >= self.patience:
                if Config.USE_SWA and epoch >= Config.SWA_START_EPOCH:
                    print("Patience exceeded, but continuing for SWA...")
                else:
                    print("Early stopping triggered.")
                    break

        # End of training - SWA Finalization
        if Config.USE_SWA and swa_model is not None:
            self._update_bn(self.train_loader, swa_model)

            # Validate SWA model
            print("Validating SWA model...")
            # Swap model for validation
            original_model = self.model
            self.model = swa_model

            swa_loss, swa_f1 = self.validate()
            print(f"SWA Validation F1: {swa_f1}")

            if swa_f1 > best_f1:
                print(
                    "SWA model outperformed best regular model. Saving SWA model as best."
                )
                best_f1 = swa_f1
                # Save SWA model weights (extract from AveragedModel)
                state = {
                    "epoch": num_epochs,
                    "state_dict": swa_model.module.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "best_f1": swa_f1,
                }
                save_checkpoint(state, is_best=True, filename=Config.MODEL_SAVE_PATH)

            # Restore original model
            self.model = original_model

        print(f"Training complete. Best Validation F1: {best_f1}")
