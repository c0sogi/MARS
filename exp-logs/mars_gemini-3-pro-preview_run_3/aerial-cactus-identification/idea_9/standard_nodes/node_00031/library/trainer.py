import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from library.config import Config
from library.utils import AverageMeter, calculate_auc, save_checkpoint, setup_logger
from library.data_loader import mixup_data, mixup_criterion


class CactusTrainer:
    """
    Trainer class for the Cactus Identification task.
    Encapsulates training, validation, and inference logic with TTA.
    """

    def __init__(
        self,
        model,
        device=None,
        optimizer=None,
        criterion=None,
        scheduler=None,
        logger=None,
    ):
        """
        Args:
            model: PyTorch model to train.
            device: torch.device.
            optimizer: Optimizer instance. If None, defaults to AdamW based on Config.
            criterion: Loss function. If None, defaults to BCEWithLogitsLoss.
            scheduler: Learning rate scheduler (optional).
            logger: Logger instance.
        """
        self.model = model
        self.device = device if device else torch.device(Config.DEVICE)
        self.model.to(self.device)

        # Default Optimizer: AdamW
        if optimizer is None:
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=Config.LEARNING_RATE,
                weight_decay=Config.WEIGHT_DECAY,
            )
        else:
            self.optimizer = optimizer

        # Default Criterion: BCEWithLogitsLoss
        if criterion is None:
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = criterion

        self.scheduler = scheduler
        self.logger = logger if logger else setup_logger("CactusTrainer")

    def train_one_epoch(self, train_loader, epoch):
        """
        Trains the model for one epoch using Mixup augmentation.
        """
        self.model.train()
        losses = AverageMeter()

        for i, (images, labels) in enumerate(train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Apply Mixup Augmentation
            if Config.USE_MIXUP:
                images, targets_a, targets_b, lam = mixup_data(
                    images,
                    labels,
                    Config.MIXUP_ALPHA,
                    use_cuda=(self.device.type == "cuda"),
                )

                # Forward pass
                outputs = self.model(images).squeeze(1)
                loss = mixup_criterion(
                    self.criterion, outputs, targets_a, targets_b, lam
                )
            else:
                outputs = self.model(images).squeeze(1)
                loss = self.criterion(outputs, labels)

            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses.update(loss.item(), images.size(0))

        return losses.avg

    def validate(self, val_loader):
        """
        Evaluates the model on the validation set.
        Returns loss, auc, predictions, and targets.
        """
        self.model.eval()
        losses = AverageMeter()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images).squeeze(1)
                loss = self.criterion(outputs, labels)

                losses.update(loss.item(), images.size(0))

                # Apply sigmoid to get probabilities
                preds = torch.sigmoid(outputs)

                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())

        auc = calculate_auc(all_targets, all_preds)
        return losses.avg, auc, np.array(all_preds), np.array(all_targets)

    def fit(
        self,
        train_loader,
        val_loader,
        epochs=Config.EPOCHS,
        patience=Config.EARLY_STOPPING_PATIENCE,
        save_path=None,
    ):
        """
        Main training loop with Early Stopping.
        """
        best_auc = 0.0
        patience_counter = 0

        if save_path is None:
            save_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

        self.logger.info(
            f"Starting training for {epochs} epochs with patience {patience}..."
        )

        for epoch in range(epochs):
            start_time = time.time()

            train_loss = self.train_one_epoch(train_loader, epoch)
            val_loss, val_auc, _, _ = self.validate(val_loader)

            elapsed = time.time() - start_time

            # Print full precision metrics
            self.logger.info(
                f"Epoch {epoch+1}/{epochs} - "
                f"Time: {elapsed:.2f}s - "
                f"Train Loss: {train_loss} - "
                f"Val Loss: {val_loss} - "
                f"Val AUC: {val_auc}"
            )

            # Scheduler Step
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_auc)
                else:
                    self.scheduler.step()

            # Early Stopping and Checkpointing
            if val_auc > best_auc:
                best_auc = val_auc
                patience_counter = 0
                save_checkpoint(self.model, save_path)
                self.logger.info(
                    f"New best AUC found: {best_auc}. Model saved to {save_path}."
                )
            else:
                patience_counter += 1
                self.logger.info(
                    f"No improvement. Patience: {patience_counter}/{patience}"
                )
                if patience_counter >= patience:
                    self.logger.info("Early stopping triggered.")
                    break

        return best_auc

    def predict_with_tta(self, test_loader):
        """
        Generates predictions using Test-Time Augmentation (Original + HFlip + VFlip).
        Returns a numpy array of probabilities.
        """
        self.model.eval()
        all_preds = []

        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(self.device)

                # 1. Original
                out_orig = self.model(images).squeeze(1)
                preds_orig = torch.sigmoid(out_orig)

                # 2. Horizontal Flip (dim 3 is width)
                images_h = torch.flip(images, dims=[3])
                out_h = self.model(images_h).squeeze(1)
                preds_h = torch.sigmoid(out_h)

                # 3. Vertical Flip (dim 2 is height)
                images_v = torch.flip(images, dims=[2])
                out_v = self.model(images_v).squeeze(1)
                preds_v = torch.sigmoid(out_v)

                # Average predictions
                preds_avg = (preds_orig + preds_h + preds_v) / 3.0

                all_preds.extend(preds_avg.cpu().numpy())

        return np.array(all_preds)

    def generate_submission(
        self, test_loader, test_ids, output_path=Config.SUBMISSION_PATH
    ):
        """
        Generates predictions for the test set and saves them to a CSV file.

        Args:
            test_loader: DataLoader for the test set.
            test_ids: Array of test image IDs corresponding to the loader order.
            output_path: Path to save the submission CSV.
        """
        self.logger.info("Generating predictions with TTA...")
        preds = self.predict_with_tta(test_loader)

        if len(preds) != len(test_ids):
            raise ValueError(
                f"Shape mismatch: {len(preds)} predictions vs {len(test_ids)} IDs."
            )

        df = pd.DataFrame({"id": test_ids, "has_cactus": preds})

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        self.logger.info(f"Submission saved to {output_path}")
