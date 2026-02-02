import os
import torch
import numpy as np
import pandas as pd
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

from library.config import Config
from library.utils import (
    seed_everything,
    get_logger,
    AverageMeter,
    calculate_weighted_log_loss,
)
from library.data import get_dataloaders
from library.model import CervicalFractureNet
from library.loss import HybridLoss

# Initialize logger
logger = get_logger("train_module")


class Trainer:
    """
    Trainer class for the 2.5D Anatomically-Guided Attention Network.
    Handles training loops, validation, gradient accumulation, and checkpointing.
    """

    def __init__(self, config):
        self.config = config
        self.device = self.config.DEVICE

        # DataLoaders
        self.train_loader, self.val_loader, _ = get_dataloaders(self.config)

        # Model
        self.model = CervicalFractureNet(self.config)
        self.model.to(self.device)

        # Loss
        self.criterion = HybridLoss()

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY,
        )

        # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=self.config.SCHEDULER_FACTOR,
            patience=self.config.SCHEDULER_PATIENCE,
            min_lr=self.config.MIN_LR,
        )

        # Mixed Precision Scaler
        self.scaler = GradScaler()

        # State
        self.best_val_loss = float("inf")
        self.patience_counter = 0

    def train_one_epoch(self, epoch):
        """
        Runs one epoch of training with gradient accumulation.
        """
        self.model.train()
        loss_meter = AverageMeter()
        main_loss_meter = AverageMeter()
        aux_loss_meter = AverageMeter()

        self.optimizer.zero_grad()

        # Iterate over dataloader (no progress bar as requested)
        for step, batch in enumerate(self.train_loader):
            # Move data to device
            images = batch["images"].to(self.device)
            fracture_labels = batch["fracture_labels"].to(self.device)
            aux_labels = batch["aux_labels"].to(self.device)

            targets = {"fracture_labels": fracture_labels, "aux_labels": aux_labels}

            # Mixed Precision Forward Pass
            with autocast():
                outputs = self.model(images)
                loss_dict = self.criterion(outputs, targets)

                # normalize loss for gradient accumulation
                loss = loss_dict["loss"] / self.config.GRAD_ACCUM_STEPS

            # Backward Pass
            self.scaler.scale(loss).backward()

            # Update Meters (scale back up for logging)
            loss_meter.update(loss_dict["loss"].item(), images.size(0))
            main_loss_meter.update(loss_dict["main_loss"].item(), images.size(0))
            aux_loss_meter.update(loss_dict["aux_loss"].item(), images.size(0))

            # Gradient Accumulation Step
            if (step + 1) % self.config.GRAD_ACCUM_STEPS == 0:
                # Clip Gradients
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.MAX_GRAD_NORM
                )

                # Optimizer Step
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

        logger.info(
            f"Epoch {epoch} Train | "
            f"Loss: {loss_meter.avg:.8f} | "
            f"Main: {main_loss_meter.avg:.8f} | "
            f"Aux: {aux_loss_meter.avg:.8f}"
        )

        return loss_meter.avg

    def validate(self, epoch):
        """
        Runs validation loop and calculates weighted log loss.
        """
        self.model.eval()
        loss_meter = AverageMeter()

        # Containers for metric calculation
        all_preds = []
        all_targets = []
        all_study_ids = []

        with torch.no_grad():
            for batch in self.val_loader:
                images = batch["images"].to(self.device)
                fracture_labels = batch["fracture_labels"].to(self.device)
                aux_labels = batch["aux_labels"].to(self.device)
                study_ids = batch["study_id"]

                targets = {"fracture_labels": fracture_labels, "aux_labels": aux_labels}

                # Forward
                outputs = self.model(images)
                loss_dict = self.criterion(outputs, targets)

                loss_meter.update(loss_dict["loss"].item(), images.size(0))

                # Store predictions (sigmoid applied) and targets for metric
                probs = torch.sigmoid(outputs["fracture_logits"]).cpu().numpy()
                truth = fracture_labels.cpu().numpy()

                all_preds.append(probs)
                all_targets.append(truth)
                all_study_ids.extend(study_ids)

        # Concatenate
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        # Calculate Competition Metric
        metric_score = self._calculate_metric(all_study_ids, all_preds, all_targets)

        logger.info(
            f"Epoch {epoch} Val   | "
            f"Loss: {loss_meter.avg:.8f} | "
            f"Metric (Weighted Log Loss): {metric_score:.8f}"
        )

        return loss_meter.avg, metric_score

    def _calculate_metric(self, study_ids, preds, targets):
        """
        Helper to format data and call the official metric function.
        """
        # Column names corresponding to model output indices
        # Defined in Dataset: C1, C2, C3, C4, C5, C6, C7, patient_overall
        cols = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "patient_overall"]

        # 1. Create y_true_df
        # Must have: StudyInstanceUID, patient_overall, C1...C7
        y_true_data = {"StudyInstanceUID": study_ids}
        for i, col in enumerate(cols):
            y_true_data[col] = targets[:, i]
        y_true_df = pd.DataFrame(y_true_data)

        # 2. Create y_pred_df
        # Must have: row_id, fractured
        row_ids = []
        fractured_probs = []

        for i, uid in enumerate(study_ids):
            for j, col in enumerate(cols):
                # Construct row_id
                if col == "patient_overall":
                    rid = f"{uid}_patient_overall"
                else:
                    rid = f"{uid}_{col}"

                row_ids.append(rid)
                fractured_probs.append(preds[i, j])

        y_pred_df = pd.DataFrame({"row_id": row_ids, "fractured": fractured_probs})

        return calculate_weighted_log_loss(y_true_df, y_pred_df)

    def fit(self):
        """
        Main training loop with early stopping.
        """
        logger.info("Starting training...")

        for epoch in range(1, self.config.EPOCHS + 1):
            # Train
            train_loss = self.train_one_epoch(epoch)

            # Validate
            val_loss, val_metric = self.validate(epoch)

            # Scheduler Step
            self.scheduler.step(val_loss)

            # Checkpointing & Early Stopping
            # We save based on Validation Loss (Hybrid) as it reflects both tasks,
            # but one could also save based on val_metric.
            if val_loss < self.best_val_loss - self.config.EARLY_STOPPING_MIN_DELTA:
                self.best_val_loss = val_loss
                self.patience_counter = 0

                logger.info(
                    f"New best model found (Val Loss: {val_loss:.8f}). Saving..."
                )
                torch.save(self.model.state_dict(), self.config.MODEL_SAVE_PATH)
            else:
                self.patience_counter += 1
                logger.info(
                    f"No improvement. Patience: {self.patience_counter}/{self.config.EARLY_STOPPING_PATIENCE}"
                )

            if self.patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
                logger.info("Early stopping triggered.")
                break

        logger.info("Training complete.")


def train_model():
    """
    Initializes and runs the training process.
    """
    config = Config()
    seed_everything(config.SEED)

    # Ensure working directory exists
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    trainer = Trainer(config)
    trainer.fit()
