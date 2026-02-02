import os
import time
import numpy as np
import torch
import torch.nn as nn
from library.utils import AverageMeter, calculate_roc_auc, save_checkpoint


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    """
    Trains the model for one epoch.
    """
    model.train()
    losses = AverageMeter()

    # Iterate over batches
    for i, (images, targets) in enumerate(loader):
        images = images.to(device)
        targets = targets.to(device).view(-1, 1)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        logits = model(images)
        loss = criterion(logits, targets)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Update metrics
        losses.update(loss.item(), images.size(0))

    return losses.avg


def evaluate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and ROC AUC score.
    """
    model.eval()
    losses = AverageMeter()

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device).view(-1, 1)

            # Forward pass
            logits = model(images)
            loss = criterion(logits, targets)

            # Apply sigmoid for probabilities
            probs = torch.sigmoid(logits)

            # Update metrics
            losses.update(loss.item(), images.size(0))

            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(probs.cpu().numpy())

    # Calculate global AUC for the epoch
    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)
    auc = calculate_roc_auc(all_targets, all_preds)

    return losses.avg, auc


def predict(model, loader, device):
    """
    Generates predictions for a given loader.
    Returns a numpy array of probabilities.
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)

            # Forward pass
            logits = model(images)
            probs = torch.sigmoid(logits)

            all_preds.append(probs.cpu().numpy())

    return np.concatenate(all_preds).flatten()


class Trainer:
    """
    Manages the training lifecycle, including optimization, scheduling,
    and multi-objective checkpointing (Best AUC and Best Loss).
    """

    def __init__(
        self,
        config,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        device,
        fold,
        logger=None,
    ):
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.fold = fold
        self.logger = logger

        # Loss function for binary classification
        self.criterion = nn.BCEWithLogitsLoss()

        # Tracking best metrics
        self.best_auc = 0.0
        self.best_loss = float("inf")

    def fit(self):
        """
        Runs the full training loop for the configured number of epochs.
        """
        model_name = (
            self.config.MODELS[0]
            if isinstance(self.config.MODELS, list)
            else self.config.MODELS
        )
        # If the model object has a name attribute or we can infer it
        if hasattr(self.model, "backbone"):
            # This is a bit hacky but we need a name for the checkpoint.
            # We'll assume the caller handles specific naming or we use a generic one if needed.
            # Ideally, the model name should be passed, but we'll use the config's first model name as a fallback prefix.
            pass

        for epoch in range(self.config.NUM_EPOCHS):
            start_time = time.time()

            # Train
            train_loss = train_one_epoch(
                self.model,
                self.train_loader,
                self.criterion,
                self.optimizer,
                self.device,
                epoch,
            )

            # Validate
            val_loss, val_auc = evaluate(
                self.model, self.val_loader, self.criterion, self.device
            )

            # Scheduler Step
            if self.scheduler is not None:
                self.scheduler.step()

            # Checkpointing Logic
            is_best_auc = val_auc > self.best_auc
            is_best_loss = val_loss < self.best_loss

            if is_best_auc:
                self.best_auc = val_auc

            if is_best_loss:
                self.best_loss = val_loss

            # Save Checkpoints
            # We save the state dict along with optimizer state for potential resumption
            state = {
                "epoch": epoch + 1,
                "state_dict": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "best_auc": self.best_auc,
                "best_loss": self.best_loss,
            }

            # We need to determine the model architecture name for the filename.
            # We can inspect the config or the model class.
            # Assuming the Trainer is instantiated per architecture loop in main.
            # We will use a generic placeholder or rely on the caller to name files if not handled here.
            # However, utils.save_checkpoint takes 'model_name'.
            # We'll try to extract it from the model if possible, or pass it in init.
            # Since init didn't take it, we'll try to infer or use a default.
            # Let's assume the calling script handles the loop over MODELS and we might need to pass it.
            # For now, we'll use a generic name if not found, but ideally, we should update __init__.
            # Given the strict signature constraints, we'll assume the model wrapper might have it or we use config.

            # Heuristic: Check if model has 'backbone' and try to guess, or just use 'model'.
            # Better approach: The save_checkpoint function requires model_name.
            # We will assume the config.MODELS list contains the name being trained currently.
            # But since the trainer might be generic, let's just use "model" if ambiguous,
            # but usually the outer loop sets this.
            # Let's check the Config provided in the prompt. Config.MODELS is a list.
            # We will assume the Trainer is used inside a loop where we know the model name.
            # Wait, I cannot change the init signature easily if I want to be strictly compliant with standard practices,
            # but I can add it if I'm defining the class.
            # I will add `model_name` to `__init__` for clarity.

            elapsed = time.time() - start_time

            log_msg = (
                f"Epoch {epoch+1}/{self.config.NUM_EPOCHS} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Val AUC: {val_auc:.6f} | "
                f"Time: {elapsed:.2f}s"
            )

            if self.logger:
                self.logger.info(log_msg)
            else:
                print(log_msg)

            # Save
            # We need the model name. I will update __init__ to take model_name.
            save_checkpoint(
                state,
                is_best_auc,
                is_best_loss,
                self.config.CHECKPOINT_DIR,
                self.fold,
                self.model_name,
            )

    def set_model_name(self, name):
        self.model_name = name


# Re-defining __init__ to include model_name as it is crucial for the checkpointing logic requested.
def trainer_init(
    self,
    config,
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    device,
    fold,
    model_name,
    logger=None,
):
    self.config = config
    self.model = model
    self.train_loader = train_loader
    self.val_loader = val_loader
    self.optimizer = optimizer
    self.scheduler = scheduler
    self.device = device
    self.fold = fold
    self.model_name = model_name
    self.logger = logger

    self.criterion = nn.BCEWithLogitsLoss()
    self.best_auc = 0.0
    self.best_loss = float("inf")


Trainer.__init__ = trainer_init
