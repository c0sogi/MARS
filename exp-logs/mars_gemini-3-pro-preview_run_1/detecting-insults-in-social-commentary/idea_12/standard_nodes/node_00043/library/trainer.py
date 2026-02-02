import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from library.configuration import Config
from library.utilities import AverageMeter, AWP


def train_fn(
    model, train_loader, optimizer, scheduler, device, epoch, config, awp=None
):
    """
    Executes one training epoch.
    Supports Adversarial Weight Perturbation (AWP) and Gradient Accumulation.
    """
    model.train()
    losses = AverageMeter()
    criterion = nn.BCEWithLogitsLoss()

    # Determine if AWP should be active for this epoch
    use_awp = False
    if config.use_awp and awp is not None and epoch >= config.awp_start_epoch:
        use_awp = True

    for step, batch in enumerate(train_loader):
        # Move inputs to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        svd_features = batch["svd_features"].to(device)
        targets = batch["target"].to(device).unsqueeze(1)  # Shape: (batch_size, 1)

        batch_size = input_ids.size(0)

        # 1. Forward Pass
        logits = model(input_ids, attention_mask, svd_features)
        loss = criterion(logits, targets)

        # Scale loss for gradient accumulation
        if config.gradient_accumulation_steps > 1:
            loss = loss / config.gradient_accumulation_steps

        # 2. Backward Pass
        loss.backward()

        # 3. Optimization Step
        if (step + 1) % config.gradient_accumulation_steps == 0:

            # Adversarial Weight Perturbation (AWP)
            if use_awp:
                awp.attack()
                # Re-forward pass with perturbed weights
                adv_logits = model(input_ids, attention_mask, svd_features)
                adv_loss = criterion(adv_logits, targets)

                if config.gradient_accumulation_steps > 1:
                    adv_loss = adv_loss / config.gradient_accumulation_steps

                adv_loss.backward()
                awp.restore()

            # Gradient Clipping
            nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

            optimizer.step()
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

        # Update metrics (scale loss back up for reporting)
        losses.update(loss.item() * config.gradient_accumulation_steps, batch_size)

    return losses.avg


def valid_fn(model, val_loader, device):
    """
    Evaluates the model on the validation set.
    Returns average loss, AUC score, and predictions.
    """
    model.eval()
    losses = AverageMeter()
    criterion = nn.BCEWithLogitsLoss()

    preds = []
    true_labels = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            svd_features = batch["svd_features"].to(device)
            targets = batch["target"].to(device).unsqueeze(1)

            batch_size = input_ids.size(0)

            # Forward pass
            logits = model(input_ids, attention_mask, svd_features)
            loss = criterion(logits, targets)

            losses.update(loss.item(), batch_size)

            # Apply sigmoid for probabilities
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            labels = targets.detach().cpu().numpy()

            preds.append(probs)
            true_labels.append(labels)

    preds = np.concatenate(preds)
    true_labels = np.concatenate(true_labels)

    # Compute AUC
    # Handle case where only one class is present in batch (though unlikely in full val set)
    try:
        auc_score = roc_auc_score(true_labels, preds)
    except ValueError:
        auc_score = 0.0

    return losses.avg, auc_score, preds


def inference_fn(model, loader, device, return_logits=False):
    """
    Generates predictions for a dataset.
    Args:
        return_logits (bool): If True, returns raw logits (useful for distillation/soft labels).
                              If False, returns sigmoid probabilities.
    """
    model.eval()
    preds = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            svd_features = batch["svd_features"].to(device)

            logits = model(input_ids, attention_mask, svd_features)

            if return_logits:
                output = logits.detach().cpu().numpy()
            else:
                output = torch.sigmoid(logits).detach().cpu().numpy()

            preds.append(output)

    return np.concatenate(preds)


class Trainer:
    """
    Orchestrates the training process for a single model/fold.
    """

    def __init__(
        self, config, model, train_loader, val_loader, optimizer, scheduler, device
    ):
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

        # Initialize AWP if enabled
        self.awp = None
        if self.config.use_awp:
            self.awp = AWP(
                model=self.model,
                optimizer=self.optimizer,
                adv_lr=self.config.awp_lr,
                adv_eps=self.config.awp_eps,
            )

    def fit(self, epochs):
        """
        Runs the training loop for the specified number of epochs.
        """
        best_auc = 0.0

        for epoch in range(epochs):
            start_time = time.time()

            # Train
            train_loss = train_fn(
                model=self.model,
                train_loader=self.train_loader,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                device=self.device,
                epoch=epoch,
                config=self.config,
                awp=self.awp,
            )

            # Validate
            val_loss, val_auc, _ = valid_fn(
                model=self.model, val_loader=self.val_loader, device=self.device
            )

            elapsed = time.time() - start_time

            # Print metrics with full precision
            print(f"Epoch {epoch + 1}/{epochs} - Time: {elapsed}s")
            print(f"Train Loss: {train_loss}")
            print(f"Val Loss: {val_loss}")
            print(f"Val AUC: {val_auc}")

            # Track best score
            if val_auc > best_auc:
                best_auc = val_auc

        return best_auc

    def predict(self, loader, return_logits=False):
        """
        Wrapper for inference.
        """
        return inference_fn(
            self.model, loader, self.device, return_logits=return_logits
        )
