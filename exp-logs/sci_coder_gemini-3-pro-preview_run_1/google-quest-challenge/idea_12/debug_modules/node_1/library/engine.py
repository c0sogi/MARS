import torch
import torch.nn as nn
import numpy as np
import sys
from library.config import Config
from library.utils import compute_spearman_metric


class EarlyStopping:
    """
    Early stopping utility to stop training when the validation metric stops improving.
    Also handles saving the best model state.
    """

    def __init__(self, patience=3, mode="max", delta=0.0, save_path="best_model.pth"):
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.save_path = save_path
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        self.val_score_min = np.Inf
        self.val_score_max = -np.Inf

        if mode == "min":
            self.check_improvement = self._check_min
        else:
            self.check_improvement = self._check_max

    def _check_min(self, score):
        return score < (self.best_score - self.delta)

    def _check_max(self, score):
        return score > (self.best_score + self.delta)

    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model)
        elif self.check_improvement(score):
            self.best_score = score
            self.save_checkpoint(score, model)
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, score, model):
        """Saves model when validation score improves."""
        torch.save(model.state_dict(), self.save_path)


def get_optimizer_params(model):
    """
    Configures parameter groups for the optimizer with differential learning rates
    and weight decay exclusion for bias/LayerNorm layers.
    """
    # Define parameter groups
    backbone_params_decay = []
    backbone_params_no_decay = []
    head_params_decay = []
    head_params_no_decay = []

    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Check if parameter belongs to backbone or head
        is_backbone = "backbone" in name

        # Check if parameter should be excluded from weight decay
        is_no_decay = any(nd in name for nd in no_decay)

        if is_backbone:
            if is_no_decay:
                backbone_params_no_decay.append(param)
            else:
                backbone_params_decay.append(param)
        else:
            if is_no_decay:
                head_params_no_decay.append(param)
            else:
                head_params_decay.append(param)

    optimizer_grouped_parameters = [
        # Backbone parameters (Low LR)
        {
            "params": backbone_params_decay,
            "weight_decay": Config.WEIGHT_DECAY,
            "lr": Config.LR_BACKBONE,
        },
        {
            "params": backbone_params_no_decay,
            "weight_decay": 0.0,
            "lr": Config.LR_BACKBONE,
        },
        # Head parameters (High LR)
        {
            "params": head_params_decay,
            "weight_decay": Config.WEIGHT_DECAY,
            "lr": Config.LR_HEAD,
        },
        {
            "params": head_params_no_decay,
            "weight_decay": 0.0,
            "lr": Config.LR_HEAD,
        },
    ]

    return optimizer_grouped_parameters


def train_one_epoch(model, dataloader, optimizer, device, epoch, scheduler=None):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    criterion = nn.BCEWithLogitsLoss()

    for batch_idx, batch in enumerate(dataloader):
        # Move batch to device
        q_input_ids = batch["q_input_ids"].to(device)
        q_attention_mask = batch["q_attention_mask"].to(device)
        a_input_ids = batch["a_input_ids"].to(device)
        a_attention_mask = batch["a_attention_mask"].to(device)
        labels = batch["labels"].to(device)

        batch_size = labels.size(0)

        # Forward pass
        optimizer.zero_grad()

        logits = model(
            q_input_ids=q_input_ids,
            q_attention_mask=q_attention_mask,
            a_input_ids=a_input_ids,
            a_attention_mask=a_attention_mask,
        )

        loss = criterion(logits, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    print(f"Epoch {epoch} | Train Loss: {epoch_loss:.6f}")

    return epoch_loss


def validate(model, dataloader, device):
    """
    Evaluates the model on the validation set.
    Computes Mean Column-wise Spearman's Correlation.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    criterion = nn.BCEWithLogitsLoss()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            q_input_ids = batch["q_input_ids"].to(device)
            q_attention_mask = batch["q_attention_mask"].to(device)
            a_input_ids = batch["a_input_ids"].to(device)
            a_attention_mask = batch["a_attention_mask"].to(device)
            labels = batch["labels"].to(device)

            batch_size = labels.size(0)

            logits = model(
                q_input_ids=q_input_ids,
                q_attention_mask=q_attention_mask,
                a_input_ids=a_input_ids,
                a_attention_mask=a_attention_mask,
            )

            loss = criterion(logits, labels)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Apply sigmoid for predictions
            preds = torch.sigmoid(logits)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(labels.cpu().numpy())

    epoch_loss = running_loss / dataset_size

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    spearman_score = compute_spearman_metric(all_preds, all_targets)

    print(f"Val Loss: {epoch_loss:.6f} | Spearman Correlation: {spearman_score:.10f}")

    return epoch_loss, spearman_score


def predict(model, dataloader, device):
    """
    Generates predictions for the test set.
    Returns a numpy array of probabilities (N, 30).
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for batch in dataloader:
            q_input_ids = batch["q_input_ids"].to(device)
            q_attention_mask = batch["q_attention_mask"].to(device)
            a_input_ids = batch["a_input_ids"].to(device)
            a_attention_mask = batch["a_attention_mask"].to(device)

            logits = model(
                q_input_ids=q_input_ids,
                q_attention_mask=q_attention_mask,
                a_input_ids=a_input_ids,
                a_attention_mask=a_attention_mask,
            )

            preds = torch.sigmoid(logits)
            all_preds.append(preds.cpu().numpy())

    return np.concatenate(all_preds, axis=0)
