import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from library.config import DEVICE, AUX_LOSS_WEIGHT, SUBMISSION_PATH


# -----------------------------------------------------------------------------
# Loss Function
# -----------------------------------------------------------------------------
def loss_fn(outputs, targets, identities):
    """
    Computes the weighted sum of Toxicity BCE and Identity BCE.

    Args:
        outputs (tuple): (toxicity_logits, identity_logits)
        targets (torch.Tensor): Toxicity targets (batch_size,)
        identities (torch.Tensor): Identity targets (batch_size, num_identities)
    """
    tox_logits, identity_logits = outputs

    # Reshape targets to match logits (batch_size, 1)
    tox_loss = nn.BCEWithLogitsLoss()(tox_logits, targets.view(-1, 1))

    # Identity loss (multi-label binary cross entropy)
    identity_loss = nn.BCEWithLogitsLoss()(identity_logits, identities)

    # Weighted combination
    total_loss = (1 - AUX_LOSS_WEIGHT) * tox_loss + AUX_LOSS_WEIGHT * identity_loss
    return total_loss


# -----------------------------------------------------------------------------
# Training Engine
# -----------------------------------------------------------------------------
def train_fn(data_loader, model, optimizer, device, scheduler=None):
    """
    Trains the model for one epoch.
    """
    model.train()
    total_loss = 0.0

    for batch_idx, data in enumerate(data_loader):
        input_ids, masks, targets, identities = data

        input_ids = input_ids.to(device)
        masks = masks.to(device)
        targets = targets.to(device)
        identities = identities.to(device)

        # Device-Side Trimming
        # Calculate the maximum sequence length in this batch (ignoring padding)
        # masks is (batch_size, seq_len), sum(dim=1) gives real length per sample
        max_len = masks.sum(dim=1).max().item()

        # Slice inputs to the dynamic max length to save compute
        input_ids = input_ids[:, :max_len]
        masks = masks[:, :max_len]

        optimizer.zero_grad()

        # Forward pass
        outputs = model(input_ids, masks)

        # Compute loss
        loss = loss_fn(outputs, targets, identities)

        # Backward pass
        loss.backward()
        optimizer.step()

        if scheduler:
            scheduler.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    print(f"Average Train Loss: {avg_loss}")
    return avg_loss


# -----------------------------------------------------------------------------
# Evaluation Engine
# -----------------------------------------------------------------------------
def eval_fn(data_loader, model, device):
    """
    Evaluates the model on the validation set.
    Returns lists of predictions, targets, and identities for metric calculation.
    """
    model.eval()

    fin_targets = []
    fin_outputs = []
    fin_identities = []

    with torch.no_grad():
        for data in data_loader:
            input_ids, masks, targets, identities = data

            input_ids = input_ids.to(device)
            masks = masks.to(device)

            # Device-Side Trimming
            max_len = masks.sum(dim=1).max().item()
            input_ids = input_ids[:, :max_len]
            masks = masks[:, :max_len]

            # Forward pass
            tox_logits, _ = model(input_ids, masks)

            # Convert logits to probabilities
            tox_probs = torch.sigmoid(tox_logits)

            # Move to CPU and collect
            # targets are already on CPU from dataloader usually, but .numpy() handles it
            fin_targets.extend(targets.numpy().tolist())
            fin_outputs.extend(tox_probs.cpu().numpy().flatten().tolist())
            fin_identities.extend(identities.numpy().tolist())

    return fin_outputs, fin_targets, fin_identities


# -----------------------------------------------------------------------------
# Inference Engine
# -----------------------------------------------------------------------------
def inference_fn(data_loader, model, device):
    """
    Generates predictions for the test set.
    """
    model.eval()

    fin_ids = []
    fin_preds = []

    with torch.no_grad():
        for data in data_loader:
            input_ids, masks, ids = data

            input_ids = input_ids.to(device)
            masks = masks.to(device)

            # Device-Side Trimming
            max_len = masks.sum(dim=1).max().item()
            input_ids = input_ids[:, :max_len]
            masks = masks[:, :max_len]

            # Forward pass
            tox_logits, _ = model(input_ids, masks)
            tox_probs = torch.sigmoid(tox_logits)

            fin_ids.extend(ids.numpy().tolist())
            fin_preds.extend(tox_probs.cpu().numpy().flatten().tolist())

    return fin_ids, fin_preds


def save_submission(ids, preds):
    """
    Saves the predictions to the submission file.
    """
    df = pd.DataFrame({"id": ids, "prediction": preds})
    # Ensure IDs are integers
    df["id"] = df["id"].astype(int)

    df.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
class EarlyStopping:
    """
    Early stops the training if validation score doesn't improve after a given patience.
    """

    def __init__(
        self, patience=2, mode="max", delta=0.0001, save_path="checkpoint.pth"
    ):
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.save_path = save_path
        self.counter = 0
        self.best_score = None
        self.early_stop = False

        if mode == "min":
            self.val_score = np.inf
        else:
            self.val_score = -np.inf

    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model)
        elif self.mode == "min":
            if score > self.best_score + self.delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint(score, model)
                self.counter = 0
        elif self.mode == "max":
            if score < self.best_score - self.delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint(score, model)
                self.counter = 0

    def save_checkpoint(self, score, model):
        self.val_score = score
        torch.save(model.state_dict(), self.save_path)
