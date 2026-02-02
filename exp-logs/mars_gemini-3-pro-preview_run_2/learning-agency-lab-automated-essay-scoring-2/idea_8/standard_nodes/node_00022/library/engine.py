import torch
import torch.nn as nn
import numpy as np
import time
from library.config import CFG


class AWP:
    """
    Adversarial Weight Perturbation (AWP) implementation.
    Perturbs model weights in the direction of the gradient to maximize loss,
    regularizing the model and flattening the loss landscape.
    """

    def __init__(self, model, optimizer, adv_lr, adv_eps, start_epoch, scaler=None):
        self.model = model
        self.optimizer = optimizer
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.start_epoch = start_epoch
        self.scaler = scaler
        self.backup = {}

    def attack(self):
        """
        Perturbs the weights based on the current gradients.
        """
        e = 1e-6
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and "weight" in name:
                # Save original weights
                self.backup[name] = param.data.clone()

                grad = param.grad
                norm_grad = torch.norm(grad)
                norm_data = torch.norm(param.data)

                if norm_grad != 0 and not torch.isnan(norm_grad):
                    # Calculate perturbation
                    r_at = self.adv_lr * grad / (norm_grad + e) * (norm_data + e)
                    # Apply perturbation
                    param.data.add_(r_at)

    def restore(self):
        """
        Restores the original weights from backup.
        """
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


def train_one_epoch(epoch, model, dataloader, optimizer, scheduler, device, awp=None):
    """
    Trains the model for one epoch using SmoothL1Loss and optional AWP.
    """
    model.train()
    scaler = torch.cuda.amp.GradScaler()
    criterion = nn.SmoothL1Loss(reduction="mean")

    dataset_size = 0
    running_loss = 0.0

    start_time = time.time()

    for step, data in enumerate(dataloader):
        input_ids = data["input_ids"].to(device)
        attention_mask = data["attention_mask"].to(device)
        labels = data["labels"].to(device)

        batch_size = input_ids.size(0)

        # --- Forward Pass (Clean) ---
        with torch.cuda.amp.autocast():
            y_preds = model(input_ids, attention_mask)
            loss = criterion(y_preds, labels)

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

        # --- Backward Pass (Clean) ---
        scaler.scale(loss).backward()

        # --- Adversarial Weight Perturbation ---
        if awp is not None and epoch >= awp.start_epoch:
            # Unscale gradients to get correct magnitude for attack
            scaler.unscale_(optimizer)

            # Perturb weights
            awp.attack()

            # Forward Pass (Adversarial)
            with torch.cuda.amp.autocast():
                y_preds_adv = model(input_ids, attention_mask)
                loss_adv = criterion(y_preds_adv, labels)

            # Backward Pass (Adversarial)
            # We replace the clean gradients with adversarial gradients
            optimizer.zero_grad()
            scaler.scale(loss_adv).backward()

            # Restore original weights
            awp.restore()

        # --- Optimization Step ---
        # Unscale gradients (if not already done or if replaced by AWP) for clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        if scheduler is not None:
            scheduler.step()

    epoch_loss = running_loss / dataset_size
    print(
        f"Epoch {epoch+1} | Train Loss: {epoch_loss:.6f} | Time: {time.time() - start_time:.0f}s"
    )

    return epoch_loss


def valid_one_epoch(model, dataloader, device):
    """
    Evaluates the model on the validation set.
    Returns average loss, predictions, and true labels.
    """
    model.eval()
    criterion = nn.SmoothL1Loss(reduction="mean")

    dataset_size = 0
    running_loss = 0.0

    preds = []
    targets = []

    with torch.no_grad():
        for step, data in enumerate(dataloader):
            input_ids = data["input_ids"].to(device)
            attention_mask = data["attention_mask"].to(device)
            labels = data["labels"].to(device)

            batch_size = input_ids.size(0)

            with torch.cuda.amp.autocast():
                y_preds = model(input_ids, attention_mask)
                loss = criterion(y_preds, labels)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            preds.append(y_preds.to("cpu").numpy())
            targets.append(labels.to("cpu").numpy())

    predictions = np.concatenate(preds)
    true_labels = np.concatenate(targets)
    epoch_loss = running_loss / dataset_size

    # Print full precision as requested
    print(f"Validation Loss: {epoch_loss:.15f}")

    return epoch_loss, predictions, true_labels


def inference_fn(model, dataloader, device):
    """
    Generates predictions for the test set.
    """
    model.eval()
    preds = []

    with torch.no_grad():
        for step, data in enumerate(dataloader):
            input_ids = data["input_ids"].to(device)
            attention_mask = data["attention_mask"].to(device)

            with torch.cuda.amp.autocast():
                y_preds = model(input_ids, attention_mask)

            preds.append(y_preds.to("cpu").numpy())

    predictions = np.concatenate(preds)
    return predictions
