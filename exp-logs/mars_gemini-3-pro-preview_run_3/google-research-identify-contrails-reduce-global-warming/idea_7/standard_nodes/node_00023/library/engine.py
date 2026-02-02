import os
import torch
import torch.nn as nn
import glob

from library.config import Config
from library.utils import dice_coef


class CheckpointManager:
    """
    Manages saving and deleting checkpoints to keep only the top K best models
    based on a specific metric (higher is better).
    """

    def __init__(self, save_dir, top_k=Config.TOP_K_CHECKPOINTS):
        self.save_dir = save_dir
        self.top_k = top_k
        self.checkpoints = (
            []
        )  # List of dicts: {'path': str, 'score': float, 'epoch': int}

    def save(self, model, optimizer, epoch, score):
        """
        Saves the model if it qualifies as one of the top K checkpoints.
        """
        filename = f"checkpoint_epoch_{epoch}_dice_{score:.6f}.pth"
        path = os.path.join(self.save_dir, filename)

        # Create state dictionary
        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "score": score,
        }

        # Logic to determine if we should save
        # 1. If we haven't filled top_k yet, save.
        # 2. If we have filled top_k, check if current score is better than the worst score.

        should_save = False
        if len(self.checkpoints) < self.top_k:
            should_save = True
        else:
            # Sort by score descending
            self.checkpoints.sort(key=lambda x: x["score"], reverse=True)
            worst_entry = self.checkpoints[-1]
            if score > worst_entry["score"]:
                should_save = True

        if should_save:
            torch.save(state, path)
            self.checkpoints.append({"path": path, "score": score, "epoch": epoch})

            # Prune if we exceed top_k
            self.checkpoints.sort(key=lambda x: x["score"], reverse=True)
            while len(self.checkpoints) > self.top_k:
                to_remove = self.checkpoints.pop()  # Removes the last (worst) one
                if os.path.exists(to_remove["path"]):
                    os.remove(to_remove["path"])

            return path

        return None


def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        batch_size = images.size(0)

        optimizer.zero_grad()

        # Mixed Precision Training
        with torch.cuda.amp.autocast(enabled=True):
            logits = model(images)
            loss = criterion(logits, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Accumulate loss (weighted by batch size for correct averaging)
        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size if dataset_size > 0 else 0.0
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and average Dice coefficient.
    """
    model.eval()
    running_loss = 0.0
    running_dice = 0.0
    dataset_size = 0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            batch_size = images.size(0)

            with torch.cuda.amp.autocast(enabled=True):
                logits = model(images)
                loss = criterion(logits, masks)

            # Calculate Dice
            # Apply sigmoid to logits to get probabilities
            probs = torch.sigmoid(logits)

            # dice_coef calculates global dice for the batch (flattened)
            batch_dice = dice_coef(probs, masks)

            running_loss += loss.item() * batch_size
            running_dice += batch_dice.item() * batch_size
            dataset_size += batch_size

    epoch_loss = running_loss / dataset_size if dataset_size > 0 else 0.0
    epoch_dice = running_dice / dataset_size if dataset_size > 0 else 0.0

    return epoch_loss, epoch_dice
