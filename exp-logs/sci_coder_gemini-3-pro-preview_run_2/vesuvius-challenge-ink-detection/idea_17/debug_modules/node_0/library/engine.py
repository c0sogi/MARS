import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import sys
from library.config import Config
from library.utils import fbeta_score


class BCEDiceLoss(nn.Module):
    """
    Combined Binary Cross Entropy and Dice Loss.
    """

    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1e-6):
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, preds, targets):
        # BCE Loss
        bce = self.bce_loss(preds, targets)

        # Dice Loss
        preds_sigmoid = torch.sigmoid(preds)

        # Flatten
        preds_flat = preds_sigmoid.view(-1)
        targets_flat = targets.view(-1)

        intersection = (preds_flat * targets_flat).sum()
        union = preds_flat.sum() + targets_flat.sum()

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice

        return self.bce_weight * bce + self.dice_weight * dice_loss


def train_one_epoch(model, dataloader, optimizer, criterion, device, scheduler=None):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    # Iterate over batches
    # Using tqdm for progress tracking is standard, but prompt asked not to print progress bars.
    # We will iterate silently or with minimal print if needed, but standard practice usually allows tqdm.
    # However, "Only print the required information. Do not print progress bars" in prompt.
    # So I will use a standard loop.

    for batch_idx, batch in enumerate(dataloader):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        batch_size = images.size(0)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    if scheduler:
        scheduler.step()

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def evaluate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.
    Computes Loss and F0.5 Score.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            batch_size = images.size(0)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Apply sigmoid for metric calculation
            preds_prob = torch.sigmoid(outputs)

            # Store for global metric calculation (CPU to save GPU memory)
            all_preds.append(preds_prob.cpu().numpy())
            all_targets.append(labels.cpu().numpy())

    epoch_loss = running_loss / dataset_size

    # Concatenate all batches
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Flatten for metric calculation
    all_preds_flat = all_preds.flatten()
    all_targets_flat = all_targets.flatten()

    # Calculate F0.5 Score
    # Using the utility function provided in library.utils
    val_score = fbeta_score(
        all_preds_flat, all_targets_flat, beta=0.5, threshold=Config.THRESHOLD
    )

    return epoch_loss, val_score


def fit(
    model, train_loader, valid_loader, epochs=Config.NUM_EPOCHS, device=Config.DEVICE
):
    """
    Main training loop with Early Stopping.
    """
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    criterion = BCEDiceLoss()

    # Cosine Annealing Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-7
    )

    best_score = -np.inf
    patience_counter = 0
    best_model_path = Config.BEST_MODEL_PATH

    print(f"Starting training for {epochs} epochs on device: {device}")

    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scheduler
        )
        val_loss, val_score = evaluate(model, valid_loader, criterion, device)

        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val F0.5: {val_score}"
        )

        # Early Stopping & Model Checkpointing
        if val_score > best_score:
            best_score = val_score
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with F0.5 score: {best_score}")
        else:
            patience_counter += 1
            print(
                f"No improvement. Patience: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE}"
            )

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation F0.5 Score: {best_score}")
    return best_score
