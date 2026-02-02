import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from library.config import Config
from library.utils import set_seed, fbeta_score, dice_coef


class BCEDiceLoss(nn.Module):
    """
    Combined Binary Cross Entropy and Dice Loss for segmentation.
    """

    def __init__(self, smooth=1e-7):
        super(BCEDiceLoss, self).__init__()
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, preds, targets):
        # BCE Loss
        bce_loss = self.bce(preds, targets)

        # Dice Loss
        # Apply sigmoid to convert logits to probabilities
        preds_sigmoid = torch.sigmoid(preds)

        # Flatten tensors to compute global Dice for the batch
        preds_flat = preds_sigmoid.view(-1)
        targets_flat = targets.view(-1)

        intersection = (preds_flat * targets_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (
            preds_flat.sum() + targets_flat.sum() + self.smooth
        )
        dice_loss = 1.0 - dice

        return bce_loss + dice_loss


def train_specialist(model, train_loader, val_loader, specialist_config):
    """
    Trains a single specialist model on a specific Z-range.

    Args:
        model: The PyTorch model instance to train.
        train_loader: DataLoader for the training set.
        val_loader: DataLoader for the validation set.
        specialist_config: Dictionary containing 'name' and 'checkpoint_path'.

    Returns:
        float: The best validation F0.5 score achieved.
    """
    # Ensure reproducibility
    set_seed(Config.SEED)

    device = Config.DEVICE
    model = model.to(device)

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Loss Function
    criterion = BCEDiceLoss()

    # Training State
    best_val_f05 = 0.0
    patience_counter = 0
    checkpoint_path = specialist_config["checkpoint_path"]

    print(f"Starting training for specialist: {specialist_config['name']}")
    print(f"Training on device: {device}")

    for epoch in range(Config.EPOCHS):
        # --- Training Phase ---
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)

        # --- Validation Phase ---
        model.eval()
        val_f05_scores = []
        val_dice_scores = []

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)

                outputs = model(images)

                # Compute metrics
                # fbeta_score and dice_coef handle sigmoid internally
                f05 = fbeta_score(outputs, labels, beta=Config.BETA, threshold=0.5)
                dice = dice_coef(outputs, labels)

                val_f05_scores.append(f05.item())
                val_dice_scores.append(dice.item())

        avg_val_f05 = np.mean(val_f05_scores)
        avg_val_dice = np.mean(val_dice_scores)

        # Print metrics with full precision
        print(
            f"Epoch {epoch + 1}/{Config.EPOCHS} | Loss: {avg_train_loss} | Val F0.5: {avg_val_f05} | Val Dice: {avg_val_dice}"
        )

        # --- Checkpointing & Early Stopping ---
        # We track the best score to manage patience.
        # We only save the model if it passes the validation gating threshold.

        if avg_val_f05 > best_val_f05:
            best_val_f05 = avg_val_f05
            patience_counter = 0

            # Validation Gating
            if avg_val_f05 > Config.VALID_THRESHOLD:
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Model saved to {checkpoint_path}")
        else:
            patience_counter += 1

        if patience_counter >= Config.PATIENCE:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

    return best_val_f05
