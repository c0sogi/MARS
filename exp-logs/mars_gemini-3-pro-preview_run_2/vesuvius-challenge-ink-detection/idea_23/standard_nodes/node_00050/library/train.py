import os
import torch
import torch.nn as nn
import torch.optim as optim
from library.config import Config
from library.utils import set_seed, fbeta_score
from library.data import get_dataloaders
from library.model import SegFormerSpecialist


class BCEDiceLoss(nn.Module):
    """
    Combined Binary Cross Entropy and Dice Loss.
    Optimizes for both pixel-wise classification accuracy and geometric overlap.
    """

    def __init__(self, smooth=1e-6):
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.smooth = smooth

    def forward(self, logits, targets):
        # BCE Loss
        bce_loss = self.bce(logits, targets)

        # Dice Loss
        probs = torch.sigmoid(logits)

        # Flatten for Dice calculation
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (probs_flat * targets_flat).sum()
        union = probs_flat.sum() + targets_flat.sum()

        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice_score

        return bce_loss + dice_loss


def train_specialist(
    specialist_type,
    epochs=Config.EPOCHS,
    patience=Config.PATIENCE,
    gating_threshold=0.55,
):
    """
    Trains a single specialist model (High, Mid, or Low) using the MDSE strategy.

    Args:
        specialist_type (str): The type of specialist to train ('High', 'Mid', 'Low').
        epochs (int): Maximum number of training epochs.
        patience (int): Early stopping patience counter.
        gating_threshold (float): Minimum validation F0.5 score required to save the model checkpoint.

    Returns:
        float: The best validation F0.5 score achieved during training.
    """
    # Set reproducible seed
    set_seed(Config.SEED)

    print(f"--- Initializing Training for Specialist: {specialist_type} ---")

    # Initialize DataLoaders
    # Config.DEBUG and Config.MAX_TRAIN_SAMPLES are handled internally by get_dataloaders
    train_loader, val_loader = get_dataloaders(specialist_type)

    # Initialize Model
    model = SegFormerSpecialist()
    model.to(Config.DEVICE)

    # Optimizer and Loss
    # Using conservative learning rate as per Config
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    criterion = BCEDiceLoss()

    # Training State
    best_val_score = 0.0
    patience_counter = 0
    save_path = os.path.join(Config.WORKING_DIR, f"model_{specialist_type}.pth")

    print(f"Training started on device: {Config.DEVICE}")

    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(Config.DEVICE)
            labels = labels.to(Config.DEVICE)

            optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)

        # --- Validation Phase ---
        model.eval()
        val_running_loss = 0.0
        val_fbeta_sum = 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(Config.DEVICE)
                labels = labels.to(Config.DEVICE)

                logits = model(images)
                loss = criterion(logits, labels)
                val_running_loss += loss.item()

                # Compute Metric (F0.5)
                probs = torch.sigmoid(logits)
                score = fbeta_score(
                    probs, labels, beta=Config.DICE_BETA, threshold=Config.THRESHOLD
                )
                val_fbeta_sum += score

        avg_val_loss = val_running_loss / len(val_loader)
        avg_val_score = val_fbeta_sum / len(val_loader)

        # Print metrics with full precision
        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {avg_train_loss} | "
            f"Val Loss: {avg_val_loss} | "
            f"Val F0.5: {avg_val_score}"
        )

        # --- Checkpointing & Early Stopping ---
        if avg_val_score > best_val_score:
            best_val_score = avg_val_score
            patience_counter = 0

            # Validation Gating: Only save if above threshold
            if avg_val_score >= gating_threshold:
                torch.save(model.state_dict(), save_path)
                print(
                    f"Validation score improved and exceeded threshold ({gating_threshold}). Model saved to {save_path}"
                )
            else:
                print(
                    f"Validation score improved ({avg_val_score}) but is below gating threshold ({gating_threshold}). Model NOT saved."
                )
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

    print(
        f"--- Training Finished for {specialist_type}. Best Val F0.5: {best_val_score} ---"
    )
    return best_val_score
