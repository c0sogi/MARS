import os
import torch
import torch.nn as nn
import torch.optim as optim
from library.config import Config
from library.utils import calculate_f05


class DiceLoss(nn.Module):
    """
    Implementation of Dice Loss for binary segmentation.
    Computes 1 - Dice Coefficient.
    """

    def __init__(self, smooth=1e-7):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # Apply sigmoid to get probabilities from logits
        probs = torch.sigmoid(logits)

        # Flatten the tensors to compute the metric over the batch
        probs = probs.view(-1)
        targets = targets.view(-1)

        intersection = (probs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (
            probs.sum() + targets.sum() + self.smooth
        )

        return 1 - dice


def train_specialist(
    model, train_loader, val_loader, specialist_key, epochs=Config.EPOCHS, patience=5
):
    """
    Executes the training lifecycle for a single specialist model.

    Args:
        model (nn.Module): The PyTorch model (SpecialistSegFormer) to train.
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
        specialist_key (str): The identifier for the specialist ('A', 'B', or 'C').
        epochs (int): Maximum number of training epochs. Defaults to Config.EPOCHS.
        patience (int): Number of epochs to wait for improvement before early stopping.

    Returns:
        float: The best validation F0.5 score achieved during training.
    """
    device = torch.device(Config.DEVICE)
    model.to(device)

    # Configure Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)

    # Configure Loss Functions
    # We use a combination of Binary Cross Entropy and Dice Loss
    bce_criterion = nn.BCEWithLogitsLoss()
    dice_criterion = DiceLoss()

    best_val_score = -1.0
    patience_counter = 0

    # Prepare directory for saving checkpoints
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    save_path = os.path.join(Config.WORKING_DIR, f"model_{specialist_key}_best.pth")

    print(f"Starting training for Specialist {specialist_key} on device: {device}")

    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        running_train_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            # Calculate combined loss
            loss_bce = bce_criterion(outputs, labels)
            loss_dice = dice_criterion(outputs, labels)
            loss = loss_bce + loss_dice

            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

        avg_train_loss = running_train_loss / len(train_loader)

        # --- Validation Phase ---
        model.eval()
        running_val_loss = 0.0
        running_val_f05 = 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                # Calculate Validation Loss
                loss_bce = bce_criterion(outputs, labels)
                loss_dice = dice_criterion(outputs, labels)
                loss = loss_bce + loss_dice

                running_val_loss += loss.item()

                # Calculate Validation F0.5 Score
                # Apply sigmoid to convert logits to probabilities
                probs = torch.sigmoid(outputs)
                f05 = calculate_f05(labels, probs)
                running_val_f05 += f05

        avg_val_loss = running_val_loss / len(val_loader)
        avg_val_f05 = running_val_f05 / len(val_loader)

        # Print metrics with full precision
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Train Loss: {avg_train_loss}")
        print(f"Val Loss: {avg_val_loss}")
        print(f"Val F0.5: {avg_val_f05}")

        # --- Checkpointing and Early Stopping ---
        if avg_val_f05 > best_val_score:
            best_val_score = avg_val_f05
            patience_counter = 0

            # Save the model only if it exceeds the quality threshold defined in Config
            if avg_val_f05 >= Config.VALID_THRESHOLD:
                print(
                    f"Validation score {avg_val_f05} exceeds threshold {Config.VALID_THRESHOLD}. Saving model to {save_path}"
                )
                torch.save(model.state_dict(), save_path)
            else:
                print(
                    f"Validation score {avg_val_f05} is the best so far, but below threshold {Config.VALID_THRESHOLD}. Model not saved."
                )
        else:
            patience_counter += 1
            print(
                f"No improvement in validation score. Patience: {patience_counter}/{patience}"
            )

        if patience_counter >= patience:
            print(
                f"Early stopping triggered after {patience} epochs without improvement."
            )
            break

    print(
        f"Training finished for Specialist {specialist_key}. Best Validation F0.5 Score: {best_val_score}"
    )
    return best_val_score
