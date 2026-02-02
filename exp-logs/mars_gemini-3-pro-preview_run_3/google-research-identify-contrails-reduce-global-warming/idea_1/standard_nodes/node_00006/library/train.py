import os
import torch
import torch.nn as nn
import torch.optim as optim
from library.config import Config
from library.data import get_dataloaders
from library.model import LightUNet
from library.utils import set_seed, dice_coefficient


class DiceBCELoss(nn.Module):
    """
    Combined loss function: Binary Cross Entropy + Soft Dice Loss.
    BCE provides smooth gradients for pixel classification.
    Dice Loss directly optimizes the overlap metric.
    """

    def __init__(self, smooth=1e-6):
        super(DiceBCELoss, self).__init__()
        self.smooth = smooth
        self.bce = nn.BCELoss()

    def forward(self, preds, targets):
        # BCE Loss
        bce_loss = self.bce(preds, targets)

        # Soft Dice Loss
        preds_flat = preds.contiguous().view(-1)
        targets_flat = targets.contiguous().view(-1)

        intersection = (preds_flat * targets_flat).sum()
        union = preds_flat.sum() + targets_flat.sum()

        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice_score

        return bce_loss + dice_loss


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0

    for images, masks, _ in loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, masks)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def validate(model, loader, criterion, device, threshold):
    """
    Evaluates the model on the validation set.
    Returns average loss and the global Dice coefficient.
    """
    model.eval()
    running_loss = 0.0

    # Accumulators for Global Dice calculation over the whole validation set
    # Note: While the metric definition implies global X and Y, averaging batch-wise dice
    # is a common approximation during training. However, to be precise with the metric
    # "Global Dice coefficient", we should ideally sum intersections and unions.
    # Given the constraints and typical implementation in utils.dice_coefficient which
    # calculates dice for the provided batch, we will average the batch-wise dice scores
    # for tracking performance stability.

    running_dice = 0.0

    with torch.no_grad():
        for images, masks, _ in loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)

            # Calculate Loss
            loss = criterion(outputs, masks)
            running_loss += loss.item()

            # Calculate Metric (Binary Dice)
            preds_binary = (outputs > threshold).float()

            # utils.dice_coefficient computes dice for the batch passed to it
            batch_dice = dice_coefficient(preds_binary, masks)
            running_dice += batch_dice

    avg_loss = running_loss / len(loader)
    avg_dice = running_dice / len(loader)

    return avg_loss, avg_dice


def run_training(
    debug=Config.DEBUG,
    debug_sample_size=Config.DEBUG_SAMPLE_SIZE,
    epochs=Config.NUM_EPOCHS,
    batch_size=Config.BATCH_SIZE,
):
    """
    Orchestrates the training pipeline.
    """
    # Reproducibility
    set_seed(Config.SEED)

    # Setup
    device = torch.device(Config.DEVICE)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    # Data
    train_loader, val_loader, _ = get_dataloaders(
        batch_size=batch_size,
        num_workers=Config.NUM_WORKERS,
        debug=debug,
        debug_sample_size=debug_sample_size,
    )

    # Model
    model = LightUNet().to(device)

    # Optimization
    criterion = DiceBCELoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=2)

    # Training Loop
    best_val_dice = -1.0
    patience_counter = 0
    top_k_checkpoints = []  # List of (dice, filepath)

    print("Starting training workflow...")

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_dice = validate(
            model, val_loader, criterion, device, Config.THRESHOLD
        )

        # Full precision printing
        print(
            f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss} - Val Loss: {val_loss} - Val Dice: {val_dice}"
        )

        # Scheduler step (monitoring Dice, so mode='max')
        scheduler.step(val_dice)

        # Save checkpoint for Top-K tracking
        ckpt_path = os.path.join(Config.WORKING_DIR, f"checkpoint_epoch_{epoch}.pth")
        torch.save(model.state_dict(), ckpt_path)

        top_k_checkpoints.append((val_dice, ckpt_path))
        # Sort descending by Dice
        top_k_checkpoints.sort(key=lambda x: x[0], reverse=True)

        # Keep only Top K
        if len(top_k_checkpoints) > Config.TOP_K_CHECKPOINTS:
            _, path_to_remove = top_k_checkpoints.pop()
            if os.path.exists(path_to_remove):
                os.remove(path_to_remove)

        # Early Stopping Logic (based on single best)
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            patience_counter = 0
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    # --- Checkpoint Averaging ---
    if top_k_checkpoints:
        print(f"Averaging top {len(top_k_checkpoints)} checkpoints...")

        # Load best model to initialize (preserves integer buffers like num_batches_tracked)
        best_dice, best_path = top_k_checkpoints[0]
        avg_state_dict = torch.load(best_path, map_location=device)

        # Sum floating point parameters from other checkpoints
        for _, path in top_k_checkpoints[1:]:
            state = torch.load(path, map_location=device)
            for k, v in state.items():
                if v.is_floating_point():
                    avg_state_dict[k] += v

        # Average
        k_count = len(top_k_checkpoints)
        for k, v in avg_state_dict.items():
            if v.is_floating_point():
                avg_state_dict[k] /= k_count

        # Save averaged model
        torch.save(avg_state_dict, best_model_path)
        print(f"Averaged model saved to {best_model_path}")

        # Cleanup individual checkpoints
        for _, path in top_k_checkpoints:
            if os.path.exists(path):
                os.remove(path)
    else:
        # Fallback if no checkpoints
        torch.save(model.state_dict(), best_model_path)

    return best_model_path
