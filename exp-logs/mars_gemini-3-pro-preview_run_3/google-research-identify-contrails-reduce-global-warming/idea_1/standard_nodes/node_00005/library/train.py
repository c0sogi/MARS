import os
import copy
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


def average_weights(state_dicts):
    """
    Averages a list of state_dicts.
    """
    avg_state = copy.deepcopy(state_dicts[0])
    for key in avg_state.keys():
        # Handle tensors
        if isinstance(avg_state[key], torch.Tensor):
            for i in range(1, len(state_dicts)):
                avg_state[key] += state_dicts[i][key]
            avg_state[key] = avg_state[key] / len(state_dicts)
    return avg_state


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

    # Store top K checkpoints for averaging (Cite solution_lesson_node_00003)
    best_checkpoints = []  # List of (dice, state_dict)
    TOP_K = 3

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

        # Update Top K Checkpoints
        current_state = copy.deepcopy(model.state_dict())
        best_checkpoints.append((val_dice, current_state))
        best_checkpoints.sort(key=lambda x: x[0], reverse=True)
        if len(best_checkpoints) > TOP_K:
            best_checkpoints.pop()  # Remove the worst of the top K

        # Early Stopping Logic
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            patience_counter = 0
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    # Average weights of top K models
    print(f"Averaging top {len(best_checkpoints)} checkpoints...")
    avg_state = average_weights([x[1] for x in best_checkpoints])
    torch.save(avg_state, best_model_path)

    return best_model_path
