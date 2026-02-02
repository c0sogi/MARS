import os
import time
import random
import numpy as np
import torch
import torch.optim as optim
from library.config import Config
from library.utils import calculate_fbeta, DiceBCELoss
from library.data import get_dataloaders
from library.model import StratifiedSegFormer


def set_seed(seed):
    """
    Sets the random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs, masks)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and average F0.5 score.
    """
    model.eval()
    running_loss = 0.0
    running_score = 0.0

    # We accumulate scores per batch and average them.
    # Alternatively, one could accumulate TP/FP/FN globally, but averaging batch F-scores
    # is a standard approximation for monitoring training progress.

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            running_loss += loss.item() * images.size(0)

            # Apply sigmoid for metric calculation
            probs = torch.sigmoid(outputs)

            # Calculate F0.5 score for this batch
            # We calculate metric per image or per batch?
            # calculate_fbeta in utils handles flattened tensors, effectively treating the batch as one large volume.
            batch_score = calculate_fbeta(
                probs, masks, beta=0.5, threshold=Config.THRESHOLD
            )
            running_score += batch_score * images.size(0)

    avg_loss = running_loss / len(loader.dataset)
    avg_score = running_score / len(loader.dataset)

    return avg_loss, avg_score


def train_model(debug=False):
    """
    Main training routine.

    Args:
        debug (bool): If True, uses a smaller subset of data for quick debugging.
    """
    set_seed(Config.SEED)

    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 1. Prepare Data
    # load_cached_data=True ensures we use existing .npy files if available, or generate them.
    train_loader, val_loader = get_dataloaders(load_cached_data=True, debug=debug)

    # 2. Prepare Model
    model = StratifiedSegFormer(pretrained=True)
    model.to(device)

    # 3. Optimizer & Loss
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=Config.FACTOR,
        patience=Config.PATIENCE,
    )
    criterion = DiceBCELoss()

    # 4. Training Loop
    best_val_score = -1.0
    early_stopping_counter = 0
    # Hardcoded baseline mentioned in the task description for logging context,
    # though we save based on the best in *this* run.
    BASELINE_SCORE = 0.551

    print("Starting training...")

    for epoch in range(Config.NUM_EPOCHS):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_loss, val_score = validate(model, val_loader, criterion, device)

        # Scheduler Step
        scheduler.step(val_score)

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS} - "
            f"Time: {elapsed:.2f}s - "
            f"Train Loss: {train_loss} - "
            f"Val Loss: {val_loss} - "
            f"Val F0.5: {val_score}"
        )

        # Save Best Model
        if val_score > best_val_score:
            print(
                f"Validation score improved from {best_val_score} to {val_score}. Saving model..."
            )
            if val_score > BASELINE_SCORE:
                print(f"Note: Current score exceeds baseline of {BASELINE_SCORE}.")

            best_val_score = val_score
            early_stopping_counter = 0

            save_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
            torch.save(model.state_dict(), save_path)
        else:
            early_stopping_counter += 1
            print(
                f"No improvement. Early stopping counter: {early_stopping_counter}/{Config.PATIENCE + 2}"
            )

        # Early Stopping
        # We add a small buffer (+2) to the scheduler patience for actual stopping
        if early_stopping_counter >= (Config.PATIENCE + 2):
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation F0.5 Score: {best_val_score}")
