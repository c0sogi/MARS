import os
import torch
import torch.nn as nn
import torch.optim as optim
from library.config import (
    DEVICE,
    EPOCHS,
    BATCH_SIZE,
    LEARNING_RATE,
    POS_WEIGHT,
    CHECKPOINT_DIR,
    SEED,
    NUM_WORKERS,
)
from library.dataset import get_dataloaders
from library.model import ParallelDilatedCNN
from library.utils import calculate_fbeta, set_seed


def train_epoch(model, loader, optimizer, criterion, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0

    for batch_idx, (data, target, _) in enumerate(loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(loader)
    return avg_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and F0.5 score.
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target, _ in loader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            loss = criterion(output, target)
            running_loss += loss.item()

            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(output)

            # Binarize with default threshold 0.5 for monitoring
            preds = (probs > 0.5).float()

            # Move to CPU to accumulate for metric calculation
            all_preds.append(preds.cpu())
            all_targets.append(target.cpu())

    avg_loss = running_loss / len(loader)

    # Concatenate all batches
    if len(all_preds) > 0:
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        f05_score = calculate_fbeta(all_preds, all_targets, beta=0.5)
    else:
        f05_score = 0.0

    return avg_loss, f05_score


def run_training(
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    lr=LEARNING_RATE,
    pos_weight_val=POS_WEIGHT,
    patience=5,
    load_cached_data=True,
):
    """
    Main training routine.
    """
    # Reproducibility
    set_seed(SEED)

    # Data
    train_loader, val_loader, _ = get_dataloaders(
        batch_size=batch_size,
        num_workers=NUM_WORKERS,
        load_cached_data=load_cached_data,
    )

    # Model
    model = ParallelDilatedCNN().to(DEVICE)

    # Optimization
    # pos_weight must be a tensor
    pos_weight_tensor = torch.tensor([pos_weight_val], device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Tracking
    best_val_score = -1.0
    patience_counter = 0
    best_model_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")

    print(f"Starting training on {DEVICE}...")
    print(
        f"Configuration: Epochs={epochs}, Batch Size={batch_size}, LR={lr}, Pos Weight={pos_weight_val}"
    )

    for epoch in range(1, epochs + 1):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE)

        # Validate
        val_loss, val_score = validate(model, val_loader, criterion, DEVICE)

        print(
            f"Epoch {epoch}: Train Loss: {train_loss}, Val Loss: {val_loss}, Val F0.5: {val_score}"
        )

        # Checkpoint & Early Stopping
        if val_score > best_val_score:
            best_val_score = val_score
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved to {best_model_path}")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch} epochs.")
            break

    print(f"Training completed. Best Validation F0.5 Score: {best_val_score}")
    return best_val_score
