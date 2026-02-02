import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from library.config import Config
from library.dataset import InkDataset
from library.model import InkUNet
from library.utils import optimize_threshold, generate_submission


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        batch_size = inputs.size(0)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size if dataset_size > 0 else 0.0
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set and optimizes the decision threshold.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    val_preds = []
    val_labels = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            batch_size = inputs.size(0)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Apply sigmoid to get probabilities for threshold search
            probs = torch.sigmoid(outputs).cpu().numpy()
            labels = targets.cpu().numpy()

            val_preds.append(probs)
            val_labels.append(labels)

    avg_loss = running_loss / dataset_size if dataset_size > 0 else 0.0

    # Find optimal threshold for F0.5 score
    best_thresh, best_score = optimize_threshold(val_preds, val_labels)

    return avg_loss, best_thresh, best_score


def run_training(limit=None, epochs=None):
    """
    Main execution function for the training pipeline.

    Args:
        limit (int, optional): Limit dataset size for debugging.
        epochs (int, optional): Override default number of epochs.
    """
    # 1. Setup
    Config.setup()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    num_epochs = epochs if epochs is not None else Config.EPOCHS

    # 2. Data Loading
    print("Initializing datasets...")
    train_dataset = InkDataset(mode="train", limit=limit)
    val_dataset = InkDataset(mode="val", limit=limit)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Initialization
    print("Initializing model...")
    model = InkUNet(z_dim=Config.Z_DIM).to(device)

    # 4. Loss and Optimizer
    # Use positive class weight to handle imbalance
    pos_weight = torch.tensor([Config.POS_WEIGHT]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    # 5. Training Loop
    best_score = -1.0
    best_threshold_global = 0.5
    patience = 5
    patience_counter = 0
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    print(f"Starting training for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_loss, threshold, score = validate(model, val_loader, criterion, device)

        elapsed = time.time() - start_time

        print(f"Epoch {epoch+1}/{num_epochs} - Time: {elapsed:.2f}s")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss:   {val_loss:.6f}")
        print(f"  Best Thresh: {threshold}")
        print(f"  F0.5 Score: {score}")

        # Checkpoint & Early Stopping
        if score > best_score:
            best_score = score
            best_threshold_global = threshold
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print("  New best model saved!")
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    # 6. Submission
    print("\nGenerating submission...")
    if os.path.exists(best_model_path):
        print(f"Loading best model from {best_model_path} (Score: {best_score})")
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        print("Warning: No best model found. Using current model state.")

    print(f"Using optimal threshold: {best_threshold_global}")
    generate_submission(model, device, threshold=best_threshold_global)
    print("Done.")
