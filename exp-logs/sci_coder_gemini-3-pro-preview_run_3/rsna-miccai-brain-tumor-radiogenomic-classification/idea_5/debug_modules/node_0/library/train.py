import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import roc_auc_score
from library.config import Config, seed_everything
from library.dataset import get_dataloader
from library.model import VFPNet


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Training loop for one epoch.
    """
    model.train()
    running_loss = 0.0
    count = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Ensure shapes match for BCEWithLogitsLoss
        # outputs: (B, 1) -> squeeze -> (B,)
        # targets: (B,)
        loss = criterion(outputs.squeeze(1), targets)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        count += inputs.size(0)

    epoch_loss = running_loss / count if count > 0 else 0.0
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Validation loop. Returns average loss and AUC.
    """
    model.eval()
    running_loss = 0.0
    count = 0

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)

            # Calculate loss
            loss = criterion(outputs.squeeze(1), targets)
            running_loss += loss.item() * inputs.size(0)
            count += inputs.size(0)

            # Store for AUC calculation
            # Apply sigmoid to logits to get probabilities
            probs = torch.sigmoid(outputs).squeeze(1)

            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(probs.cpu().numpy())

    epoch_loss = running_loss / count if count > 0 else 0.0

    # Calculate AUC
    # Handle edge case where only one class is present in the batch
    try:
        if len(np.unique(all_targets)) > 1:
            epoch_auc = roc_auc_score(all_targets, all_preds)
        else:
            epoch_auc = 0.5
    except Exception:
        epoch_auc = 0.5

    return epoch_loss, epoch_auc


def run_training():
    """
    Main function to orchestrate training.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 2. Data Loaders
    print("Initializing DataLoaders...")
    train_loader = get_dataloader("train", shuffle=True)
    val_loader = get_dataloader("val", shuffle=False)

    # 3. Model
    print("Initializing VFPNet...")
    model = VFPNet(num_classes=1, pretrained=True)
    model = model.to(device)

    # 4. Optimizer & Loss
    # Instructions: Adam, lr=1e-4, weight_decay=0.0
    optimizer = optim.Adam(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    criterion = nn.BCEWithLogitsLoss()

    # 5. Training Loop
    best_auc = 0.0
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    # Early stopping parameters
    patience = 5
    counter = 0

    print(f"Starting training for {Config.NUM_EPOCHS} epochs...")

    for epoch in range(1, Config.NUM_EPOCHS + 1):
        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        print(f"Epoch {epoch}/{Config.NUM_EPOCHS}")
        print(f"Train Loss: {train_loss}")
        print(f"Val Loss: {val_loss}")
        print(f"Val AUC: {val_auc}")

        # Checkpointing
        if val_auc > best_auc:
            print(
                f"Validation AUC improved from {best_auc} to {val_auc}. Saving model..."
            )
            best_auc = val_auc
            torch.save(model.state_dict(), best_model_path)
            counter = 0  # Reset patience
        else:
            counter += 1
            print(f"Validation AUC did not improve. Patience: {counter}/{patience}")

        # Early Stopping
        if counter >= patience:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation AUC: {best_auc}")
    print(f"Best model saved to: {best_model_path}")
