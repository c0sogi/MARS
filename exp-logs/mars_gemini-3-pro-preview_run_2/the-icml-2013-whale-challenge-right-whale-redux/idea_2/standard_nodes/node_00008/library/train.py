import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from library.config import Config
from library.utils import seed_everything, calculate_roc_auc, MetricMonitor
from library.dataset import get_dataloaders
from library.model import WhaleEfficientNet


def mixup_data(x, y, alpha=1.0, device="cpu"):
    """
    Applies Mixup augmentation to the input batch.
    Returns mixed inputs, pairs of targets, and the mixing coefficient lambda.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Calculates the loss for Mixup augmentation.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_one_epoch(model, train_loader, criterion, optimizer, device, config):
    """
    Trains the model for one epoch using Mixup.
    """
    model.train()
    loss_monitor = MetricMonitor()

    for batch_idx, (data, target, _) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device).view(-1, 1)

        # Apply Mixup
        data, target_a, target_b, lam = mixup_data(
            data, target, config.MIXUP_ALPHA, device
        )

        optimizer.zero_grad()
        output = model(data)

        loss = mixup_criterion(criterion, output, target_a, target_b, lam)
        loss.backward()
        optimizer.step()

        loss_monitor.update("loss", loss.item())

    return loss_monitor.get_avg("loss")


def validate_one_epoch(model, val_loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and ROC AUC score.
    """
    model.eval()
    loss_monitor = MetricMonitor()
    predictions = []
    targets = []

    with torch.no_grad():
        for data, target, _ in val_loader:
            data = data.to(device)
            target = target.to(device).view(-1, 1)

            output = model(data)
            loss = criterion(output, target)

            loss_monitor.update("loss", loss.item())

            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(output)

            # Flatten and store for AUC calculation
            predictions.extend(probs.cpu().numpy().flatten())
            targets.extend(target.cpu().numpy().flatten())

    avg_loss = loss_monitor.get_avg("loss")
    auc = calculate_roc_auc(np.array(targets), np.array(predictions))

    return avg_loss, auc


def run_training():
    """
    Main function to execute the training pipeline:
    1. Setup environment and data.
    2. Train loop with Early Stopping.
    3. Inference on Test set.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = Config.DEVICE
    print(f"Device: {device}")

    # Load Data
    train_loader, val_loader, test_loader = get_dataloaders(Config)

    # Initialize Model
    model = WhaleEfficientNet(Config)
    model = model.to(device)

    # Optimization
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.NUM_EPOCHS)

    criterion = nn.BCEWithLogitsLoss()

    # 2. Training Loop
    best_auc = 0.0
    patience_counter = 0

    print("Starting training...")

    for epoch in range(1, Config.NUM_EPOCHS + 1):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, Config
        )
        val_loss, val_auc = validate_one_epoch(model, val_loader, criterion, device)

        # Update learning rate
        scheduler.step()

        # Print metrics (full precision)
        print(f"Epoch {epoch}/{Config.NUM_EPOCHS}")
        print(f"Train Loss: {train_loss}")
        print(f"Val Loss: {val_loss}")
        print(f"Val AUC: {val_auc}")

        # Early Stopping Check
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
            print(f"New best model saved with AUC: {best_auc}")
        else:
            patience_counter += 1
            print(
                f"Early stopping counter: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE}"
            )

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    # 3. Inference
    print("Starting inference on test set...")

    # Load best model
    model = WhaleEfficientNet(Config)
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()

    test_clips = []
    test_probs = []

    with torch.no_grad():
        for data, _, clips in test_loader:
            data = data.to(device)

            # Forward pass
            output = model(data)
            probs = torch.sigmoid(output).cpu().numpy().flatten()

            test_clips.extend(clips)
            test_probs.extend(probs)

    # Save Submission
    submission_df = pd.DataFrame({"clip": test_clips, "probability": test_probs})

    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
