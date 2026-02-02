import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from library import config, utils, data, model


def train_one_epoch(net, dataloader, optimizer, criterion, device):
    """
    Executes one training epoch.

    Args:
        net: The neural network model.
        dataloader: The training data loader.
        optimizer: The optimizer.
        criterion: The loss function.
        device: The device (CPU or CUDA) to run on.

    Returns:
        float: The average training loss for the epoch.
    """
    net.train()
    running_loss = 0.0
    total_samples = 0

    for batch in dataloader:
        continuous = batch["continuous"].to(device)
        sequence = batch["sequence"].to(device)
        targets = batch["target"].to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = net(continuous, sequence)
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Accumulate loss
        batch_size = continuous.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

    avg_loss = running_loss / total_samples if total_samples > 0 else 0.0
    return avg_loss


def validate(net, dataloader, device):
    """
    Evaluates the model on the validation set.

    Args:
        net: The neural network model.
        dataloader: The validation data loader.
        device: The device to run on.

    Returns:
        float: The Area Under the ROC Curve (AUC) score.
    """
    net.eval()
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for batch in dataloader:
            continuous = batch["continuous"].to(device)
            sequence = batch["sequence"].to(device)
            targets = batch["target"]  # Keep targets on CPU for metric calculation

            outputs = net(continuous, sequence)
            preds = torch.sigmoid(outputs)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())

    auc = utils.calculate_auc(all_targets, all_preds)
    return auc


def predict(net, dataloader, device):
    """
    Generates predictions for the test set.

    Args:
        net: The neural network model.
        dataloader: The test data loader.
        device: The device to run on.

    Returns:
        np.ndarray: Flattened array of predicted probabilities.
    """
    net.eval()
    all_preds = []

    with torch.no_grad():
        for batch in dataloader:
            continuous = batch["continuous"].to(device)
            sequence = batch["sequence"].to(device)

            outputs = net(continuous, sequence)
            preds = torch.sigmoid(outputs)

            all_preds.extend(preds.cpu().numpy())

    return np.array(all_preds).flatten()


def run_training():
    """
    Main execution function.
    - Sets seeds for reproducibility.
    - Loads data.
    - Initializes the HybridNetwork, Optimizer, and Scheduler.
    - Runs the training loop for the configured number of epochs.
    - Saves the model with the best Validation AUC.
    - Generates the submission file using the best model.
    """
    # 1. Setup
    utils.set_seed(config.SEED)
    device = torch.device(config.DEVICE)
    print(f"Using device: {device}")

    # 2. Data Loading
    print("Loading data...")
    train_loader, val_loader, test_loader = data.get_dataloaders(
        batch_size=config.BATCH_SIZE, load_cached_data=True
    )

    # 3. Model Initialization
    print("Initializing model...")
    net = model.HybridNetwork().to(device)

    # 4. Optimization
    optimizer = optim.AdamW(
        net.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=config.LR_STEP_SIZE, gamma=config.LR_GAMMA
    )

    criterion = nn.BCEWithLogitsLoss()

    # 5. Training Loop
    best_auc = 0.0
    print(f"Starting training for {config.EPOCHS} epochs...")

    for epoch in range(1, config.EPOCHS + 1):
        # Train Phase
        train_loss = train_one_epoch(net, train_loader, optimizer, criterion, device)

        # Validation Phase
        val_auc = validate(net, val_loader, device)

        # Scheduler Step
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Logging (Printing full precision for metrics)
        print(
            f"Epoch {epoch} | LR: {current_lr} | Train Loss: {train_loss} | Val AUC: {val_auc}"
        )

        # Save Best Model
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(net.state_dict(), config.MODEL_SAVE_PATH)

    print(f"Training finished. Best Validation AUC: {best_auc}")

    # 6. Submission Generation
    print("Generating submission with best model...")

    # Load best model weights
    net.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))

    # Generate predictions
    predictions = predict(net, test_loader, device)

    # Load Test Metadata to ensure correct ID mapping
    # The test_loader is constructed based on test_metadata order
    test_meta = pd.read_csv(config.TEST_META_PATH)

    submission = pd.DataFrame({"id": test_meta["id"], "target": predictions})

    # Save submission
    submission.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {config.SUBMISSION_PATH}")
