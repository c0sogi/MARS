import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.utils import seed_everything, get_optimizer_grouped_parameters
from library.dataset import get_dataloaders
from library.model import HybridSwiGLUNet


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = len(dataloader.dataset)

    for batch in dataloader:
        # Move data to device
        continuous = batch["continuous"].to(device)
        categorical = batch["categorical"].to(device)
        targets = batch["target"].to(device).unsqueeze(1)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(continuous, categorical)
        loss = criterion(outputs, targets)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Update running loss
        running_loss += loss.item() * continuous.size(0)

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def evaluate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and ROC AUC score.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = len(dataloader.dataset)

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for batch in dataloader:
            continuous = batch["continuous"].to(device)
            categorical = batch["categorical"].to(device)
            targets = batch["target"].to(device).unsqueeze(1)

            outputs = model(continuous, categorical)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * continuous.size(0)

            # Store predictions and targets for AUC calculation
            all_targets.append(targets.cpu().numpy())
            all_preds.append(outputs.cpu().numpy())

    epoch_loss = running_loss / dataset_size

    # Concatenate all batches
    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)

    # Calculate AUC
    try:
        auc_score = roc_auc_score(all_targets, all_preds)
    except ValueError:
        # Handle edge case where only one class is present in validation (unlikely)
        auc_score = 0.5

    return epoch_loss, auc_score


def predict(model, dataloader, device):
    """
    Generates predictions for the test set.
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for batch in dataloader:
            continuous = batch["continuous"].to(device)
            categorical = batch["categorical"].to(device)

            outputs = model(continuous, categorical)
            all_preds.append(outputs.cpu().numpy())

    return np.concatenate(all_preds).flatten()


def run_training():
    """
    Main function to run the training pipeline.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Device: {device}")

    # 2. Data Loading
    print("Loading data...")
    train_loader, val_loader, test_loader, test_ids = get_dataloaders(
        load_cached_data=True
    )

    # 3. Model Initialization
    print("Initializing model...")
    model = HybridSwiGLUNet().to(device)

    # 4. Optimizer & Scheduler
    # Use strict decoupled weight decay
    param_groups = get_optimizer_grouped_parameters(
        model,
        weight_decay_group1=Config.WEIGHT_DECAY_GROUP1,
        weight_decay_group2=Config.WEIGHT_DECAY_GROUP2,
    )

    optimizer = torch.optim.AdamW(param_groups, lr=Config.LEARNING_RATE)

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=Config.SCHEDULER_STEP_SIZE, gamma=Config.SCHEDULER_GAMMA
    )

    # 5. Loss Function
    # Model outputs sigmoid probabilities, so we use BCELoss
    criterion = nn.BCELoss()

    # 6. Training Loop
    best_auc = 0.0
    print("Starting training...")

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_loss, val_auc = evaluate(model, val_loader, criterion, device)

        # Scheduler Step
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Log Metrics (Full Precision)
        print(
            f"Epoch {epoch + 1}/{Config.EPOCHS} | LR: {current_lr} | "
            f"Train Loss: {train_loss} | Val Loss: {val_loss} | Val AUC: {val_auc}"
        )

        # Save Best Model
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), Config.MODEL_PATH)
            print(f"New best model saved to {Config.MODEL_PATH} (AUC: {best_auc})")

    # 7. Inference
    print("\nTraining complete. Loading best model for inference...")

    # Load best weights
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))

    # Generate predictions
    predictions = predict(model, test_loader, device)

    # 8. Save Submission
    print("Saving submission...")
    submission_df = pd.DataFrame({"id": test_ids, "target": predictions})

    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
