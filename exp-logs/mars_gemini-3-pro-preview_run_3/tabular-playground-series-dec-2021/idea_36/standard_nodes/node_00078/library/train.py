import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import (
    seed_everything,
    get_model_copy,
    save_checkpoint,
    load_checkpoint,
)
from library.data_processing import get_dataloaders
from library.model import DeepSupervisedNet


def train_one_epoch(model, loader, criterion, optimizer, device, config):
    """
    Performs one epoch of training with Deep Supervision.
    Loss = Primary_Loss + aux_weight * Aux_Loss
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Forward pass returns tuple (primary_logits, aux_logits)
        primary_logits, aux_logits = model(inputs)

        # Calculate Primary Loss
        loss_primary = criterion(primary_logits, targets)

        # Calculate Auxiliary Loss (if aux_logits is not None)
        loss_aux = 0.0
        if aux_logits is not None:
            loss_aux = criterion(aux_logits, targets)

        # Combined Loss
        loss = loss_primary + (config.aux_loss_weight * loss_aux)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

        # Calculate accuracy on primary head only
        _, predicted = torch.max(primary_logits, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device):
    """
    Validates the model. Uses only the primary head for metrics.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            primary_logits, _ = model(inputs)

            # We track loss for the scheduler (using primary loss usually, or combined)
            # Standard practice for validation is to evaluate the primary objective.
            loss = criterion(primary_logits, targets)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(primary_logits, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    val_loss = running_loss / total
    val_acc = correct / total
    return val_loss, val_acc


def generate_submission(model, loader, test_ids, device, config):
    """
    Generates predictions for the test set and saves to CSV.
    """
    print("Generating submission...")
    model.eval()
    predictions = []

    with torch.no_grad():
        for inputs in loader:
            inputs = inputs.to(device)
            primary_logits, _ = model(inputs)
            _, predicted = torch.max(primary_logits, 1)
            predictions.extend(predicted.cpu().numpy())

    # Map 0-indexed predictions back to 1-7 range
    predictions = np.array(predictions) + 1

    submission_df = pd.DataFrame({"Id": test_ids, "Cover_Type": predictions})

    # Ensure output directory exists
    os.makedirs(os.path.dirname(config.submission_path), exist_ok=True)
    submission_df.to_csv(config.submission_path, index=False)
    print(f"Submission saved to {config.submission_path}")


def run_training():
    # 1. Configuration & Setup
    config = Config()
    seed_everything(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Data Loading
    print("Loading data...")
    train_loader, val_loader, test_loader, test_ids, input_dim = get_dataloaders(config)

    # 3. Model Initialization
    # Number of classes is 7 (mapped to 0-6 internally)
    # The config defines NUM_CLASSES = 7, but PyTorch needs 0-6 indices.
    # The model output dimension should be 7 to accommodate 0-6 indices if we treated them sparsely,
    # but strictly speaking we have 7 classes.
    # The dataset processing subtracts 1 from targets, so targets are 0..6.
    # So num_classes for the model output layer should be 7.
    from library.config import NUM_CLASSES

    model = DeepSupervisedNet(input_dim, NUM_CLASSES, config)
    model = model.to(device)

    # 4. Optimization
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    # Scheduler: Reduce LR when validation loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
        min_lr=config.min_lr,
    )

    # 5. Training Loop
    best_acc = 0.0
    patience_counter = 0
    best_model_path = os.path.join(config.working_dir, "best_model.pth")

    print(f"Starting training for {config.epochs} epochs...")

    for epoch in range(config.epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, config
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Update Scheduler based on Validation Loss
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch+1}/{config.epochs} - "
            f"Train Loss: {train_loss:.6f}, Train Acc: {train_acc:.6f} - "
            f"Val Loss: {val_loss:.6f}, Val Acc: {val_acc:.6f}"
        )

        # Early Stopping Check
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            # Save best model
            print(f"New best accuracy: {best_acc:.6f}. Saving model...")
            state = {
                "state_dict": get_model_copy(model),
                "best_acc": best_acc,
                "epoch": epoch,
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(state, best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= config.early_stopping_patience:
                print(
                    f"Early stopping triggered after {patience_counter} epochs without improvement."
                )
                break

    # 6. Final Evaluation & Submission
    print("Training complete. Loading best model for submission...")

    # Load best weights
    if os.path.exists(best_model_path):
        load_checkpoint(best_model_path, model, device)
    else:
        print("Warning: No checkpoint found. Using current model weights.")

    generate_submission(model, test_loader, test_ids, device, config)


if __name__ == "__main__":
    run_training()
