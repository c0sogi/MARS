import os
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from library.config import Config
from library.model_utils import DeepSupervisedHybridModel


def set_seed(seed):
    """Sets the random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # As per instructions, we disable strict determinism for performance
    # but keep seeds fixed.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def train_one_epoch(
    model, dataloader, optimizer, criterion, device, epoch, total_epochs
):
    """
    Runs one epoch of training with Annealed Multi-Loss Optimization.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # Calculate Annealed Lambda for Auxiliary Loss
    # Linearly decay from START to END
    # Formula: lambda_t = Start - (Start - End) * (epoch / total_epochs)
    # Note: epoch is 0-indexed, so at epoch 0, progress is 0.
    if total_epochs > 0:
        progress = epoch / total_epochs
    else:
        progress = 1.0

    lambda_t = (
        Config.AUX_LOSS_WEIGHT_START
        - (Config.AUX_LOSS_WEIGHT_START - Config.AUX_LOSS_WEIGHT_END) * progress
    )
    lambda_t = max(0.0, lambda_t)

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        # Forward Pass
        # Returns (primary_logits, aux_logits)
        prim_logits, aux_logits = model(inputs)

        # Multi-Loss Calculation
        loss_prim = criterion(prim_logits, labels)

        # Aux logits might be None if the model structure changes, but per design it returns tensor
        if aux_logits is not None:
            loss_aux = criterion(aux_logits, labels)
            loss = loss_prim + lambda_t * loss_aux
        else:
            loss = loss_prim

        # Backward Pass
        loss.backward()
        optimizer.step()

        # Metrics
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(prim_logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward (Aux ignored in inference/val)
            prim_logits, _ = model(inputs)
            loss = criterion(prim_logits, labels)

            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(prim_logits, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_loss = val_loss / val_total
    val_acc = val_correct / val_total

    return val_loss, val_acc


def run_training(train_loader, val_loader):
    """
    Main driver function for the training pipeline.
    """
    set_seed(Config.SEED)
    device = Config.DEVICE

    # Determine Input Dimension from Data
    dummy_x, _ = next(iter(train_loader))
    input_dim = dummy_x.shape[1]

    print(
        f"Initializing DeepSupervisedHybridModel with Input Dim: {input_dim}, Hidden Dim: 512"
    )
    model = DeepSupervisedHybridModel(
        input_dim, num_classes=Config.NUM_CLASSES, hidden_dim=512
    ).to(device)

    # Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
    )

    criterion = nn.CrossEntropyLoss()

    # Early Stopping State
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    early_stop_counter = 0

    print(f"Starting training for {Config.EPOCHS} epochs on {device}...")

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch, Config.EPOCHS
        )

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Print Metrics (Full Precision)
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Loss: {train_loss} Acc: {train_acc} | "
            f"Val Loss: {val_loss} Val Acc: {val_acc}"
        )

        # Scheduler Step
        scheduler.step(val_acc)

        # Early Stopping Check
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= Config.EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    # Restore Best Model
    print(f"Training complete. Best Validation Accuracy: {best_acc}")
    model.load_state_dict(best_model_wts)

    # Save checkpoint
    save_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Best model saved to {save_path}")

    return model


def predict(model, test_loader, test_ids):
    """
    Generates predictions for the test set and saves to submission.csv.
    """
    device = Config.DEVICE
    model.eval()
    predictions = []

    print("Generating predictions on test set...")

    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs.to(device)

            # Forward pass (ignore aux)
            prim_logits, _ = model(inputs)
            _, preds = torch.max(prim_logits, 1)

            # Map 0-6 back to original 1-7 class labels
            preds = preds + 1
            predictions.extend(preds.cpu().numpy())

    # Create Submission DataFrame
    df_sub = pd.DataFrame({"Id": test_ids, "Cover_Type": predictions})

    # Save
    df_sub.to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {Config.SUBMISSION_FILE}")
