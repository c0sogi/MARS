import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from library.config import (
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    PATIENCE,
    SCHEDULER_FACTOR,
    SCHEDULER_PATIENCE,
    MODEL_SAVE_PATH,
    DEVICE,
)
from library.utils import seed_everything
from library.data import get_dataloaders
from library.model import ParallelDCNResNet


def train_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for X_cont, X_bin, y in loader:
        X_cont = X_cont.to(device)
        X_bin = X_bin.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(X_cont, X_bin)
        loss = criterion(outputs, y)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Metrics
        running_loss += loss.item() * y.size(0)
        _, predicted = torch.max(outputs, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate_epoch(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for X_cont, X_bin, y in loader:
            X_cont = X_cont.to(device)
            X_bin = X_bin.to(device)
            y = y.to(device)

            outputs = model(X_cont, X_bin)
            loss = criterion(outputs, y)

            running_loss += loss.item() * y.size(0)
            _, predicted = torch.max(outputs, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def run_training():
    """
    Main training loop with Early Stopping and Scheduler.
    """
    # 1. Setup
    seed_everything()

    print("Loading data...")
    train_loader, val_loader, test_loader, input_info = get_dataloaders(
        load_cached_data=True
    )

    print(f"Initializing model on {DEVICE}...")
    model = ParallelDCNResNet(input_info).to(DEVICE)

    # 2. Optimization
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    # Scheduler: Reduce LR when validation accuracy plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=SCHEDULER_FACTOR,
        patience=SCHEDULER_PATIENCE,
        verbose=True,
    )

    # 3. Training Loop
    best_val_acc = -1.0
    best_model_state = None
    patience_counter = 0

    print(f"Starting training for {EPOCHS} epochs...")

    for epoch in range(EPOCHS):
        start_time = time.time()

        # Train and Validate
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, DEVICE
        )
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, DEVICE)

        # Step Scheduler
        scheduler.step(val_acc)

        # Early Stopping Check
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            improved = True
        else:
            patience_counter += 1
            improved = False

        elapsed = time.time() - start_time

        # Print Metrics (Full precision for Val Acc as requested)
        status = (
            "(New Best)" if improved else f"(Patience: {patience_counter}/{PATIENCE})"
        )
        print(
            f"Epoch {epoch+1}/{EPOCHS} | Time: {elapsed:.2f}s | "
            f"Train Loss: {train_loss:.6f} | Train Acc: {train_acc:.6f} | "
            f"Val Loss: {val_loss:.6f} | Val Acc: {val_acc} {status}"
        )

        if patience_counter >= PATIENCE:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    # 4. Save and Finish
    if best_model_state is not None:
        print(f"Saving best model (Val Acc: {best_val_acc}) to {MODEL_SAVE_PATH}")
        torch.save(best_model_state, MODEL_SAVE_PATH)
        # Reload best weights
        model.load_state_dict(best_model_state)
    else:
        print(
            "Warning: No best model state found (did training fail?). Saving current state."
        )
        torch.save(model.state_dict(), MODEL_SAVE_PATH)

    return model, test_loader
