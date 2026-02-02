import torch
import torch.nn as nn
import torch.optim as optim
import os
import copy
from library.config import Config
from library.utils import seed_everything, EarlyStopping
from library.data_loader import get_dataloaders
from library.model import DeepParallelDCNResNet


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def evaluate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def run_training(
    load_cached_data=True,
    epochs=Config.EPOCHS,
    batch_size=Config.BATCH_SIZE,
    learning_rate=Config.LEARNING_RATE,
    weight_decay=Config.WEIGHT_DECAY,
    patience=Config.PATIENCE,
    factor=Config.FACTOR,
    hidden_dim=Config.HIDDEN_DIM,
    dropout=Config.DROPOUT,
    num_blocks=Config.NUM_BLOCKS,
):
    """
    Main training function. Initializes data, model, optimizer, and runs the training loop.
    """
    # Set seed for reproducibility
    seed_everything(Config.SEED)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load Data
    print("Loading data...")
    train_loader, val_loader, _ = get_dataloaders(
        batch_size=batch_size, load_cached_data=load_cached_data
    )

    # Determine input dimension from a single batch
    sample_inputs, _ = next(iter(train_loader))
    input_dim = sample_inputs.shape[1]
    print(f"Input Dimension: {input_dim}")

    # Initialize Model
    model = DeepParallelDCNResNet(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_blocks=num_blocks,
        dropout=dropout,
        num_classes=Config.NUM_CLASSES,
    ).to(device)

    # Optimizer (AdamW)
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # Scheduler (ReduceLROnPlateau)
    # Note: Scheduler steps based on validation accuracy (mode='max')
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=factor, patience=patience
    )

    # Loss Function
    criterion = nn.CrossEntropyLoss()

    # Early Stopping
    # Note: EarlyStopping usually monitors validation loss (minimization)
    early_stopping = EarlyStopping(patience=10, verbose=True)

    print("Starting training...")

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        # Print full precision metrics
        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss} | Train Acc: {train_acc} | "
            f"Val Loss: {val_loss} | Val Acc: {val_acc}"
        )

        # Step Scheduler (monitoring accuracy)
        scheduler.step(val_acc)

        # Step Early Stopping (monitoring loss)
        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    # Load best model weights
    if early_stopping.best_model_state is not None:
        print("Loading best model weights from EarlyStopping...")
        model.load_state_dict(early_stopping.best_model_state)

    # Save the best model
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    model_save_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Best model saved to {model_save_path}")

    return model
