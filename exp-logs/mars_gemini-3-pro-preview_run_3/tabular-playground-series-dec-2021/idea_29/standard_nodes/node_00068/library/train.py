import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from library.utils import seed_everything, get_device, generate_submission
from library.data_loader import get_dataloaders
from library.model import DeepParallelVectorDCNResNet


class EarlyStopping:
    """
    Early stops the training if validation accuracy doesn't improve after a given patience.
    Saves the best model weights using deepcopy.
    """

    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = copy.deepcopy(model.state_dict())
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = copy.deepcopy(model.state_dict())
            self.counter = 0


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    val_loss = running_loss / total
    val_acc = correct / total
    return val_loss, val_acc


def run_training(
    epochs=60,
    batch_size=4096,
    lr=1e-3,
    patience=10,
    data_dir="./metadata",
    cache_dir="./working/idea_29",
    submission_path="./submission/submission.csv",
):
    """
    Orchestrates the entire training pipeline.
    """
    # 1. Setup
    seed_everything(42)
    device = get_device()
    print(f"Device: {device}")

    # 2. Data
    # get_dataloaders returns: train_loader, val_loader, test_loader, test_ids, input_dim
    train_loader, val_loader, test_loader, test_ids, input_dim = get_dataloaders(
        load_cached_data=True,
        batch_size=batch_size,
        data_dir=data_dir,
        cache_dir=cache_dir,
    )

    # 3. Model
    model = DeepParallelVectorDCNResNet(
        input_dim=input_dim,
        num_classes=7,  # Cover_Type 1-7
        hidden_dim=512,
        num_cross_layers=3,
        num_res_blocks=4,
        dropout_rate=0.2,
    ).to(device)

    # 4. Optimizer & Scheduler
    criterion = nn.CrossEntropyLoss()
    # Decoupled Weight Decay (AdamW)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    # ReduceLROnPlateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    # 5. Early Stopping
    early_stopping = EarlyStopping(patience=patience)

    # 6. Training Loop
    print(f"Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Print full precision metrics
        print(
            f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss} Acc: {train_acc} | Val Loss: {val_loss} Acc: {val_acc}"
        )

        # Scheduler Step
        scheduler.step(val_acc)

        # Early Stopping Step
        early_stopping(val_acc, model)

        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    # 7. Restore Best Weights
    if early_stopping.best_model_state is not None:
        print(
            f"Loading best model weights with Validation Accuracy: {early_stopping.best_score}"
        )
        model.load_state_dict(early_stopping.best_model_state)

    # 8. Submission
    generate_submission(
        model, test_loader, test_ids, device, output_path=submission_path
    )
