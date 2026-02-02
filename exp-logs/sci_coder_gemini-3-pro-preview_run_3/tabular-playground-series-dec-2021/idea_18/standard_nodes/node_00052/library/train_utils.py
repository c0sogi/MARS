import torch
import torch.nn as nn
import torch.optim as optim
import copy
import os
from library.config import Config


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
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
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


class Trainer:
    """
    Helper class to manage training, validation, and early stopping.
    """

    def __init__(self, model):
        self.model = model
        self.device = Config.DEVICE
        self.model.to(self.device)

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Scheduler: Cosine Annealing
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=Config.EPOCHS, eta_min=0
        )

    def fit(self, train_loader, val_loader):
        """
        Runs the training pipeline with Early Stopping.
        """
        best_val_acc = 0.0
        best_model_wts = copy.deepcopy(self.model.state_dict())
        patience = Config.PATIENCE
        patience_counter = 0

        print(f"Starting training on {self.device} for {Config.EPOCHS} epochs...")

        for epoch in range(Config.EPOCHS):
            # Train
            train_loss, train_acc = train_one_epoch(
                self.model, train_loader, self.criterion, self.optimizer, self.device
            )

            # Validate
            val_loss, val_acc = validate(
                self.model, val_loader, self.criterion, self.device
            )

            # Step Scheduler
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]

            # Print metrics (Full precision as requested)
            print(
                f"Epoch {epoch+1}/{Config.EPOCHS} | LR: {current_lr} | Train Loss: {train_loss} Acc: {train_acc} | Val Loss: {val_loss} Acc: {val_acc}"
            )

            # Early Stopping Logic
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_wts = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
                # Save checkpoint immediately
                torch.save(best_model_wts, Config.MODEL_CHECKPOINT_PATH)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break

        print(f"Training complete. Best Val Acc: {best_val_acc}")

        # Restore best model weights
        self.model.load_state_dict(best_model_wts)
        return self.model
