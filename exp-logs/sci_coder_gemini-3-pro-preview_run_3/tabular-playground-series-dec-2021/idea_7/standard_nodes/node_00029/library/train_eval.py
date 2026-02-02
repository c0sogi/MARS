import copy
import torch
import torch.nn as nn
from library.config import Config


class EarlyStopping:
    """
    Early stopping to stop the training when the metric does not improve after
    certain epochs. Uses copy.deepcopy to secure the best model weights.
    """

    def __init__(
        self,
        patience=Config.EARLY_STOPPING_PATIENCE,
        path=Config.MODEL_CHECKPOINT_PATH,
        verbose=False,
    ):
        self.patience = patience
        self.path = path
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_wts = None

    def __call__(self, val_acc, model):
        # We are maximizing accuracy
        score = val_acc

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_acc, model)
        elif score <= self.best_score:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_acc, model)
            self.counter = 0

    def save_checkpoint(self, val_acc, model):
        """Saves model when validation metric improves."""
        if self.verbose:
            print(
                f"Validation accuracy increased ({self.best_score} --> {val_acc}).  Saving model ..."
            )
        # Deep copy the model state dict to ensure it's not mutated later
        self.best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(model.state_dict(), self.path)


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def train_model(
    model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device
):
    """
    Orchestrates the training process including early stopping and learning rate scheduling.
    """
    early_stopping = EarlyStopping(
        patience=Config.EARLY_STOPPING_PATIENCE, path=Config.MODEL_CHECKPOINT_PATH
    )

    print("Starting Training...")

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        # Step the scheduler based on validation accuracy (max mode expected)
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_acc)
            else:
                scheduler.step()

        print(
            f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.6f}, Train Acc: {train_acc:.6f}, Val Loss: {val_loss:.6f}"
        )
        # Print full precision as requested
        print(f"Validation Accuracy: {val_acc}")

        # Check early stopping
        early_stopping(val_acc, model)

        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    # Load best model weights before returning
    if early_stopping.best_model_wts is not None:
        print(
            f"Loading best model weights with Validation Accuracy: {early_stopping.best_score}"
        )
        model.load_state_dict(early_stopping.best_model_wts)

    return model, early_stopping.best_score
