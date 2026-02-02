import time
import copy
import numpy as np
import torch
from sklearn.metrics import log_loss, accuracy_score
from library import config


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    Saves the best model state in memory and optionally to a file.
    """

    def __init__(self, patience=7, verbose=False, delta=0, path=None, trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            verbose (bool): If True, prints a message for each validation loss improvement.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            path (str): Path for the checkpoint to be saved to.
            trace_func (function): trace print function.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.best_model_state = None

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(
                    f"EarlyStopping counter: {self.counter} out of {self.patience}"
                )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(
                f"Validation loss decreased ({self.val_loss_min} --> {val_loss}).  Saving model ..."
            )

        # Strictly preserve the best weights using deepcopy
        self.best_model_state = copy.deepcopy(model.state_dict())
        self.val_loss_min = val_loss

        if self.path:
            torch.save(self.best_model_state, self.path)


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Trains the model for one epoch.

    Args:
        model: PyTorch model.
        dataloader: Training dataloader.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device to train on.

    Returns:
        avg_loss: Average loss for the epoch.
        avg_acc: Average accuracy for the epoch.
    """
    model.train()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    for inputs, inc_angles, targets in dataloader:
        inputs = inputs.to(device)
        inc_angles = inc_angles.to(device)
        targets = targets.to(device).unsqueeze(
            1
        )  # Ensure target shape matches output (B, 1)

        optimizer.zero_grad()

        outputs = model(inputs, inc_angles)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

        # Store predictions for metrics
        probs = torch.sigmoid(outputs).detach().cpu().numpy()
        all_preds.extend(probs)
        all_targets.extend(targets.cpu().numpy())

    avg_loss = running_loss / len(dataloader.dataset)

    # Calculate accuracy
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    preds_binary = (all_preds > 0.5).astype(int)
    avg_acc = accuracy_score(all_targets, preds_binary)

    return avg_loss, avg_acc


def validate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model: PyTorch model.
        dataloader: Validation dataloader.
        criterion: Loss function.
        device: Device to evaluate on.

    Returns:
        avg_loss: Average loss.
        avg_acc: Average accuracy.
        avg_log_loss: Average Log Loss (sklearn).
    """
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for inputs, inc_angles, targets in dataloader:
            inputs = inputs.to(device)
            inc_angles = inc_angles.to(device)
            targets = targets.to(device).unsqueeze(1)

            outputs = model(inputs, inc_angles)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)

            probs = torch.sigmoid(outputs).detach().cpu().numpy()
            all_preds.extend(probs)
            all_targets.extend(targets.cpu().numpy())

    avg_loss = running_loss / len(dataloader.dataset)

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Calculate metrics
    # Sklearn Log Loss (clip to avoid inf, though BCEWithLogitsLoss is the main optimization target)
    avg_log_loss = log_loss(all_targets, all_preds, labels=[0, 1], eps=1e-15)

    preds_binary = (all_preds > 0.5).astype(int)
    avg_acc = accuracy_score(all_targets, preds_binary)

    return avg_loss, avg_acc, avg_log_loss


def run_training(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    num_epochs,
    early_stopping,
):
    """
    Orchestrates the training process for a specific fold/split.

    Args:
        model: The neural network model.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        criterion: Loss function.
        optimizer: Optimizer.
        scheduler: Learning rate scheduler.
        device: Device to run on.
        num_epochs: Maximum number of epochs.
        early_stopping: EarlyStopping instance.

    Returns:
        model: The model with the best weights loaded.
        history: Dictionary containing training history.
    """
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_log_loss": [],
    }

    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, val_log_loss = validate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_log_loss"].append(val_log_loss)

        # Scheduler step (ReduceLROnPlateau expects a metric)
        if scheduler:
            scheduler.step(val_loss)

        epoch_time = time.time() - epoch_start

        # Print metrics with full precision
        print(f"Epoch {epoch+1}/{num_epochs} - Time: {epoch_time:.2f}s")
        print(f"Train Loss: {train_loss}, Train Acc: {train_acc}")
        print(f"Val Loss: {val_loss}, Val Acc: {val_acc}, Val Log Loss: {val_log_loss}")

        # Early Stopping check
        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    total_time = time.time() - start_time
    print(f"Training complete in {total_time // 60:.0f}m {total_time % 60:.0f}s")

    # Load best model weights
    if early_stopping.best_model_state is not None:
        print("Loading best model weights...")
        model.load_state_dict(early_stopping.best_model_state)

    return model, history
