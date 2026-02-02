import torch
import torch.nn as nn
import copy
import numpy as np
from tqdm import tqdm
from library.config import Config
from library.utils import save_checkpoint, log_metrics, get_logger

# Initialize logger
logger = get_logger()


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    Saves the best model state to disk and keeps a copy in memory.
    """

    def __init__(self, patience=Config.PATIENCE, delta=0, path="checkpoint.pth"):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            path (str): Path for the checkpoint to be saved to.
        """
        self.patience = patience
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float("inf")
        self.best_model_state = None

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        self.best_model_state = copy.deepcopy(model.state_dict())
        save_checkpoint(model, self.path)
        self.val_loss_min = val_loss

    def restore_best_weights(self, model):
        """Restores the best model weights from memory."""
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)
        return model


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch in dataloader:
        images = batch["image"].to(device)
        inc_angles = batch["inc_angle"].to(device)
        labels = batch["label"].to(device)

        batch_size = images.size(0)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images, inc_angles)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    # For accuracy calculation (optional, purely for logging context)
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            inc_angles = batch["inc_angle"].to(device)
            labels = batch["label"].to(device)

            batch_size = images.size(0)

            outputs = model(images, inc_angles)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Calculate accuracy
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += batch_size

    epoch_loss = running_loss / dataset_size
    epoch_acc = correct / total if total > 0 else 0.0

    return epoch_loss, epoch_acc


def train_fold(fold_idx, model, train_loader, val_loader, device):
    """
    Executes the full training loop for a single fold, including:
    - Optimizer setup (Adam)
    - Scheduler setup (ReduceLROnPlateau)
    - Early Stopping
    - Logging

    Returns:
        model: The model with the best weights loaded.
    """
    logger.info(f"Starting training for Fold {fold_idx}...")

    # 1. Setup Training Components
    criterion = nn.BCEWithLogitsLoss()

    # Revert to Adam (not AdamW) as per instructions
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    # Scheduler: ReduceLROnPlateau
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # Early Stopping
    checkpoint_path = Config.MODEL_CHECKPOINT_PATTERN.format(fold=fold_idx)
    early_stopping = EarlyStopping(patience=Config.PATIENCE, path=checkpoint_path)

    # 2. Training Loop
    for epoch in range(1, Config.NUM_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Log full precision metrics
        log_metrics(logger, epoch, train_loss, val_loss, val_acc)

        # Update Scheduler
        scheduler.step(val_loss)

        # Check Early Stopping
        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            logger.info("Early stopping triggered")
            break

    # 3. Restore Best Weights
    logger.info(
        f"Fold {fold_idx} finished. Restoring best weights from epoch with loss {early_stopping.val_loss_min}"
    )
    model = early_stopping.restore_best_weights(model)

    return model


def predict(model, dataloader, device):
    """
    Generates predictions for the test set.

    Returns:
        ids (list): List of image IDs.
        probs (np.array): Predicted probabilities (0-1).
    """
    model.eval()
    ids = []
    probs = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            inc_angles = batch["inc_angle"].to(device)
            batch_ids = batch["id"]

            outputs = model(images, inc_angles)
            batch_probs = torch.sigmoid(outputs).cpu().numpy().flatten()

            ids.extend(batch_ids)
            probs.extend(batch_probs)

    return ids, np.array(probs)
