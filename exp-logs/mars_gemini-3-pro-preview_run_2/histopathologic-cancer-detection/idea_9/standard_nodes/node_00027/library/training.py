import time
import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from library.config import Config
from library.models import ModelEMA


def set_seed(seed=42):
    """Sets the random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mixup_data(x, y, alpha=1.0, device="cuda"):
    """
    Applies Mixup augmentation to the batch.
    Returns mixed inputs and mixed targets.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index]
    # Mix targets directly (works with BCEWithLogitsLoss)
    mixed_y = lam * y + (1 - lam) * y[index]

    return mixed_x, mixed_y


def train_one_epoch(
    model, loader, optimizer, criterion, device, ema_model=None, mixup_alpha=0.0
):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    # Initialize GradScaler for Automatic Mixed Precision
    scaler = torch.amp.GradScaler("cuda")

    for images, labels, _ in loader:
        images = images.to(device)
        labels = labels.to(device)

        batch_size = images.size(0)

        # Apply Mixup
        if mixup_alpha > 0:
            images, labels = mixup_data(images, labels, mixup_alpha, device)

        # Reshape labels for BCEWithLogitsLoss: (Batch,) -> (Batch, 1)
        labels = labels.view(-1, 1)

        optimizer.zero_grad()

        # Forward pass with AMP
        with torch.amp.autocast("cuda"):
            outputs = model(images)
            loss = criterion(outputs, labels)

        # Backward pass with Scaler
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Update EMA model
        if ema_model:
            ema_model.update(model)

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device)
            labels = labels.to(device)

            batch_size = images.size(0)

            # Forward pass with AMP (for speed)
            with torch.amp.autocast("cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels.view(-1, 1))

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(outputs)

            all_targets.append(labels.cpu().numpy())
            all_preds.append(probs.cpu().numpy())

    epoch_loss = running_loss / dataset_size

    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)

    try:
        auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        # Handle edge case if a batch has only one class
        auc = 0.5

    return epoch_loss, auc


class Trainer:
    """
    Manages the training lifecycle, including optimization, scheduling, and checkpointing.
    """

    def __init__(self, model, train_loader, val_loader, device, fold_idx=0):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.fold_idx = fold_idx

        # Initialize EMA
        self.ema = (
            ModelEMA(self.model, decay=Config.EMA_DECAY) if Config.USE_EMA else None
        )

        # Optimizer: AdamW
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Scheduler: Cosine Annealing
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=Config.EPOCHS, eta_min=Config.MIN_LR
        )

        # Loss Function: BCEWithLogitsLoss (numerically stable)
        self.criterion = nn.BCEWithLogitsLoss()

        self.best_auc = 0.0
        self.patience_counter = 0

    def fit(self, epochs=None):
        """
        Runs the training loop for the specified number of epochs.
        """
        if epochs is None:
            epochs = Config.EPOCHS

        print(f"Starting training for Fold {self.fold_idx} on device {self.device}")

        for epoch in range(epochs):
            start_time = time.time()

            # Train
            train_loss = train_one_epoch(
                self.model,
                self.train_loader,
                self.optimizer,
                self.criterion,
                self.device,
                self.ema,
                Config.MIXUP_ALPHA,
            )

            # Validate
            # We validate the EMA model because that is what we will use for inference
            val_model = self.ema.model if self.ema else self.model
            val_loss, val_auc = validate(
                val_model, self.val_loader, self.criterion, self.device
            )

            # Step Scheduler
            self.scheduler.step()

            elapsed = time.time() - start_time

            # Print metrics with full precision
            print(
                f"Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {train_loss:.8f} - "
                f"Val Loss: {val_loss:.8f} - "
                f"Val AUC: {val_auc:.10f} - "
                f"Time: {elapsed:.2f}s"
            )

            # Checkpointing
            if val_auc > self.best_auc:
                self.best_auc = val_auc
                self.save_checkpoint("best_model")
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            # Early Stopping
            if self.patience_counter >= Config.PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        # Save last model state
        self.save_checkpoint("last_model")
        print(f"Training complete. Best Val AUC: {self.best_auc:.10f}")

    def save_checkpoint(self, name):
        """
        Saves the model weights. If EMA is used, saves the EMA weights.
        """
        model_to_save = self.ema.model if self.ema else self.model

        # Determine architecture name for filename
        if hasattr(self.model, "model_name"):
            arch_name = self.model.model_name
        else:
            arch_name = "model"

        filename = f"{name}_{arch_name}_fold_{self.fold_idx}.pth"
        path = os.path.join(Config.CHECKPOINT_DIR, filename)

        torch.save(model_to_save.state_dict(), path)
