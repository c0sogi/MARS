import os
import torch
import torch.nn as nn
import torch.optim as optim
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy
from library import config
from library import utils


def train_one_epoch(model, loader, optimizer, criterion, device, scaler, mixup_fn=None):
    """
    Trains the model for one epoch using Mixed Precision.
    """
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for images, targets, _ in loader:
        images = images.to(device)
        targets = targets.to(device)

        if mixup_fn is not None:
            images, targets = mixup_fn(images, targets)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)

        # Calculate accuracy (approximate if mixup is used)
        # If mixup is used, targets are soft, so we can't strictly compute acc here easily
        # We just use argmax of output vs argmax of target (if soft) or target (if hard)
        _, predicted = torch.max(outputs, 1)

        if mixup_fn is not None:
            # Recover hard label for accuracy calculation
            _, targets_hard = torch.max(targets, 1)
            correct_predictions += (predicted == targets_hard).sum().item()
        else:
            correct_predictions += (predicted == targets).sum().item()

        total_samples += images.size(0)

    avg_loss = running_loss / total_samples
    avg_acc = correct_predictions / total_samples

    return avg_loss, avg_acc


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set using Mixed Precision.
    """
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for images, targets, _ in loader:
            images = images.to(device)
            targets = targets.to(device)

            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, targets)

            running_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == targets).sum().item()
            total_samples += targets.size(0)

    avg_loss = running_loss / total_samples
    avg_acc = correct_predictions / total_samples

    return avg_loss, avg_acc


def run_stage(model, train_loader, val_loader, stage_config):
    """
    Runs a specific training stage including setup, loop, early stopping, and saving.

    Args:
        model: The PyTorch model.
        train_loader: DataLoader for training.
        val_loader: DataLoader for validation.
        stage_config: Dictionary containing stage hyperparameters.

    Returns:
        model: The model loaded with the best weights from this stage.
    """
    logger = utils.get_logger(stage_config["stage_name"])
    device = config.DEVICE
    model = model.to(device)

    # Hyperparameters
    lr = stage_config["learning_rate"]
    weight_decay = stage_config["weight_decay"]
    epochs = stage_config["epochs"]
    patience = stage_config["patience"]
    checkpoint_name = stage_config["checkpoint_name"]
    label_smoothing = stage_config.get("label_smoothing", 0.0)

    # Setup Optimizer and Scheduler
    # Using AdamW as a robust default for EfficientNetV2
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Setup Mixup
    mixup_fn = None
    if stage_config.get("use_mixup", False):
        logger.info("Enabling Mixup/Cutmix for this stage.")
        mixup_fn = Mixup(**config.MIXUP_CONFIG)
        # Use SoftTargetCrossEntropy when Mixup is enabled
        criterion = SoftTargetCrossEntropy()
    else:
        # Setup Standard Loss Function
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # Setup GradScaler for AMP
    scaler = torch.cuda.amp.GradScaler()

    # Early Stopping Variables
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(config.WORKING_DIR, checkpoint_name)

    logger.info(f"Starting {stage_config['stage_name']} with {epochs} epochs.")

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            scaler,
            mixup_fn=mixup_fn,
        )
        # Validation always uses standard CrossEntropy (no mixup)
        val_criterion = nn.CrossEntropyLoss()
        val_loss, val_acc = validate(model, val_loader, val_criterion, device)

        # Step the scheduler
        scheduler.step()

        # Logging full precision metrics
        logger.info(f"Epoch {epoch+1}/{epochs}")
        logger.info(f"Train Loss: {train_loss}")
        logger.info(f"Train Acc: {train_acc}")
        logger.info(f"Val Loss: {val_loss}")
        logger.info(f"Val Acc: {val_acc}")

        # Checkpoint and Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"New best model saved to {best_model_path}")
        else:
            patience_counter += 1
            logger.info(f"Early stopping counter: {patience_counter}/{patience}")

        if patience_counter >= patience:
            logger.info("Early stopping triggered.")
            break

    # Load best weights before returning
    if os.path.exists(best_model_path):
        logger.info(f"Loading best weights from {best_model_path}")
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    return model
