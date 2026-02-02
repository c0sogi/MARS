import os
import gc
import numpy as np
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import f1_score

from library.config import Config
from library.utils import get_logger, save_checkpoint, load_checkpoint, seed_everything
from library.data import get_dataloaders
from library.model import HerbariumNet
from library.loss import TaxonomicFocalLoss

logger = get_logger("train")


def train_one_epoch(epoch, model, loader, criterion, optimizer, device, scaler):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        batch_size = images.size(0)

        optimizer.zero_grad()

        with autocast():
            # Forward pass
            # ArcFaceHead requires labels during training to apply margin
            logits = model(images, labels)
            loss = criterion(logits, labels)

        # Backward pass with scaler for mixed precision
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and Macro F1 score.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            batch_size = images.size(0)

            # Forward pass
            # We pass labels to ensure the loss calculation matches training (margin applied)
            # This provides a consistent metric for loss monitoring.
            with autocast():
                logits = model(images, labels)
                loss = criterion(logits, labels)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Predictions
            # Logits are scaled cosine similarities. Argmax gives the class.
            preds = torch.argmax(logits, dim=1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    avg_loss = running_loss / dataset_size

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # Calculate Macro F1
    macro_f1 = f1_score(all_labels, all_preds, average="macro")

    return avg_loss, macro_f1


def fit(
    phase_config,
    model,
    criterion,
    device,
    checkpoint_dir,
    start_epoch=0,
    resume_best_score=0.0,
):
    """
    Runs the training loop for a specific phase configuration.
    """
    phase_name = phase_config["name"]
    img_size = phase_config["img_size"]
    batch_size = phase_config["batch_size"]
    epochs = phase_config["epochs"]
    lr = phase_config["lr"]
    weight_decay = phase_config["weight_decay"]

    logger.info(
        f"Starting Phase: {phase_name} | Img: {img_size} | Batch: {batch_size} | LR: {lr}"
    )

    # Get DataLoaders
    train_loader, val_loader = get_dataloaders(img_size, batch_size, debug=Config.DEBUG)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=phase_config["scheduler_factor"],
        patience=phase_config["scheduler_patience"],
        verbose=True,
    )

    # Scaler for AMP
    scaler = GradScaler()

    best_score = resume_best_score
    best_epoch = start_epoch

    # Early Stopping tracking
    patience = 3
    patience_counter = 0

    for epoch in range(start_epoch, start_epoch + epochs):
        logger.info(f"Epoch {epoch+1}/{start_epoch + epochs}")

        # Train
        train_loss = train_one_epoch(
            epoch, model, train_loader, criterion, optimizer, device, scaler
        )

        # Validate
        val_loss, val_f1 = validate(model, val_loader, criterion, device)

        # Update Scheduler
        scheduler.step(val_f1)

        # Print metrics with full precision
        logger.info(
            f"Train Loss: {train_loss:.9f} | Val Loss: {val_loss:.9f} | Val F1: {val_f1:.9f}"
        )

        # Checkpointing
        is_best = val_f1 > best_score
        if is_best:
            best_score = val_f1
            best_epoch = epoch
            patience_counter = 0
            logger.info(f"New best score: {best_score:.9f}. Saving model...")
        else:
            patience_counter += 1

        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_score": best_score,
                "phase": phase_name,
            },
            is_best,
            checkpoint_dir,
            filename=f"checkpoint_{phase_name}.pth",
        )

        # Early Stopping
        if patience_counter >= patience:
            logger.info(
                f"Early stopping triggered after {patience} epochs without improvement."
            )
            break

        # Memory cleanup
        gc.collect()
        torch.cuda.empty_cache()

    logger.info(
        f"Phase {phase_name} completed. Best F1: {best_score:.9f} at Epoch {best_epoch+1}"
    )
    return best_score


def run_training_pipeline():
    """
    Orchestrates the Progressive Resizing training strategy (Phase 1 -> Phase 2).
    """
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Initialize Model
    logger.info("Initializing model...")
    model = HerbariumNet(pretrained=True)
    model.to(device)

    # Initialize Loss
    criterion = TaxonomicFocalLoss(
        gamma=Config.FOCAL_LOSS_GAMMA, epsilon=Config.LABEL_SMOOTHING_EPS
    )
    criterion.to(device)

    # -------------------------------------------------------------------------
    # Phase 1: Coarse Training (224x224)
    # -------------------------------------------------------------------------
    phase1_best_score = fit(
        Config.PHASE1, model, criterion, device, Config.CHECKPOINT_DIR
    )

    # -------------------------------------------------------------------------
    # Phase 2: Fine-Tuning (300x300)
    # -------------------------------------------------------------------------
    # Load best model from Phase 1
    logger.info("Loading best model from Phase 1 for Phase 2 fine-tuning...")
    best_phase1_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    if os.path.exists(best_phase1_path):
        # We load the weights. We reset the optimizer/scheduler in fit(), which is desired for fine-tuning.
        _, _ = load_checkpoint(best_phase1_path, model, device=Config.DEVICE)
    else:
        logger.warning("Phase 1 best model not found. Proceeding with current weights.")

    # Run Phase 2
    fit(
        Config.PHASE2,
        model,
        criterion,
        device,
        Config.CHECKPOINT_DIR,
        start_epoch=0,  # Reset epoch counter for clarity in logs
        resume_best_score=phase1_best_score,  # Keep track of absolute best
    )

    logger.info("Training Pipeline Completed.")
