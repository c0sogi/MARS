import os
import time
import torch
import pandas as pd
import numpy as np
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

from library.config import Config
from library.utils import (
    Averager,
    Logger,
    get_device,
    format_submission_string,
    set_seed,
)
from library.data import get_dataloaders, get_test_dataloader
from library.model import get_model


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=50):
    """
    Trains the model for one epoch using Automatic Mixed Precision (AMP).
    """
    model.train()
    loss_hist = Averager()

    # Initialize GradScaler for AMP
    scaler = torch.cuda.amp.GradScaler()

    print(f"Epoch {epoch+1}/{Config.EPOCHS} - Training started")

    for step, (images, targets, image_ids) in enumerate(data_loader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Use AMP autocast
        with torch.cuda.amp.autocast():
            loss_dict = model(images, targets)
            # loss_dict contains 'classification' and 'bbox_regression' losses
            losses = sum(loss for loss in loss_dict.values())

        loss_value = losses.item()

        if not np.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        optimizer.zero_grad()

        # Scale losses and step optimizer
        scaler.scale(losses).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_hist.send(loss_value)

        if (step + 1) % print_freq == 0:
            print(f"Epoch: {epoch+1}, Step: {step+1}, Loss: {loss_hist.value}")

    return loss_hist.value


@torch.no_grad()
def validate(model, data_loader, device):
    """
    Evaluates the model on the validation set.
    Note: We keep the model in train mode to retrieve the loss dictionary
    for monitoring convergence (Early Stopping).
    """
    # Keep in train mode to get losses, but disable gradients
    model.train()
    loss_hist = Averager()

    print("Validating...")

    for images, targets, image_ids in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        loss_hist.send(losses.item())

    return loss_hist.value


def train_model(debug=False):
    """
    Main training loop with Early Stopping.
    """
    set_seed(Config.SEED)
    device = get_device()

    # Initialize Logger
    logger = Logger(os.path.join(Config.WORKING_DIR, "train_log.txt"))

    # DataLoaders
    train_loader, val_loader = get_dataloaders(debug=debug)

    # Model
    model = get_model(pretrained=Config.PRETRAINED)
    model.to(device)

    # Optimizer & Scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = SGD(
        params,
        lr=Config.LEARNING_RATE,
        momentum=Config.MOMENTUM,
        weight_decay=Config.WEIGHT_DECAY,
    )

    lr_scheduler = CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)

    # Training Loop Variables
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    logger.log("Starting training...")

    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)

        # Validate
        val_loss = validate(model, val_loader, device)

        # Update Scheduler
        lr_scheduler.step()

        elapsed = time.time() - start_time

        log_msg = (
            f"Epoch {epoch+1} - "
            f"Train Loss: {train_loss} "
            f"Val Loss: {val_loss} "
            f"Time: {elapsed:.2f}s"
        )
        logger.log(log_msg)

        # Early Stopping & Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            logger.log(
                f"New best model saved at epoch {epoch+1} with Val Loss: {val_loss}"
            )
        else:
            patience_counter += 1
            logger.log(
                f"No improvement. Patience: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE}"
            )

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            logger.log("Early stopping triggered.")
            break

    return best_model_path


@torch.no_grad()
def predict_and_submit(model_path=None):
    """
    Runs inference on the test set and generates the submission file.
    """
    device = get_device()

    # Load Test Metadata
    test_df = pd.read_csv(Config.TEST_META_PATH)
    test_loader = get_test_dataloader(test_df)

    # Load Model
    model = get_model(pretrained=False)
    if model_path is None:
        model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"Warning: Model path {model_path} not found. Using random weights.")

    model.to(device)
    model.eval()

    results = []
    print("Running inference on test set...")

    for images, targets, image_ids in test_loader:
        images = list(image.to(device) for image in images)

        # Forward pass
        # In eval mode, model returns list of dicts: [{'boxes':..., 'scores':..., 'labels':...}, ...]
        outputs = model(images)

        for i, output in enumerate(outputs):
            image_id = image_ids[i]

            boxes = output["boxes"].data.cpu().numpy()
            scores = output["scores"].data.cpu().numpy()
            labels = output["labels"].data.cpu().numpy()

            # Filter by confidence threshold
            mask = scores >= Config.CONF_THRESHOLD
            boxes = boxes[mask]
            scores = scores[mask]
            labels = labels[mask]

            prediction_string = format_submission_string(boxes, scores, labels)

            results.append(
                {"image_id": image_id, "PredictionString": prediction_string}
            )

    # Create Submission DataFrame
    submission_df = pd.DataFrame(results)

    # Ensure column order
    submission_df = submission_df[["image_id", "PredictionString"]]

    # Save
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(submission_df.head())
