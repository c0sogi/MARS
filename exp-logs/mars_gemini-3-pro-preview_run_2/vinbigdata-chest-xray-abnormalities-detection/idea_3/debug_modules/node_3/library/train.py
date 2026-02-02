import os
import sys
import time
import math
import torch
import pandas as pd
import numpy as np
import torchvision
from library.config import Config
from library.utils import get_logger, set_seed, AverageMeter, format_prediction_string
from library.dataset import get_dataloaders
from library.model import get_model
from library.preprocess import DicomPreprocessor


def train_one_epoch(model, optimizer, data_loader, device, epoch, logger):
    """
    Trains the model for one epoch.
    """
    model.train()
    loss_meter = AverageMeter()

    for i, (images, targets) in enumerate(data_loader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()

        if not math.isfinite(loss_value):
            logger.error(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        loss_meter.update(loss_value, len(images))

        if i % 50 == 0:
            logger.info(
                f"Epoch: [{epoch}] Iter: [{i}/{len(data_loader)}] Loss: {loss_meter.val:.4f} ({loss_meter.avg:.4f})"
            )

    return loss_meter.avg


@torch.no_grad()
def evaluate(model, data_loader, device, logger):
    """
    Evaluates the model on the validation set.
    Uses loss as a proxy for performance to select the best model.
    """
    # Set to train mode to obtain loss values, but disable gradients
    model.train()
    loss_meter = AverageMeter()

    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        loss_meter.update(losses.item(), len(images))

    return loss_meter.avg


@torch.no_grad()
def generate_submission(model, test_loader, device, logger):
    """
    Generates predictions for the test set and saves them to a CSV file.
    """
    model.eval()
    results = []
    logger.info("Generating predictions for test set...")

    for images, targets in test_loader:
        images = list(img.to(device) for img in images)
        outputs = model(images)

        for i, output in enumerate(outputs):
            image_id = targets[i]["image_id"]

            boxes = output["boxes"].cpu().numpy()
            scores = output["scores"].cpu().numpy()
            labels = output["labels"].cpu().numpy()

            # Filter by confidence
            mask = scores >= Config.CONFIDENCE_THRESHOLD
            boxes = boxes[mask]
            scores = scores[mask]
            labels = labels[mask]

            # Apply NMS
            if len(boxes) > 0:
                # Using 0.5 as standard NMS threshold
                keep = torchvision.ops.nms(
                    torch.from_numpy(boxes), torch.from_numpy(scores), 0.5
                )
                boxes = boxes[keep]
                scores = scores[keep]
                labels = labels[keep]

            # Convert model labels (1-14) back to dataset class IDs (0-13)
            # Note: Class 14 (No finding) is handled by format_prediction_string if list is empty
            final_labels = [int(l) - 1 for l in labels]

            pred_str = format_prediction_string(boxes, scores, final_labels)
            results.append({"image_id": image_id, "PredictionString": pred_str})

    df_sub = pd.DataFrame(results)
    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")


def run():
    # 1. Setup
    logger = get_logger(os.path.join(Config.LOG_DIR, "train_log.txt"))
    set_seed(Config.SEED)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger.info(f"Using device: {device}")

    # 2. Data Loading with Caching Mechanism
    logger.info("Initializing DataLoaders...")
    try:
        train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)
    except FileNotFoundError:
        logger.warning("Cached data not found. Running offline preprocessing...")
        preprocessor = DicomPreprocessor()
        preprocessor.run(load_cached_data=False)
        train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # 3. Model Initialization
    logger.info("Initializing Model...")
    model = get_model(num_classes=Config.NUM_CLASSES, img_size=Config.IMG_SIZE)
    model.to(device)

    # 4. Optimizer & Scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=Config.LEARNING_RATE,
        momentum=Config.MOMENTUM,
        weight_decay=Config.WEIGHT_DECAY,
    )

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS
    )

    # 5. Training Loop
    best_val_loss = float("inf")
    patience = 5
    patience_counter = 0

    logger.info("Starting training...")

    for epoch in range(Config.EPOCHS):
        logger.info(f"--- Epoch {epoch+1}/{Config.EPOCHS} ---")

        # Train
        train_loss = train_one_epoch(
            model, optimizer, train_loader, device, epoch, logger
        )
        logger.info(f"Train Loss: {train_loss:.6f}")

        # Validate
        val_loss = evaluate(model, val_loader, device, logger)
        logger.info(f"Val Loss: {val_loss:.6f}")

        # Scheduler Step
        lr_scheduler.step()
        logger.info(f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Checkpoint & Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            logger.info(f"New best model saved! (Loss: {best_val_loss:.6f})")
        else:
            patience_counter += 1
            logger.info(f"No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            logger.info("Early stopping triggered.")
            break

    # 6. Submission
    logger.info("Training complete. Loading best model for inference...")
    if os.path.exists(Config.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    else:
        logger.warning("No model checkpoint found. Using current model state.")

    generate_submission(model, test_loader, device, logger)
