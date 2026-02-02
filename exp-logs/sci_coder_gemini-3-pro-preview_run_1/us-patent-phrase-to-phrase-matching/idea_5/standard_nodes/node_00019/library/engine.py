import os
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import (
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)
from library.config import Config
from library.utils import get_score, get_logger
from library.model import CustomModel

# -----------------------------------------------------------------------------
# Optimizer and Scheduler Helpers
# -----------------------------------------------------------------------------


def get_optimizer(model):
    """
    Constructs the optimizer with Layer-wise Learning Rate Decay (LLRD).
    Leverages the grouping logic defined in the CustomModel.
    """
    optimizer_grouped_parameters = model.get_optimizer_params(
        base_lr=Config.lr,
        weight_decay=Config.weight_decay,
        layer_decay=Config.layer_decay,
    )

    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters, lr=Config.lr, eps=1e-6, betas=(0.9, 0.999)
    )
    return optimizer


def get_scheduler(optimizer, num_train_steps):
    """
    Constructs the learning rate scheduler based on the Config.
    """
    if Config.scheduler == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(num_train_steps * Config.warmup_ratio),
            num_training_steps=num_train_steps,
        )
    elif Config.scheduler == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(num_train_steps * Config.warmup_ratio),
            num_training_steps=num_train_steps,
        )
    else:
        scheduler = None
    return scheduler


# -----------------------------------------------------------------------------
# Core Execution Loops
# -----------------------------------------------------------------------------


def train_fn(model, dataloader, optimizer, scheduler, device, epoch):
    """
    Executes one epoch of training.
    """
    model.train()

    # Use torch.amp for mixed precision
    scaler = torch.amp.GradScaler("cuda")
    loss_fn = nn.MSELoss()

    running_loss = 0.0
    dataset_size = 0

    for step, data in enumerate(dataloader):
        input_ids = data["input_ids"].to(device)
        attention_mask = data["attention_mask"].to(device)
        labels = data["label"].to(device)

        batch_size = input_ids.size(0)

        with torch.amp.autocast("cuda"):
            outputs = model(input_ids, attention_mask)
            # outputs: [batch_size, 1], labels: [batch_size]
            loss = loss_fn(outputs.view(-1), labels.view(-1))

            if Config.gradient_accumulation_steps > 1:
                loss = loss / Config.gradient_accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % Config.gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

        # Accumulate scaled loss for reporting
        running_loss += (loss.item() * Config.gradient_accumulation_steps) * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def valid_fn(model, dataloader, device):
    """
    Executes validation loop.
    Returns average loss, pearson score, and predictions.
    """
    model.eval()

    preds = []
    labels_list = []
    running_loss = 0.0
    dataset_size = 0

    loss_fn = nn.MSELoss()

    with torch.no_grad():
        for data in dataloader:
            input_ids = data["input_ids"].to(device)
            attention_mask = data["attention_mask"].to(device)
            labels = data["label"].to(device)

            batch_size = input_ids.size(0)

            with torch.amp.autocast("cuda"):
                outputs = model(input_ids, attention_mask)
                loss = loss_fn(outputs.view(-1), labels.view(-1))

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            preds.append(outputs.view(-1).cpu().numpy())
            labels_list.append(labels.view(-1).cpu().numpy())

    predictions = np.concatenate(preds)
    ground_truth = np.concatenate(labels_list)

    epoch_loss = running_loss / dataset_size
    score = get_score(ground_truth, predictions)

    return epoch_loss, score, predictions


def inference_fn(model, dataloader, device):
    """
    Executes inference on the test set.
    """
    model.eval()
    preds = []

    with torch.no_grad():
        for data in dataloader:
            input_ids = data["input_ids"].to(device)
            attention_mask = data["attention_mask"].to(device)

            with torch.amp.autocast("cuda"):
                outputs = model(input_ids, attention_mask)

            preds.append(outputs.view(-1).cpu().numpy())

    predictions = np.concatenate(preds)
    return predictions


# -----------------------------------------------------------------------------
# Orchestration
# -----------------------------------------------------------------------------


def run_fold(fold, train_loader, valid_loader, device):
    """
    Runs the training and validation loop for a single fold.
    Saves the best model based on validation Pearson score.
    """
    logger = get_logger(f"train_fold_{fold}")
    logger.info(f"Starting execution for Fold {fold}")

    # Initialize Model
    model = CustomModel()
    model.to(device)

    # Initialize Optimizer and Scheduler
    optimizer = get_optimizer(model)
    num_train_steps = int(
        len(train_loader) / Config.gradient_accumulation_steps * Config.epochs
    )
    scheduler = get_scheduler(optimizer, num_train_steps)

    best_score = -np.inf
    best_model_path = os.path.join(Config.models_dir, f"model_fold_{fold}.pth")

    for epoch in range(Config.epochs):
        train_loss = train_fn(model, train_loader, optimizer, scheduler, device, epoch)
        valid_loss, valid_score, _ = valid_fn(model, valid_loader, device)

        logger.info(
            f"Epoch {epoch+1}/{Config.epochs} - "
            f"Train Loss: {train_loss} - "
            f"Valid Loss: {valid_loss} - "
            f"Valid Score: {valid_score}"
        )

        # Model Checkpointing (Save Best)
        if valid_score > best_score:
            best_score = valid_score
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"New best score achieved. Model saved to {best_model_path}")

    # Cleanup
    del model, optimizer, scheduler
    torch.cuda.empty_cache()
    gc.collect()

    return best_score


def predict_and_submit(test_loader, device):
    """
    Loads all trained fold models, performs ensemble inference on the test set,
    and saves the final submission file.
    """
    logger = get_logger("inference")
    logger.info("Starting inference and submission generation...")

    all_preds = []

    for fold in range(Config.num_folds):
        model_path = os.path.join(Config.models_dir, f"model_fold_{fold}.pth")
        if not os.path.exists(model_path):
            logger.warning(
                f"Model for fold {fold} not found at {model_path}. Skipping."
            )
            continue

        logger.info(f"Predicting with model fold {fold}...")
        model = CustomModel()
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)

        preds = inference_fn(model, test_loader, device)
        all_preds.append(preds)

        del model
        torch.cuda.empty_cache()
        gc.collect()

    if not all_preds:
        logger.error("No predictions generated. Submission failed.")
        return

    # Ensemble: Average predictions across folds
    avg_preds = np.mean(all_preds, axis=0)

    # Load sample submission or test file to get IDs
    # Using test_path from metadata as per Config
    try:
        df_sub = pd.read_csv(Config.test_path)
        df_sub["score"] = avg_preds

        # Ensure correct format
        df_sub = df_sub[["id", "score"]]

        df_sub.to_csv(Config.submission_path, index=False)
        logger.info(f"Submission saved successfully to {Config.submission_path}")

    except Exception as e:
        logger.error(f"Failed to save submission: {e}")
