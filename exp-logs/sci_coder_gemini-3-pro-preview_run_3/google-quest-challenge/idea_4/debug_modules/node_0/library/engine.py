import torch
import torch.nn as nn
import numpy as np
import os
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from library.config import Config
from library.utils import seed_everything, compute_metric, TrainingLogger
from library.dataset import get_dataloaders
from library.modeling import (
    SegmentAwareCrossEncoder,
    train_backbone,
    extract_features,
    train_ridge_ensemble,
    predict_and_submit,
)


def train_one_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    device,
    criterion,
    gradient_accumulation_steps=1,
):
    """
    Trains the model for one epoch.

    Args:
        model: The PyTorch model.
        loader: The training DataLoader.
        optimizer: The optimizer.
        scheduler: The learning rate scheduler.
        device: The device to train on.
        criterion: The loss function.
        gradient_accumulation_steps (int): Steps for gradient accumulation.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    total_loss = 0.0

    for batch_idx, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        targets = batch["targets"].to(device)

        token_type_ids = batch.get("token_type_ids", None)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)

        outputs = model(input_ids, attention_mask, token_type_ids)
        loss = criterion(outputs, targets)

        loss = loss / gradient_accumulation_steps
        loss.backward()

        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * gradient_accumulation_steps

    return total_loss / len(loader)


def validate(model, loader, device, criterion):
    """
    Evaluates the model on the validation set.

    Args:
        model: The PyTorch model.
        loader: The validation DataLoader.
        device: The device to evaluate on.
        criterion: The loss function.

    Returns:
        tuple: (average_loss, spearman_score)
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["targets"].to(device)

            token_type_ids = batch.get("token_type_ids", None)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            outputs = model(input_ids, attention_mask, token_type_ids)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            preds = torch.sigmoid(outputs).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(targets.cpu().numpy())

    avg_loss = total_loss / len(loader)

    y_pred = np.vstack(all_preds)
    y_true = np.vstack(all_targets)
    score = compute_metric(y_true, y_pred)

    return avg_loss, score


def train_stream(model_name, save_path, device, load_cached_data=True):
    """
    High-level function to train a specific backbone stream (e.g., MPNet or RoBERTa).
    Uses the robust training loop provided in library.modeling.

    Args:
        model_name (str): HuggingFace model name.
        save_path (str): Path to save the best model.
        device (torch.device): Device to train on.
        load_cached_data (bool): Whether to use cached processed data.

    Returns:
        model: The fine-tuned model loaded with best weights.
    """
    seed_everything(Config.SEED)
    print(f"--- Training Stream: {model_name} ---")

    # Get DataLoaders
    train_loader, val_loader, _ = get_dataloaders(
        model_name, load_cached_data=load_cached_data
    )

    # Initialize Model
    model = SegmentAwareCrossEncoder(model_name)

    # Train using the provided library function which handles Early Stopping
    model = train_backbone(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=Config.NUM_EPOCHS,
        lr=Config.LEARNING_RATE,
        device=device,
        save_path=save_path,
    )

    return model


def process_stream_features(
    model, model_name, device, cache_paths, load_cached_data=True
):
    """
    Extracts and caches features for Train, Val, and Test sets for a given stream.

    Args:
        model: The fine-tuned model.
        model_name (str): HuggingFace model name (for tokenizer).
        device (torch.device): Device for inference.
        cache_paths (tuple): Paths for (train, val, test) feature caches.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (train_features, val_features, test_features)
    """
    train_path, val_path, test_path = cache_paths

    # Check if all exist to avoid reloading dataloaders if not needed
    if (
        load_cached_data
        and os.path.exists(train_path)
        and os.path.exists(val_path)
        and os.path.exists(test_path)
    ):
        print(f"Loading cached features for {model_name} stream...")
        return (np.load(train_path), np.load(val_path), np.load(test_path))

    print(f"--- Extracting Features for Stream: {model_name} ---")
    train_loader, val_loader, test_loader = get_dataloaders(
        model_name, load_cached_data=load_cached_data
    )

    train_feats = extract_features(
        model, train_loader, device, train_path, load_cached_data
    )
    val_feats = extract_features(model, val_loader, device, val_path, load_cached_data)
    test_feats = extract_features(
        model, test_loader, device, test_path, load_cached_data
    )

    return train_feats, val_feats, test_feats


def run_ensemble(X_train, y_train, X_val, y_val, save_path):
    """
    Trains the Ridge Regression ensemble head.

    Args:
        X_train, y_train: Training data and targets.
        X_val, y_val: Validation data and targets.
        save_path: Path to save the ridge model.

    Returns:
        model: Trained RidgeCV model.
    """
    seed_everything(Config.SEED)
    return train_ridge_ensemble(X_train, y_train, X_val, y_val, save_path)


def generate_submission(ridge_model, X_test, submission_path):
    """
    Generates the final submission file.

    Args:
        ridge_model: Trained Ridge model.
        X_test: Test features.
        submission_path: Path to save the CSV.
    """
    # Load test dataframe for IDs
    test_df = (
        torch.load(os.path.join(Config.WORKING_DIR, "test_processed.parquet"))
        if os.path.exists(os.path.join(Config.WORKING_DIR, "test_processed.parquet"))
        else None
    )

    if test_df is None:
        # Fallback to reading metadata if parquet not found (should be there from preprocessing)
        import pandas as pd

        test_df = pd.read_csv(Config.TEST_PATH)

    predict_and_submit(ridge_model, X_test, test_df, submission_path)
