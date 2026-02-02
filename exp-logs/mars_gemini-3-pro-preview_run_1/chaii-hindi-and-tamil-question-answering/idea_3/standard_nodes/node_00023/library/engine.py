import os
import torch
import pandas as pd
import numpy as np
from transformers import get_linear_schedule_with_warmup
from library.model import train_one_epoch, validate, predict, post_process_predictions


def train_fn(model, dataloader, optimizer, scheduler, device):
    """
    Executes a single training epoch.

    Args:
        model: The PyTorch model.
        dataloader: DataLoader for training data.
        optimizer: The optimizer.
        scheduler: The learning rate scheduler.
        device: The device to run on.

    Returns:
        float: The average training loss for the epoch.
    """
    return train_one_epoch(model, dataloader, optimizer, scheduler, device)


def eval_fn(model, dataloader, device):
    """
    Executes evaluation on the validation set.

    Args:
        model: The PyTorch model.
        dataloader: DataLoader for validation data.
        device: The device to run on.

    Returns:
        float: The average validation loss.
    """
    return validate(model, dataloader, device)


def predict_fn(model, dataloader, device):
    """
    Executes inference on the test set.

    Args:
        model: The PyTorch model.
        dataloader: DataLoader for test data.
        device: The device to run on.

    Returns:
        tuple: (start_logits, end_logits) as numpy arrays.
    """
    return predict(model, dataloader, device)


def train_loop(model, train_loader, val_loader, config, device, patience=3):
    """
    Runs the full training loop with early stopping and metric logging.

    Args:
        model: The PyTorch model.
        train_loader: DataLoader for training.
        val_loader: DataLoader for validation.
        config: Configuration object containing hyperparameters.
        device: The device to run on.
        patience (int): Number of epochs to wait for improvement before early stopping.

    Returns:
        model: The model loaded with the best weights found during training.
    """
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    num_train_steps = len(train_loader) * config.epochs
    num_warmup_steps = int(num_train_steps * config.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_train_steps,
    )

    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0

    for epoch in range(config.epochs):
        train_loss = train_fn(model, train_loader, optimizer, scheduler, device)
        val_loss = eval_fn(model, val_loader, device)

        # Print metrics with full precision as requested
        print(f"Epoch {epoch+1}: Train Loss = {train_loss}, Val Loss = {val_loss}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    # Load the best model state if training occurred
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model


def generate_submission(model, test_loader, test_features, config, output_file=None):
    """
    Generates predictions for the test set and saves the submission file.

    Args:
        model: The trained PyTorch model.
        test_loader: DataLoader for test data.
        test_features: DataFrame containing processed test features (with offset mapping).
        config: Configuration object.
        output_file (str, optional): Path to save the submission CSV. Defaults to config.submission_file.
    """
    device = next(model.parameters()).device

    # Get raw logits
    start_logits, end_logits = predict_fn(model, test_loader, device)

    # Post-process to get strings
    # Note: post_process_predictions reads the test CSV from config path internally to get context
    predictions = post_process_predictions(test_features, start_logits, end_logits)

    # Prepare submission dataframe
    sample_sub = pd.read_csv(config.sample_submission_path)

    final_preds = []
    for pid in sample_sub["id"]:
        # Default to empty string if no prediction found
        pred_str = predictions.get(pid, "")
        final_preds.append(pred_str)

    sample_sub["PredictionString"] = final_preds

    if output_file is None:
        output_file = config.submission_file

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    sample_sub.to_csv(output_file, index=False)
    print(f"Submission saved to {output_file}")
