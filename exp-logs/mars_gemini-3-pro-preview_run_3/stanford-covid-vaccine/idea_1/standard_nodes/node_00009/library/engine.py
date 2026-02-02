import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from library.config import Config
from library.model import RNA_GRU, fit, predict, train_one_epoch, validate
from library.utils import MCRMSE, seed_everything


def train_fn(model, loader, optimizer, criterion, device):
    """
    Wrapper for the single epoch training function.
    """
    return train_one_epoch(model, loader, optimizer, criterion, device)


def eval_fn(model, loader, criterion, device):
    """
    Wrapper for the validation function.
    """
    return validate(model, loader, criterion, device)


def predict_fn(model, loader, device):
    """
    Wrapper for the prediction function.
    """
    return predict(model, loader, device)


def train_model(train_loader, val_loader, device):
    """
    Initializes the model, optimizer, and criterion, then runs the training loop.

    Args:
        train_loader (DataLoader): Loader for training data.
        val_loader (DataLoader): Loader for validation data.
        device (torch.device): Device to run on.

    Returns:
        tuple: (trained_model, history_dict)
    """
    # Ensure reproducibility
    seed_everything(Config.SEED)

    # Initialize Model
    # Note: RNA_GRU is now the CRNN architecture defined in model.py
    model = RNA_GRU().to(device)

    # Initialize Optimizer
    # Using AdamW as per the design idea
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Initialize Criterion
    # Using MCRMSE for both training (differentiable) and validation (metric)
    criterion = MCRMSE()

    # Run Training Loop
    history = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=Config.EPOCHS,
        patience=Config.PATIENCE,
        save_path=Config.MODEL_SAVE_PATH,
    )

    return model, history


def generate_submission(model, test_loader, test_df, device, save_path=None):
    """
    Generates predictions for the test set and saves them to a CSV file.

    Args:
        model (nn.Module): Trained model.
        test_loader (DataLoader): Loader for test data.
        test_df (pd.DataFrame): DataFrame containing test metadata (IDs).
        device (torch.device): Device to run on.
        save_path (str, optional): Path to save the submission CSV.
    """
    if save_path is None:
        save_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 1. Generate Raw Predictions
    # Shape: (N_samples, 107, 5)
    raw_preds = predict_fn(model, test_loader, device)

    # 3. Format for Submission
    submission_data = []
    ids = test_df["id"].values
    target_cols = Config.TARGET_COLS

    for i, sample_id in enumerate(ids):
        sample_pred = raw_preds[i]  # Shape (107, 5)

        for seqpos in range(Config.SEQ_LENGTH):
            row_id = f"{sample_id}_{seqpos}"

            # Use model predictions for all positions
            vals = sample_pred[seqpos]

            # Create row dictionary
            row_dict = {"id_seqpos": row_id}
            for t_idx, col in enumerate(target_cols):
                row_dict[col] = vals[t_idx]

            submission_data.append(row_dict)

    # 4. Save to CSV
    submission_df = pd.DataFrame(submission_data)
    submission_df.to_csv(save_path, index=False)
    print(f"Submission saved to {save_path}")
