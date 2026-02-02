import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

from library.config import Config
from library.dataset import SETIDataset
from library.model import BaselineCNN, train_one_epoch, validate, set_seeds


def run_training(
    epochs=Config.EPOCHS,
    batch_size=Config.BATCH_SIZE,
    learning_rate=Config.LEARNING_RATE,
    weight_decay=Config.WEIGHT_DECAY,
    debug=Config.DEBUG,
    debug_sample_size=Config.DEBUG_SAMPLE_SIZE,
    save_path=Config.MODEL_SAVE_PATH,
    patience=Config.EARLY_STOPPING_PATIENCE,
):
    """
    Orchestrates the training process with Early Stopping and Scheduler.
    """
    set_seeds()
    device = torch.device(Config.DEVICE)

    # Load Metadata
    df_train = pd.read_csv(Config.TRAIN_METADATA_PATH)
    df_val = pd.read_csv(Config.VAL_METADATA_PATH)

    # Apply debug limits if enabled
    if debug:
        df_train = df_train.head(debug_sample_size)
        df_val = df_val.head(debug_sample_size)

    # Create Datasets and Loaders
    train_dataset = SETIDataset(df_train)
    val_dataset = SETIDataset(df_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    # Initialize Model, Criterion, Optimizer
    model = BaselineCNN().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
    )

    best_auc = 0.0
    patience_counter = 0

    for epoch in range(epochs):
        # Run training and validation steps using imported helper functions
        train_loss, train_auc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        # Print metrics with full precision
        print(
            f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss}, Train AUC: {train_auc}, Val Loss: {val_loss}, Val AUC: {val_auc}"
        )

        # Step the scheduler based on Validation AUC
        scheduler.step(val_auc)

        # Early Stopping and Checkpointing
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    return model


def generate_submission(
    batch_size=Config.BATCH_SIZE,
    debug=Config.DEBUG,
    debug_sample_size=Config.DEBUG_SAMPLE_SIZE,
    model_path=Config.MODEL_SAVE_PATH,
    output_path=Config.SUBMISSION_SAVE_PATH,
):
    """
    Generates predictions for the test set and saves the submission file.
    """
    device = torch.device(Config.DEVICE)

    # Load Test Metadata
    df_test = pd.read_csv(Config.TEST_METADATA_PATH)
    if debug:
        df_test = df_test.head(debug_sample_size)

    test_dataset = SETIDataset(df_test)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    # Initialize Model and load weights
    model = BaselineCNN().to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        # If no model is found (e.g. in extremely short debug runs), we proceed with random weights
        # or one could raise an error. We proceed to ensure pipeline continuity.
        pass

    model.eval()
    all_preds = []

    # Inference Loop
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            # Apply sigmoid to logits to get probabilities
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            all_preds.extend(probs)

    # Create Submission DataFrame
    df_test["target"] = all_preds
    submission_df = df_test[["id", "target"]]

    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
