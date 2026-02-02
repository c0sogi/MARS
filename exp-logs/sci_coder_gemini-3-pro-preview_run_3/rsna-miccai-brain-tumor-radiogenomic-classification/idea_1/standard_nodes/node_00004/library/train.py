import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

from library.config import Config
from library.utils import set_seed
from library.dataset import MGMTDataset
from library.model import MGMTNet, train_one_epoch, validate


def run_training(
    debug=False,
    epochs=Config.NUM_EPOCHS,
    batch_size=Config.BATCH_SIZE,
    learning_rate=Config.LEARNING_RATE,
    patience=Config.EARLY_STOPPING_PATIENCE,
    num_workers=Config.NUM_WORKERS,
    device_name=Config.DEVICE,
    save_path=Config.MODEL_PATH,
):
    """
    Orchestrates the training process for the MGMT prediction model.

    Args:
        debug (bool): If True, uses a small subset of data for debugging.
        epochs (int): Maximum number of training epochs.
        batch_size (int): Batch size for data loaders.
        learning_rate (float): Learning rate for the optimizer.
        patience (int): Number of epochs to wait for improvement before early stopping.
        num_workers (int): Number of subprocesses for data loading.
        device_name (str): Device to run training on ('cpu' or 'cuda').
        save_path (str): Path to save the best model checkpoint.

    Returns:
        float: The best validation AUC achieved.
    """
    # Ensure reproducibility
    set_seed(Config.SEED)
    device = torch.device(device_name)

    # Load Metadata
    train_df = pd.read_parquet(Config.TRAIN_METADATA)
    val_df = pd.read_parquet(Config.VAL_METADATA)

    if debug:
        train_df = train_df.head(Config.DEBUG_SAMPLE_SIZE)
        val_df = val_df.head(Config.DEBUG_SAMPLE_SIZE)

    # Initialize Datasets
    # MGMTDataset handles caching internally via load_cached_data=True (default)
    train_dataset = MGMTDataset(train_df, split_name="train")
    val_dataset = MGMTDataset(val_df, split_name="val")

    # Initialize DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Initialize Model
    model = MGMTNet().to(device)

    # Setup Optimization
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_auc = 0.0
    patience_counter = 0

    # Ensure output directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(epochs):
        # Run training step
        train_loss, train_auc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Run validation step
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        # Log metrics (full precision)
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss}")
        print(f"Train AUC: {train_auc}")
        print(f"Val Loss: {val_loss}")
        print(f"Val AUC: {val_auc}")

        # Checkpoint and Early Stopping
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved with AUC: {best_auc}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

    print(f"Training complete. Best Val AUC: {best_auc}")
    return best_auc


def generate_submission(
    model_path=Config.MODEL_PATH,
    output_path=Config.SUBMISSION_PATH,
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    device_name=Config.DEVICE,
):
    """
    Generates predictions for the test set using the trained model and saves to CSV.

    Args:
        model_path (str): Path to the trained model checkpoint.
        output_path (str): Path to save the submission CSV.
        batch_size (int): Batch size for inference.
        num_workers (int): Number of subprocesses for data loading.
        device_name (str): Device to run inference on.
    """
    set_seed(Config.SEED)
    device = torch.device(device_name)

    # Load Test Metadata
    test_df = pd.read_parquet(Config.TEST_METADATA)

    # Initialize Test Dataset and Loader
    test_dataset = MGMTDataset(test_df, split_name="test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Initialize and Load Model
    model = MGMTNet().to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"Warning: Model not found at {model_path}. Using random initialization.")

    model.eval()

    predictions = []

    # Inference Loop
    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            predictions.extend(probs)

    # Retrieve IDs
    ids = test_dataset.get_ids()

    # Create Submission DataFrame
    submission_df = pd.DataFrame({"BraTS21ID": ids, "MGMT_value": predictions})

    # Format BraTS21ID as integer (as per sample submission)
    submission_df["BraTS21ID"] = submission_df["BraTS21ID"].astype(int)

    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
