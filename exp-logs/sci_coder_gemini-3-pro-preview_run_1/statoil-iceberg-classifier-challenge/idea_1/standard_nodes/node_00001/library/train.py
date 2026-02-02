import sys
import os
import torch
from torch.utils.data import DataLoader

# Ensure library imports work
sys.path.append(os.getcwd())

from library.config import Config
from library.data_loader import get_dataloaders
from library.model import train_model, generate_submission


def run_training(
    num_epochs=50,
    batch_size=32,
    learning_rate=0.001,
    load_cached_data=True,
    max_samples=None,
):
    """
    Orchestrates the training and submission pipeline.

    Args:
        num_epochs (int): Number of training epochs.
        batch_size (int): Batch size for training and inference.
        learning_rate (float): Learning rate for the optimizer.
        load_cached_data (bool): Whether to use cached pre-processed data.
        max_samples (int, optional): Limit the number of samples for debugging.
    """

    # 1. Load Data
    print(f"Loading data (Cached: {load_cached_data})...")
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=load_cached_data
    )

    # 2. Adjust DataLoaders if hyperparameters require it
    # We re-wrap the datasets if batch_size changes or max_samples is set
    if batch_size != Config().BATCH_SIZE or max_samples is not None:
        print(
            f"Reconfiguring loaders: Batch Size={batch_size}, Max Samples={max_samples}"
        )

        datasets = {
            "train": train_loader.dataset,
            "val": val_loader.dataset,
            "test": test_loader.dataset,
        }

        # Apply max_samples limit if requested (for debugging)
        if max_samples is not None:
            for key, ds in datasets.items():
                limit = min(len(ds.images), max_samples)
                ds.images = ds.images[:limit]
                ds.angles = ds.angles[:limit]
                if ds.labels is not None:
                    ds.labels = ds.labels[:limit]

        # Re-create DataLoaders
        train_loader = DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )

        val_loader = DataLoader(
            datasets["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

        test_loader = DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

    # 3. Setup Configuration
    config = Config()
    config.NUM_EPOCHS = num_epochs
    config.LEARNING_RATE = learning_rate
    config.BATCH_SIZE = batch_size

    # 4. Train Model
    print("Starting training...")
    model = train_model(train_loader, val_loader, config)

    # 5. Generate Submission
    print("Generating submission...")
    generate_submission(model, test_loader, config)

    print("Process completed.")
