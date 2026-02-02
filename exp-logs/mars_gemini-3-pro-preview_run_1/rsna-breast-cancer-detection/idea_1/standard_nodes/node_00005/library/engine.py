import torch
import os
from library.config import Config
from library.model import (
    train_one_epoch as lib_train_one_epoch,
    validate as lib_validate,
    run_training as lib_run_training,
    generate_submission as lib_generate_submission,
)


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Performs a single training epoch.
    Delegates to the implementation in library.model to avoid duplication.

    Args:
        model: The neural network model.
        loader: DataLoader for training data.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device to run on (cpu/cuda).

    Returns:
        tuple: (epoch_loss, epoch_pf1)
    """
    return lib_train_one_epoch(model, loader, criterion, optimizer, device)


def valid_one_epoch(model, loader, criterion, device):
    """
    Performs a single validation epoch.
    Delegates to the implementation in library.model (named 'validate').

    Args:
        model: The neural network model.
        loader: DataLoader for validation data.
        criterion: Loss function.
        device: Device to run on (cpu/cuda).

    Returns:
        tuple: (val_loss, val_pf1)
    """
    return lib_validate(model, loader, criterion, device)


def run_training(sample_size=Config.SAMPLE_SIZE, epochs=Config.EPOCHS):
    """
    Main training loop with Early Stopping.
    Delegates to library.model.run_training.

    Args:
        sample_size (int, optional): Number of samples to use for debugging.
        epochs (int): Maximum number of epochs.

    Returns:
        model: The trained model with the best validation score.
    """
    return lib_run_training(sample_size=sample_size, epochs=epochs)


def generate_submission(model=None, sample_size=None):
    """
    Generates predictions for the test set and saves to submission.csv.
    Delegates to library.model.generate_submission.

    Args:
        model (nn.Module, optional): Trained model. If None, loads from disk.
        sample_size (int, optional): Number of samples to use for debugging.
    """
    return lib_generate_submission(model=model, sample_size=sample_size)
