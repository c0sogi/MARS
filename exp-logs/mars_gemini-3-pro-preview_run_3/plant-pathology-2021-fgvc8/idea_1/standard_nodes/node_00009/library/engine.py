import torch
import os
from library.config import Config
from library.dataset import get_loaders
from library.model import (
    train_one_epoch as lib_train_one_epoch,
    validate as lib_validate,
    train_model as lib_train_model,
    generate_submission as lib_generate_submission,
)
from library.utils import set_seed


def train_one_epoch(model, loader, optimizer, criterion, scaler, device):
    """
    Wrapper for the library's train_one_epoch function.
    Performs one epoch of training using mixed precision.

    Args:
        model: The neural network model.
        loader: DataLoader for the training set.
        optimizer: The optimizer.
        criterion: The loss function.
        scaler: Gradient scaler for AMP.
        device: Computation device.

    Returns:
        tuple: (epoch_loss, epoch_f1)
    """
    return lib_train_one_epoch(model, loader, optimizer, criterion, scaler, device)


def evaluate(model, loader, criterion, device):
    """
    Wrapper for the library's validate function.
    Evaluates the model on the validation set.

    Args:
        model: The neural network model.
        loader: DataLoader for the validation set.
        criterion: The loss function.
        device: Computation device.

    Returns:
        tuple: (val_loss, val_f1)
    """
    return lib_validate(model, loader, criterion, device)


def predict(model, test_loader, device):
    """
    Wrapper for the library's generate_submission function.
    Generates predictions for the test set and saves them to CSV.

    Args:
        model: The trained model.
        test_loader: DataLoader for the test set.
        device: Computation device.
    """
    return lib_generate_submission(model, test_loader, device)


def run_training(debug_sample_size=None, epochs=Config.EPOCHS, device=Config.DEVICE):
    """
    Orchestrates the training pipeline.

    Args:
        debug_sample_size (int, optional): Number of samples to use for debugging.
        epochs (int): Number of training epochs.
        device (str): Device to use for training ('cuda' or 'cpu').

    Returns:
        nn.Module: The trained model.
    """
    # Ensure reproducibility
    set_seed(Config.SEED)

    # Initialize DataLoaders
    # We only need train and val loaders here
    train_loader, val_loader, _ = get_loaders(debug_sample_size=debug_sample_size)

    # Execute training loop
    # This handles model initialization, optimizer setup, loop, logging, and checkpointing
    # It also implements Early Stopping as required.
    model = lib_train_model(train_loader, val_loader, epochs=epochs, device=device)

    return model


def run_inference(model, debug_sample_size=None, device=Config.DEVICE):
    """
    Orchestrates the inference pipeline.

    Args:
        model (nn.Module): The trained model.
        debug_sample_size (int, optional): Number of samples to use for debugging.
        device (str): Device to use for inference.
    """
    # Initialize DataLoaders
    # We only need the test loader here
    _, _, test_loader = get_loaders(debug_sample_size=debug_sample_size)

    # Generate and save submission
    predict(model, test_loader, device)
