import torch
import numpy as np
from library.models_semantic import train_fn as train_fn_lib
from library.models_semantic import valid_fn as valid_fn_lib
from library.models_semantic import inference_fn as inference_fn_lib


def train_one_epoch(
    model, train_loader, optimizer, scheduler, criterion, epoch, device, awp=None
):
    """
    Executes one epoch of training for the Semantic branch.

    This function handles the forward pass, loss calculation (typically SmoothL1Loss),
    backpropagation with gradient scaling, and optionally triggers Adversarial Weight
    Perturbation (AWP) if an AWP instance is provided.

    Args:
        model (torch.nn.Module): The PyTorch model to train.
        train_loader (DataLoader): DataLoader providing the training batches.
        optimizer (torch.optim.Optimizer): The optimizer for updating model weights.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        criterion (torch.nn.Module): The loss function.
        epoch (int): The current epoch number (0-indexed).
        device (torch.device): The device (CPU or GPU) to perform computations on.
        awp (object, optional): An instance of the AWP class to perform adversarial
                                weight perturbation. Defaults to None.

    Returns:
        float: The average training loss for the epoch.
    """
    return train_fn_lib(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        epoch=epoch,
        device=device,
        awp=awp,
    )


def valid_one_epoch(model, valid_loader, criterion, device):
    """
    Executes one epoch of validation.

    Evaluates the model on the provided validation loader, computing the loss
    and collecting predictions.

    Args:
        model (torch.nn.Module): The PyTorch model to evaluate.
        valid_loader (DataLoader): DataLoader providing the validation batches.
        criterion (torch.nn.Module): The loss function.
        device (torch.device): The device to perform computations on.

    Returns:
        tuple: A tuple containing:
            - float: The average validation loss.
            - np.ndarray: The model predictions (flattened).
            - np.ndarray: The true labels (flattened).
    """
    return valid_fn_lib(
        model=model, valid_loader=valid_loader, criterion=criterion, device=device
    )


def inference_fn(model, loader, device):
    """
    Generates predictions for a given dataset using the trained model.

    Args:
        model (torch.nn.Module): The trained PyTorch model.
        loader (DataLoader): DataLoader providing the input data (e.g., test set).
        device (torch.device): The device to perform computations on.

    Returns:
        np.ndarray: An array of predictions corresponding to the inputs in the loader.
    """
    return inference_fn_lib(model=model, loader=loader, device=device)
