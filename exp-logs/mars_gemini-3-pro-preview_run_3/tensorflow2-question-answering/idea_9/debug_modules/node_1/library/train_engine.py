import torch
import numpy as np
import random
import os
from library.config import Config
from library.ranker_model import train_ranker as _train_ranker_impl
from library.reader_model import train_reader as _train_reader_impl


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for random, numpy, and torch to ensure reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_ranker(train_loader, val_loader, embedding_matrix, device):
    """
    Encapsulates the training loop for the Decomposable Attention Ranker.

    Args:
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        embedding_matrix (numpy.ndarray): Pre-trained embedding matrix.
        device (torch.device): Device to run the model on.

    Returns:
        nn.Module: The trained Ranker model.
    """
    # Ensure reproducibility
    set_seed()

    # Delegate to the library implementation which handles:
    # - Model instantiation
    # - Binary Cross-Entropy Loss
    # - Adam Optimizer
    # - Gradient Clipping
    # - Validation and Early Stopping
    print("Initializing Ranker training sequence...")
    return _train_ranker_impl(train_loader, val_loader, embedding_matrix, device)


def train_reader(train_loader, val_loader, embedding_matrix, device):
    """
    Encapsulates the training loop for the Gated Convolutional Reader.

    Args:
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        embedding_matrix (numpy.ndarray): Pre-trained embedding matrix.
        device (torch.device): Device to run the model on.

    Returns:
        nn.Module: The trained Reader model.
    """
    # Ensure reproducibility
    set_seed()

    # Delegate to the library implementation which handles:
    # - Model instantiation
    # - Categorical Cross-Entropy Loss (Start + End)
    # - Adam Optimizer
    # - Gradient Clipping
    # - Validation and Early Stopping
    print("Initializing Reader training sequence...")
    return _train_reader_impl(train_loader, val_loader, embedding_matrix, device)
