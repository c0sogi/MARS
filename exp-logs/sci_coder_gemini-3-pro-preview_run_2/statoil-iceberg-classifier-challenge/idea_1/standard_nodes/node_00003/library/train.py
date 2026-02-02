import random
import numpy as np
import torch
from library.config import Config
from library.model import D2N, train_model, generate_submission
from library.data_loader import get_data_loaders


def set_seeds(seed):
    """
    Sets random seeds for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_training(
    epochs=Config.NUM_EPOCHS,
    batch_size=Config.BATCH_SIZE,
    learning_rate=Config.LEARNING_RATE,
    patience=Config.PATIENCE,
    max_samples=None,
    load_cached_data=True,
):
    """
    Manages the model training lifecycle.

    Orchestrates data loading, model initialization, training with early stopping,
    and submission generation.

    Args:
        epochs (int): Maximum number of training epochs.
        batch_size (int): Batch size for data loaders.
        learning_rate (float): Learning rate for the optimizer.
        patience (int): Patience for early stopping.
        max_samples (int, optional): If set, limits the dataset size for debugging purposes.
        load_cached_data (bool): Whether to attempt loading preprocessed data from cache.
    """
    # Ensure reproducibility
    set_seeds(Config.SEED)

    # 1. Prepare Data Loaders
    # This handles loading JSONs, resizing, flattening, imputing, scaling, and caching.
    train_loader, val_loader, test_loader, test_ids = get_data_loaders(
        batch_size=batch_size,
        load_cached_data=load_cached_data,
        max_samples=max_samples,
    )

    # 2. Initialize Model
    # Create the Downsampled Dense Neural Network (D2N)
    model = D2N(
        input_dim=Config.INPUT_DIM,
        hidden_units=Config.HIDDEN_UNITS,
        dropout_rate=Config.DROPOUT_RATE,
    )

    # 3. Train Model
    # Execute the training loop. This function handles:
    # - Loss calculation (BCE) and Backpropagation (Adam)
    # - Validation against the hold-out set
    # - Early Stopping based on validation loss
    # - Saving the best model checkpoint
    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        patience=patience,
        lr=learning_rate,
    )

    # 4. Generate Submission
    # Use the trained model to predict on the test set and save the submission CSV.
    generate_submission(trained_model, test_loader, test_ids)
