import torch
import numpy as np
import random
from torch.utils.data import DataLoader
from library.config import Config
from library.dataset import GNSSWindowDataset
from library.model import BiLSTMRegressor, train_model, generate_submission


def set_seed(seed):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def run_training(
    load_cached_data=True,
    batch_size=Config.BATCH_SIZE,
    epochs=Config.EPOCHS,
    learning_rate=Config.LEARNING_RATE,
):
    """
    Orchestrates the training pipeline: loading data, training the model, and generating submission.

    Args:
        load_cached_data (bool): If True, attempts to load preprocessed data from cache.
                                 If False or cache missing, re-computes data.
        batch_size (int): Batch size for training and inference.
        epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for the optimizer.
    """
    # 1. Setup Environment
    set_seed(Config.RANDOM_STATE)

    # Update Config with runtime arguments to ensure library functions use the correct values
    Config.BATCH_SIZE = batch_size
    Config.EPOCHS = epochs
    Config.LEARNING_RATE = learning_rate

    print(
        f"Running training with: Batch Size={batch_size}, Epochs={epochs}, LR={learning_rate}"
    )

    # 2. Data Loading
    # The GNSSWindowDataset internally handles the caching logic via preprocessing.load_dataset
    print("Initializing Datasets...")

    train_dataset = GNSSWindowDataset(mode="train", load_cached_data=load_cached_data)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    val_dataset = GNSSWindowDataset(mode="val", load_cached_data=load_cached_data)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Test dataset is loaded here to ensure pipeline continuity, though used later
    test_dataset = GNSSWindowDataset(mode="test", load_cached_data=load_cached_data)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # 3. Model Initialization
    print("Initializing Bi-LSTM Model...")
    # Determine input dimensions dynamically from the dataset features
    input_dim = train_dataset.features.shape[1]
    output_dim = len(Config.TARGET_COLUMNS)

    model = BiLSTMRegressor(
        input_dim=input_dim,
        hidden_dim=Config.HIDDEN_DIM,
        num_layers=Config.NUM_LAYERS,
        dropout=Config.DROPOUT,
        output_dim=output_dim,
    )

    # 4. Training Loop
    # train_model handles the optimization, loss calculation, metrics printing, and early stopping
    print("Starting Training Process...")
    trained_model = train_model(model, train_loader, val_loader)

    # 5. Submission Generation
    # generate_submission handles inference, coordinate reconstruction, and file saving
    print("Generating Submission...")
    generate_submission(trained_model, test_loader)

    print("Pipeline completed successfully.")
