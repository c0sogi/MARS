import torch
from torch.utils.data import TensorDataset, DataLoader
from library.config import Config
from library.data_utils import get_dataloaders
from library.model import train_model, predict


def run_training(
    epochs=None, batch_size=None, load_cached_data=True, max_train_samples=None
):
    """
    Orchestrates the training and evaluation pipeline.

    Args:
        epochs (int, optional): Override the number of training epochs.
        batch_size (int, optional): Override the batch size.
        load_cached_data (bool): Whether to load pre-processed data from cache.
        max_train_samples (int, optional): If set, truncates the training set for debugging.
    """
    # 1. Apply Configuration Overrides
    if epochs is not None:
        Config.EPOCHS = epochs
    if batch_size is not None:
        Config.BATCH_SIZE = batch_size

    print(
        f"Configuration: Epochs={Config.EPOCHS}, Batch Size={Config.BATCH_SIZE}, Device={Config.DEVICE}"
    )

    # 2. Load Data
    # get_dataloaders handles caching and preprocessing internally via process_data
    train_loader, val_loader, test_loader, test_ids, input_dim = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=load_cached_data,
    )

    # 3. Handle Debugging (Dataset Subsampling)
    # Since we cannot modify library files, we intervene here if max_train_samples is set.
    if max_train_samples is not None and max_train_samples < len(train_loader.dataset):
        print(f"Debugging: Truncating training set to {max_train_samples} samples.")

        # Access underlying tensors from the TensorDataset
        full_X, full_y = train_loader.dataset.tensors

        # Slice tensors
        small_X = full_X[:max_train_samples]
        small_y = full_y[:max_train_samples]

        # Create new Dataset and Loader
        small_dataset = TensorDataset(small_X, small_y)

        pin_memory = Config.DEVICE == "cuda"
        train_loader = DataLoader(
            small_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=pin_memory,
        )

    # 4. Train Model
    # train_model handles model instantiation, optimization, scheduling, and early stopping
    model = train_model(train_loader, val_loader, input_dim)

    # 5. Generate Predictions
    # predict handles inference and saving to CSV
    predict(model, test_loader, test_ids)

    print("Pipeline completed successfully.")
