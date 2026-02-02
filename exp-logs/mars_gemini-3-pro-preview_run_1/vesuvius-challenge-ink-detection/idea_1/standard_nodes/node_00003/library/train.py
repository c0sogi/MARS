import os
from library.config import NUM_EPOCHS, BATCH_SIZE
from library.utils import set_seed
from library.data import get_dataloaders
from library.model import train_model, optimize_threshold, predict_and_submit


def run_training(
    epochs=NUM_EPOCHS,
    batch_size=BATCH_SIZE,
    limit_samples=None,
    load_cached_data=True,
    patience=3,
):
    """
    Orchestrates the training, validation, and submission pipeline for the Ink Detection task.

    Args:
        epochs (int): Maximum number of training epochs.
        batch_size (int): Batch size for data loaders.
        limit_samples (int, optional): Limit the number of samples for debugging purposes.
        load_cached_data (bool): Whether to load pre-processed .npy files from cache.
        patience (int): Number of epochs to wait for improvement before early stopping.

    Returns:
        model: The trained PyTorch model.
    """
    # 1. Ensure Reproducibility
    set_seed(42)

    # 2. Data Loading
    # get_dataloaders handles the caching mechanism (checking ./working/idea_1)
    # and loads data based on metadata in ./metadata
    dataloaders = get_dataloaders(
        batch_size=batch_size,
        limit_samples=limit_samples,
        load_cached_data=load_cached_data,
    )

    # 3. Model Training
    # Encapsulates the training loop, validation, and early stopping
    model = train_model(dataloaders, epochs=epochs, patience=patience)

    # 4. Threshold Optimization
    # Determine the optimal threshold using the validation set to maximize F0.5 score
    best_threshold = 0.5
    if "val" in dataloaders:
        best_threshold = optimize_threshold(model, dataloaders["val"])

    # 5. Inference and Submission
    # Generate predictions for the test set and save to submission.csv
    if "test" in dataloaders:
        predict_and_submit(model, dataloaders["test"], threshold=best_threshold)

    return model
