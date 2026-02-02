import os
import torch
from library.config import (
    DEVICE,
    MODEL_SAVE_PATH,
    SUBMISSION_PATH,
    BATCH_SIZE,
    NUM_WORKERS,
)
from library.utils import set_seed, load_model
from library.dataset import get_dataloaders
from library.network import IcebergResNet
from library.trainer import predict_and_submit as _predict_and_submit_internal


def predict_and_submit(
    load_cached_data=True, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
):
    """
    Loads the best saved model checkpoint, runs inference on the test dataloader,
    and formats the results into a submission CSV file.

    Args:
        load_cached_data (bool): Whether to load preprocessed data from cache.
        batch_size (int): Batch size for inference.
        num_workers (int): Number of workers for data loading.
    """
    # Ensure reproducibility
    set_seed()

    print("Preparing for inference...")

    # 1. Get DataLoaders
    # We unpack only what we need: test_loader and test_ids
    # get_dataloaders returns: train_loader, val_loader, test_loader, test_ids
    _, _, test_loader, test_ids = get_dataloaders(
        batch_size=batch_size,
        num_workers=num_workers,
        load_cached_data=load_cached_data,
    )

    # 2. Initialize Model Architecture
    print("Initializing model architecture...")
    model = IcebergResNet().to(DEVICE)

    # 3. Load Best Model Weights
    if os.path.exists(MODEL_SAVE_PATH):
        print(f"Loading model weights from {MODEL_SAVE_PATH}...")
        model = load_model(model, MODEL_SAVE_PATH, device=DEVICE)
    else:
        print(
            f"Warning: Model checkpoint not found at {MODEL_SAVE_PATH}. Predictions will be generated using random weights."
        )

    # 4. Generate Predictions and Save Submission
    # We use the utility from library.trainer which handles the inference loop,
    # sigmoid output processing, and CSV saving.
    print(f"Running inference and saving to {SUBMISSION_PATH}...")
    _predict_and_submit_internal(model, test_loader, test_ids, SUBMISSION_PATH)
