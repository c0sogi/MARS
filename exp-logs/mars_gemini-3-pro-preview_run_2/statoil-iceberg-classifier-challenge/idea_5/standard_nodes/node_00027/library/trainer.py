import os
import torch
import numpy as np
from library.utils import seed_everything, get_device
from library.data_loader import get_dataloaders
from library.model import SAHCN, train_model, predict_and_submit


class LimitedLoader:
    """
    A wrapper around a DataLoader to limit the number of batches yielded.
    Used for debugging purposes to reduce runtime without modifying the source dataset.
    """

    def __init__(self, loader, limit):
        self.loader = loader
        self.limit = limit
        self.dataset = loader.dataset

    def __iter__(self):
        count = 0
        for batch in self.loader:
            if count >= self.limit:
                break
            yield batch
            count += 1

    def __len__(self):
        return min(len(self.loader), self.limit)


def run_task(epochs: int = 50, batch_size: int = 32, debug: bool = False):
    """
    Orchestrates the training, validation, and submission generation process.

    Args:
        epochs (int): Maximum number of training epochs.
        batch_size (int): Batch size for the DataLoaders.
        debug (bool): If True, limits the number of batches per epoch for rapid debugging.
    """
    # 1. Environment Setup
    seed_everything(42)
    device = get_device()
    print(f"Initializing task on device: {device}")

    # 2. Data Loading
    print("Loading datasets...")
    train_loader, val_loader, test_loader, ids_test = get_dataloaders(
        batch_size=batch_size
    )

    # 3. Debug Handling
    if debug:
        print("Debug mode enabled: Limiting processing to 5 batches per epoch.")
        limit_batches = 5

        # Wrap loaders to limit iteration
        train_loader = LimitedLoader(train_loader, limit_batches)
        val_loader = LimitedLoader(val_loader, limit_batches)
        test_loader = LimitedLoader(test_loader, limit_batches)

        # Slice test IDs to match the limited number of inference batches
        # Note: DataLoader may drop the last incomplete batch depending on config,
        # but for debug we assume roughly limit * batch_size samples.
        expected_samples = min(len(ids_test), limit_batches * batch_size)
        ids_test = ids_test[:expected_samples]

    # 4. Model Initialization
    print("Initializing SAHCN architecture...")
    model = SAHCN()

    # 5. Training Loop
    # Utilizes the imported train_model which implements the "Low and Slow" strategy,
    # ReduceLROnPlateau scheduler, and Early Stopping.
    print(f"Starting training for {epochs} epochs...")
    best_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        patience=30,  # Extended patience to allow convergence with low LR
        device=device,
    )

    # 6. Inference and Submission
    # Generates probabilities and saves to ./submission/submission.csv
    print("Generating predictions and submission file...")
    predict_and_submit(
        model=best_model,
        test_loader=test_loader,
        ids_test=ids_test,
        output_dir="./submission",
        device=device,
    )
