import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from library.config import Config
from library.data import get_dataloaders
from library.model import BiGRUModel
from library.utils import load_checkpoint


def generate_submission(
    device: str = Config.DEVICE,
    batch_size: int = Config.BATCH_SIZE,
    load_cached_data: bool = False,
):
    """
    Generates the submission file for the test set.

    Args:
        device (str): Computation device ('cpu' or 'cuda').
        batch_size (int): Batch size for inference.
        load_cached_data (bool): Whether to use cached data for dataloaders.
    """
    print("Starting submission generation...")

    # 1. Prepare Data
    # We only need the test_loader. get_dataloaders handles reading metadata and creating datasets.
    _, _, test_loader = get_dataloaders(load_cached_data=load_cached_data)

    # 2. Initialize Model
    model = BiGRUModel()
    model.to(device)

    # 3. Load Checkpoint
    # Try loading best model first, then fallback to the latest checkpoint
    checkpoint = load_checkpoint(model, filename="best_model.pth", device=device)
    if checkpoint is None:
        print("Best model not found, trying last checkpoint...")
        checkpoint = load_checkpoint(model, filename="checkpoint.pth", device=device)

    if checkpoint is None:
        raise FileNotFoundError("No checkpoint found. Cannot generate submission.")

    print(
        f"Loaded model from epoch {checkpoint.get('epoch', 'Unknown')}, Val Loss: {checkpoint.get('val_loss', 'Unknown')}"
    )

    # 4. Inference Loop
    model.eval()
    all_probs = []

    print("Running inference on test set...")
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)

            # Forward pass
            logits = model(data)

            # Apply Softmax to get probabilities (ensures sum to 1)
            probs = F.softmax(logits, dim=1)

            # Move to CPU and store
            all_probs.append(probs.cpu().numpy())

    # Concatenate all batches into a single array
    all_probs = np.concatenate(all_probs, axis=0)

    # 5. Prepare Submission DataFrame
    # Retrieve eeg_ids directly from the dataset metadata to ensure alignment.
    # The test_loader.dataset is an EEGDataset instance which has a .metadata attribute.
    # This handles cases where DEBUG mode might have truncated the dataset.
    test_ids = test_loader.dataset.metadata["eeg_id"].values

    # Ensure we have the same number of predictions as IDs
    if len(test_ids) != len(all_probs):
        raise ValueError(
            f"Mismatch: {len(test_ids)} IDs vs {len(all_probs)} predictions."
        )

    # Create DataFrame with correct column names
    submission_df = pd.DataFrame(all_probs, columns=Config.SUBMISSION_COLS)
    submission_df.insert(0, "eeg_id", test_ids)

    # 6. Save Submission
    save_path = Config.SUBMISSION_PATH
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    submission_df.to_csv(save_path, index=False)
    print(f"Submission saved to {save_path}")
    print("First 5 rows:")
    print(submission_df.head())
