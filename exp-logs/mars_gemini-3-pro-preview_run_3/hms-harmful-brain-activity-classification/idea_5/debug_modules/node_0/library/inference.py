import os
import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader

import library.config as config
from library.dataset import HMSDataset
from library.model import HybridModel
from library.utils import seed_everything, load_checkpoint


def predict(
    batch_size=config.BATCH_SIZE,
    num_workers=config.NUM_WORKERS,
    load_cached_data=True,
    device=config.DEVICE,
):
    """
    Runs inference on the test set and generates the submission file.

    Args:
        batch_size (int): Batch size for inference.
        num_workers (int): Number of dataloader workers.
        load_cached_data (bool): Whether to use cached preprocessed data.
        device (str): Device to run inference on ('cpu' or 'cuda').

    Returns:
        pd.DataFrame: The generated submission dataframe.
    """
    seed_everything(config.SEED)

    # ==========================
    # 1. Data Loading
    # ==========================
    print("Initializing Test Dataset...")
    # HMSDataset handles caching logic internally based on load_cached_data
    test_dataset = HMSDataset(
        csv_file=config.TEST_META_PATH,
        mode="test",
        augment=False,
        load_cached_data=load_cached_data,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # ==========================
    # 2. Model Loading
    # ==========================
    print("Initializing Model...")
    model = HybridModel()
    model.to(device)

    print(f"Loading checkpoint from {config.MODEL_PATH}...")
    try:
        # load_checkpoint handles loading state_dict from the dictionary
        load_checkpoint(model, config.MODEL_PATH, device)
    except FileNotFoundError:
        print(
            f"Warning: Checkpoint not found at {config.MODEL_PATH}. Predictions will be random (Debug Mode)."
        )

    model.eval()

    # ==========================
    # 3. Inference Loop
    # ==========================
    print("Starting inference...")
    all_probs = []

    with torch.no_grad():
        for batch_idx, (eeg, spec) in enumerate(test_loader):
            eeg = eeg.to(device, non_blocking=True)
            spec = spec.to(device, non_blocking=True)

            # Forward pass (returns logits)
            logits = model(eeg, spec)

            # Apply Softmax to get probabilities summing to 1
            probs = F.softmax(logits, dim=1)

            all_probs.append(probs.cpu().numpy())

    # Concatenate predictions from all batches
    if len(all_probs) > 0:
        predictions = np.concatenate(all_probs, axis=0)
    else:
        # Handle empty test set edge case
        predictions = np.zeros((0, config.NUM_CLASSES))

    # ==========================
    # 4. Submission Generation
    # ==========================
    print("Generating submission file...")

    # Load test metadata to get eeg_ids
    test_df = pd.read_csv(config.TEST_META_PATH)

    # Prepare DataFrame
    submission_df = pd.DataFrame()
    submission_df["eeg_id"] = test_df["eeg_id"]

    # Map target columns from internal config names (_prob) to submission names (_vote)
    # e.g., 'seizure_prob' -> 'seizure_vote'
    submission_cols = [col.replace("_prob", "_vote") for col in config.TARGET_COLS]

    # Assign predictions
    submission_df[submission_cols] = predictions

    # Save to disk
    os.makedirs(os.path.dirname(config.SUBMISSION_PATH), exist_ok=True)
    submission_df.to_csv(config.SUBMISSION_PATH, index=False)

    print(f"Submission saved to {config.SUBMISSION_PATH}")
    print(f"Submission shape: {submission_df.shape}")

    return submission_df
