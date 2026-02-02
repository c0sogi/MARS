import os
import torch
import pandas as pd
import numpy as np
from typing import List

from library.config import Config
from library.model import CalibratedHierarchicalSeqModel
from library.data import get_test_dataloader
from library.utils import get_device, seed_everything


def predict_test_set(
    load_cached_data: bool = True,
    batch_size: int = Config.BATCH_SIZE,
    debug: bool = False,
) -> None:
    """
    Generates predictions for the test set and saves the submission file.

    Args:
        load_cached_data (bool): Whether to use cached file paths for the dataset.
        batch_size (int): Batch size for inference.
        debug (bool): If True, runs on a small subset of the data.
    """
    # 1. Update Configuration based on runtime arguments
    Config.BATCH_SIZE = batch_size
    Config.DEBUG = debug

    # Ensure reproducibility
    seed_everything(Config.SEED)

    device = get_device()
    print(f"Starting inference on device: {device}")

    # 2. Load Data
    # get_test_dataloader internally handles caching and transforms
    print("Loading test data...")
    test_loader = get_test_dataloader(load_cached_data=load_cached_data)

    # 3. Load Model
    print("Initializing model...")
    model = CalibratedHierarchicalSeqModel(
        pretrained=False
    )  # Pretrained weights not needed for backbone as we load checkpoint
    model = model.to(device)

    # Load checkpoint
    checkpoint_path = Config.BEST_MODEL_PATH
    if not os.path.exists(checkpoint_path):
        print(
            f"Warning: Best model not found at {checkpoint_path}. Checking for last checkpoint..."
        )
        checkpoint_path = Config.LAST_MODEL_PATH

    if os.path.exists(checkpoint_path):
        print(f"Loading weights from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        raise FileNotFoundError(
            f"No checkpoint found at {Config.BEST_MODEL_PATH} or {Config.LAST_MODEL_PATH}"
        )

    # 4. Inference Loop
    model.eval()
    all_probs = []

    print("Running inference...")
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(test_loader):
            images = images.to(device, dtype=torch.float32)

            # Forward pass (returns logits)
            logits = model(images)

            # Apply Sigmoid to get probabilities [0, 1]
            probs = torch.sigmoid(logits)

            # Move to CPU and store
            all_probs.append(probs.cpu().numpy())

            if debug and batch_idx >= 2:
                print("Debug mode: stopping after 3 batches.")
                break

    # Concatenate all batch predictions
    # Shape: (Num_Studies, 8)
    if len(all_probs) > 0:
        predictions = np.concatenate(all_probs, axis=0)
    else:
        predictions = np.array([])

    # 5. Format Submission
    print("Formatting submission...")

    # Retrieve StudyInstanceUIDs from the dataset to ensure alignment
    # The loader is not shuffled, so order matches the dataframe
    dataset_df = test_loader.dataset.df

    # If debug, slice the dataframe to match predictions
    if debug:
        dataset_df = dataset_df.iloc[: len(predictions)]

    study_uids = dataset_df["StudyInstanceUID"].values

    if len(study_uids) != len(predictions):
        raise ValueError(
            f"Mismatch between number of studies ({len(study_uids)}) and predictions ({len(predictions)})"
        )

    # Create a DataFrame with predictions
    # Columns correspond to Config.TARGET_COLS order
    pred_df = pd.DataFrame(predictions, columns=Config.TARGET_COLS)
    pred_df["StudyInstanceUID"] = study_uids

    # Melt to long format: [StudyInstanceUID, prediction_type, fractured]
    melted_df = pred_df.melt(
        id_vars=["StudyInstanceUID"],
        value_vars=Config.TARGET_COLS,
        var_name="prediction_type",
        value_name="fractured",
    )

    # Create 'row_id' column: StudyInstanceUID + "_" + prediction_type
    melted_df["row_id"] = (
        melted_df["StudyInstanceUID"] + "_" + melted_df["prediction_type"]
    )

    # Select final columns
    submission_df = melted_df[["row_id", "fractured"]]

    # 6. Save Submission
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)

    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(f"Total rows generated: {len(submission_df)}")
    print("Sample rows:")
    print(submission_df.head().to_string())
