import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import (
    TEST_META_PATH,
    MODEL_PATH,
    SUBMISSION_PATH,
    DEVICE,
    BATCH_SIZE,
    NUM_WORKERS,
    SEED,
)
from library.utils import seed_everything
from library.data import get_dataset
from library.model import SSBHDNetwork


def generate_submission(
    test_meta_path=TEST_META_PATH,
    model_path=MODEL_PATH,
    submission_output_path=SUBMISSION_PATH,
    batch_size=BATCH_SIZE,
    load_cached_data=True,
):
    """
    Generates the submission file for the test dataset.

    Args:
        test_meta_path (str): Path to the test metadata parquet file.
        model_path (str): Path to the trained model checkpoint.
        submission_output_path (str): Path to save the submission CSV.
        batch_size (int): Batch size for inference.
        load_cached_data (bool): Whether to use cached preprocessed data.
    """
    # 1. Set seed for reproducibility
    seed_everything(SEED)

    print(f"Generating submission using model at: {model_path}")
    print(f"Device: {DEVICE}")

    # 2. Load Test Dataset
    # get_dataset handles caching, loading metadata, and preprocessing (normalization/resizing)
    # It returns (image_tensor, patient_id) tuples for the test set.
    test_dataset = get_dataset(
        metadata_path=test_meta_path,
        dataset_type="test",
        load_cached_data=load_cached_data,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    print(f"Test samples: {len(test_dataset)}")

    # 3. Load Model
    model = SSBHDNetwork()
    model = model.to(DEVICE)

    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(state_dict)
        print("Model weights loaded successfully.")
    else:
        print(
            f"WARNING: Model file not found at {model_path}. Predictions will be random (for debugging)."
        )

    model.eval()

    # 4. Inference
    predictions = []
    ids = []

    print("Starting inference...")
    with torch.no_grad():
        for images, batch_ids in test_loader:
            images = images.to(DEVICE)

            # Forward pass
            logits = model(images)

            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(logits)

            # Collect results
            # Flatten to ensure 1D array of probabilities
            predictions.extend(probs.cpu().numpy().flatten())
            ids.extend(batch_ids)

    # 5. Create Submission DataFrame
    submission_df = pd.DataFrame({"BraTS21ID": ids, "MGMT_value": predictions})

    # Ensure output directory exists
    os.makedirs(os.path.dirname(submission_output_path), exist_ok=True)

    # Save to CSV
    submission_df.to_csv(submission_output_path, index=False)
    print(f"Submission saved to {submission_output_path}")
    print("First 5 predictions:")
    print(submission_df.head())
