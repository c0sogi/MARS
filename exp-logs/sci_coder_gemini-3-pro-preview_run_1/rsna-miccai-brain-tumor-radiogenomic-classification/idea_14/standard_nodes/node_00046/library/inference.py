import os
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import set_seed
from library.data import get_dataloader
from library.model import WITSNet


def predict(model_path, metadata_path, device_name=Config.DEVICE):
    """
    Loads the trained model and generates slab-level predictions for the dataset
    specified by the metadata.

    Args:
        model_path (str): Path to the .pth model file.
        metadata_path (str): Path to the metadata CSV file.
        device_name (str): Device to run inference on ('cpu' or 'cuda').

    Returns:
        pd.DataFrame: DataFrame containing 'BraTS21ID' and 'prob' for each slab.
    """
    device = torch.device(device_name)

    # 1. Load Metadata
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

    df_test = pd.read_csv(metadata_path)

    # 2. Create DataLoader
    # mode="test" ensures:
    # - No geometric augmentations (only ToTensor)
    # - Caching uses 'test' prefix (e.g., cache_test_images.npy)
    loader = get_dataloader(
        df_test,
        mode="test",
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
    )

    # 3. Initialize Model
    model = WITSNet()
    model.to(device)

    # 4. Load Weights
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(
            f"WARNING: Model path {model_path} does not exist. Predictions will be random (untrained model)."
        )

    model.eval()

    all_probs = []
    all_ids = []

    # 5. Inference Loop
    with torch.no_grad():
        for images, _, ids in loader:
            images = images.to(device)

            # Forward pass
            logits = model(images)

            # Convert logits to probabilities
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            all_probs.extend(probs)
            all_ids.extend(ids.numpy().flatten())

    # 6. Return Slab-Level Results
    return pd.DataFrame({"BraTS21ID": all_ids, "prob": all_probs})


def generate_submission(load_cached_data=True):
    """
    End-to-end inference pipeline.
    1. Predicts probabilities for all slabs in the test set.
    2. Aggregates slab predictions (mean) to get subject-level predictions.
    3. Saves the result to submission.csv.

    Args:
        load_cached_data (bool): If True, the data loader will attempt to load
                                 pre-processed numpy arrays from disk.
    """
    # Set seed for reproducibility
    set_seed(Config.SEED)

    # Update Config based on argument (ensures data loader respects the flag)
    Config.LOAD_CACHED_DATA = load_cached_data

    # Define paths
    model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    metadata_path = Config.TEST_METADATA_PATH
    submission_path = Config.SUBMISSION_PATH

    print(f"Generating submission...")
    print(f"Model: {model_path}")
    print(f"Metadata: {metadata_path}")

    # 1. Run Prediction
    df_slabs = predict(model_path, metadata_path)

    # 2. Aggregate Predictions
    # The WITS-Net strategy generates 3 independent slabs per subject.
    # We average the probabilities to get the final subject prediction.
    df_submission = df_slabs.groupby("BraTS21ID")["prob"].mean().reset_index()
    df_submission.rename(columns={"prob": "MGMT_value"}, inplace=True)

    # 3. Save Submission
    os.makedirs(os.path.dirname(submission_path), exist_ok=True)
    df_submission.to_csv(submission_path, index=False)

    print(f"Submission saved to {submission_path}")
    print(f"Total subjects predicted: {len(df_submission)}")

    # Print first few rows for verification
    print(df_submission.head())
