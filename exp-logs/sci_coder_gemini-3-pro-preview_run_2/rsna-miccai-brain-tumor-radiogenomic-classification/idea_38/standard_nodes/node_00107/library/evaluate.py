import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import set_seed, get_device
from library.data_loader import process_dataset, load_subject_volume, BraTSDataset
from library.model import AsymmetricEfficientNet
from library.train import predict_tta


def predict_subject(subject_id, metadata_df):
    """
    Generates the input tensor for a test subject using the strict geometric pipeline.

    Args:
        subject_id (int): The BraTS21ID of the subject.
        metadata_df (pd.DataFrame): The metadata dataframe containing paths.

    Returns:
        torch.Tensor: The input tensor with shape (12, 224, 224).
    """
    # Load volume using the strict geometric pipeline defined in data_loader
    volume = load_subject_volume(subject_id, metadata_df)

    # Convert to tensor
    tensor = torch.tensor(volume, dtype=torch.float32)
    return tensor


def run_inference():
    """
    Iterates through the test dataset, generates predictions using TTA,
    and saves the submission file.
    """
    # 1. Setup
    set_seed(Config.SEED)
    device = get_device()

    # 2. Load Model
    model = AsymmetricEfficientNet().to(device)
    model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Best model not found at {model_path}. Please run training first."
        )

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 3. Prepare Test Data
    test_csv_path = os.path.join(Config.METADATA_DIR, "test.csv")
    if not os.path.exists(test_csv_path):
        raise FileNotFoundError(f"Test metadata not found at {test_csv_path}")

    # Use process_dataset to handle caching logic (load from cache or compute and save)
    # This fulfills the requirement for deterministic data processing caching
    test_data, _ = process_dataset(
        test_csv_path, f"test_cache_{Config.CACHE_VERSION}", load_cached_data=True
    )

    # Create dummy labels for the test set (not used for inference, but required by Dataset)
    test_labels = np.zeros(len(test_data), dtype=np.float32)

    # Create Dataset and DataLoader
    # transform=False because TTA handles geometric transformations manually
    test_dataset = BraTSDataset(test_data, test_labels, transform=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # 4. Inference with TTA
    predictions = []

    with torch.no_grad():
        for inputs, _ in test_loader:
            # predict_tta handles device transfer and averaging of [Original, HFlip, VFlip]
            batch_preds = predict_tta(model, inputs, device)

            # Flatten predictions and add to list
            predictions.extend(batch_preds.cpu().numpy().flatten())

    # 5. Generate Submission
    test_df = pd.read_csv(test_csv_path)

    # Ensure lengths match
    if len(predictions) != len(test_df):
        print(
            f"Warning: Number of predictions ({len(predictions)}) does not match number of test samples ({len(test_df)})."
        )

    submission = pd.DataFrame(
        {"BraTS21ID": test_df["BraTS21ID"], "MGMT_value": predictions}
    )

    # Save to file
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
