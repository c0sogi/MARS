import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import seed_everything, TargetScaler
from library.dataset import VolcanoDataset
from library.model import VolcanoHybridModel


def generate_predictions(
    batch_size=Config.BATCH_SIZE, device=Config.DEVICE, load_cached_features=True
):
    """
    Generates predictions for the test set using the trained model.

    Args:
        batch_size (int): Batch size for the data loader.
        device (str): Device to run inference on ('cpu' or 'cuda').
        load_cached_features (bool): Whether to load pre-computed features from cache.
                                     (Handled implicitly by dataset/feature_engineering logic,
                                      but kept here for consistency with requirements).
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(device)
    print(f"Running inference on device: {device}")

    # 2. Prepare Data
    # We need the scaler to perform inverse transform on predictions.
    # The scaler will load the mean/std parameters saved during training.
    print("Initializing Target Scaler...")
    target_scaler = TargetScaler()

    print("Loading Test Dataset...")
    # Dataset initialization triggers feature loading/computation
    test_dataset = VolcanoDataset(
        metadata_path=Config.TEST_METADATA, mode="test", target_scaler=target_scaler
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Load Model
    print("Loading Model...")
    model = VolcanoHybridModel()

    if not os.path.exists(Config.MODEL_SAVE_PATH):
        raise FileNotFoundError(
            f"Model checkpoint not found at {Config.MODEL_SAVE_PATH}. "
            "Please run training first."
        )

    # Load weights
    state_dict = torch.load(Config.MODEL_SAVE_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 4. Inference Loop
    predictions = []
    segment_ids = []

    print(f"Starting inference on {len(test_dataset)} samples...")

    with torch.no_grad():
        for batch in test_loader:
            # Extract data
            spectrogram = batch["spectrogram"].to(device)
            features = batch["features"].to(device)
            ids = batch["segment_id"]

            # Forward pass
            # Outputs are scaled (StandardScaler applied during training)
            outputs = model(spectrogram, features)  # Shape: [Batch, 1]

            # Inverse Transform to get original time_to_eruption
            # returns tensor on same device
            outputs_original = target_scaler.inverse_transform(outputs)

            # Convert to numpy/list for storage
            outputs_np = outputs_original.cpu().numpy().flatten()

            predictions.extend(outputs_np.tolist())
            segment_ids.extend(ids.tolist())

    # 5. Format Submission
    print("Formatting submission...")
    df_submission = pd.DataFrame(
        {"segment_id": segment_ids, "time_to_eruption": predictions}
    )

    # Ensure segment_id is integer
    df_submission["segment_id"] = df_submission["segment_id"].astype(int)

    # Save to disk
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
    df_submission.to_csv(Config.SUBMISSION_PATH, index=False)

    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print("First 5 rows:")
    print(df_submission.head())
