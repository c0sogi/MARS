import os
import torch
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.feature_engineering import FeatureEngineer
from library.dataset import GestureDataset
from library.model import CascadedNet
from library.utils import decode_predictions, write_submission_csv

# Set fixed seeds for reproducibility
torch.manual_seed(Config.SEED)
np.random.seed(Config.SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(Config.SEED)


def generate_test_predictions(load_cached_data=True, max_samples=None, device=None):
    """
    Generates predictions for the test dataset using the trained CascadedNet model.

    Args:
        load_cached_data (bool): Whether to load pre-computed features from cache.
        max_samples (int, optional): Limit the number of samples for debugging.
        device (torch.device, optional): Device to run inference on.

    Returns:
        dict: A dictionary mapping sample_ids to lists of predicted gesture IDs.
    """

    # 1. Device Configuration
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Inference process initialized on device: {device}")

    # 2. Data Loading
    # We use FeatureEngineer directly to avoid loading train/val data unnecessarily
    fe = FeatureEngineer()

    # Process test data (handles caching internally)
    test_data_dict = fe.process_dataset(
        metadata_path=Config.TEST_METADATA_PATH,
        cache_path=Config.TEST_CACHE_PATH,
        load_cached_data=load_cached_data,
        max_samples=max_samples,
    )

    # Create Dataset and DataLoader
    # is_train=False ensures we get full sequences and sample_ids
    test_dataset = GestureDataset(test_data_dict, is_train=False)

    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True
    )

    # 3. Model Loading
    model = CascadedNet().to(device)

    if not os.path.exists(Config.MODEL_SAVE_PATH):
        raise FileNotFoundError(
            f"Model checkpoint not found at {Config.MODEL_SAVE_PATH}. Cannot run inference."
        )

    print(f"Loading model weights from {Config.MODEL_SAVE_PATH}...")
    checkpoint = torch.load(Config.MODEL_SAVE_PATH, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    # 4. Inference Loop
    predictions_dict = {}

    print("Starting inference...")
    with torch.no_grad():
        for i, (features, _, sample_ids) in enumerate(test_loader):
            # features: (Batch=1, Time, Input_Dim)
            features = features.to(device)

            # Forward pass through the Cascaded Network
            # We use the Stage 2 logits for the final refined prediction
            _, s2_logits = model(features)

            # s2_logits: (Batch=1, Time, Num_Classes)
            # Remove batch dimension
            logits_seq = s2_logits.squeeze(0)  # (Time, Num_Classes)

            # Decode frame-wise logits into gesture sequence
            # Using min_len=5 to filter out very short spurious detections
            predicted_gestures = decode_predictions(logits_seq, min_len=5)

            # Extract sample_id (tuple of size 1)
            sample_id = sample_ids[0]
            predictions_dict[sample_id] = predicted_gestures

    # 5. Save Submission
    print(f"Generating submission file for {len(predictions_dict)} sequences...")
    write_submission_csv(predictions_dict, Config.SUBMISSION_FILE_PATH)

    return predictions_dict
