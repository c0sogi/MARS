import os
import pandas as pd
import torch
import torch.nn.functional as F

from library.config import (
    DEVICE,
    TEST_META_PATH,
    SUBMISSION_DIR,
    WORKING_DIR,
    BATCH_SIZE,
    NUM_WORKERS,
    SEED,
    seed_everything,
)
from library.model import TemporalCNN
from library.data_loader import get_dataloader
from library.utils import direction_to_angles


def predict_and_submit(model_path=None, batch_size=BATCH_SIZE):
    """
    Generates predictions for the test set using the trained model and saves the submission CSV.

    Args:
        model_path (str, optional): Path to the trained model checkpoint.
                                    Defaults to ./working/idea_3/best_model.pth.
        batch_size (int, optional): Batch size for inference. Defaults to config BATCH_SIZE.
    """
    # Ensure reproducibility
    seed_everything(SEED)

    # Determine model path
    if model_path is None:
        model_path = os.path.join(WORKING_DIR, "best_model.pth")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")

    print(f"Loading model from {model_path}...")

    # Initialize and load model
    model = TemporalCNN().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    print("Initializing Test DataLoader...")
    # Initialize DataLoader for the full test set
    test_loader = get_dataloader(
        metadata_path=TEST_META_PATH,
        mode="test",
        max_samples=None,
        batch_size=batch_size,
        num_workers=NUM_WORKERS,
    )

    all_event_ids = []
    all_azimuths = []
    all_zeniths = []

    print("Starting inference on Test Set...")

    with torch.no_grad():
        for inputs, event_ids in test_loader:
            inputs = inputs.to(DEVICE)

            # Forward pass
            preds = model(inputs)

            # Normalize predictions to unit vectors to ensure valid angular conversion
            preds_norm = F.normalize(preds, p=2, dim=1)

            # Convert Cartesian vectors to Spherical angles
            az, zen = direction_to_angles(
                preds_norm[:, 0], preds_norm[:, 1], preds_norm[:, 2]
            )

            # Collect results
            all_event_ids.extend(event_ids.numpy())
            all_azimuths.extend(az.cpu().numpy())
            all_zeniths.extend(zen.cpu().numpy())

    # Create Submission DataFrame
    submission_df = pd.DataFrame(
        {"event_id": all_event_ids, "azimuth": all_azimuths, "zenith": all_zeniths}
    )

    # Sort by event_id to match expected submission format
    submission_df.sort_values("event_id", inplace=True)

    # Ensure output directory exists
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    out_path = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Save to CSV
    submission_df.to_csv(out_path, index=False)
    print(f"Submission saved to {out_path} with shape {submission_df.shape}")
