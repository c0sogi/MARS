import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from library.config import Config
from library.dataset import IceCubeDataset
from library.model import SpatiotemporalPointTransformer
from library.utils import set_seed, vector_to_azimuth_zenith


def generate_submission(model, device, output_path):
    """
    Generates predictions for the test set and saves to CSV.

    Args:
        model: The trained PyTorch model.
        device: The torch device.
        output_path: Path to save the submission CSV.
    """
    print("Generating submission...")
    model.eval()

    # Load Test Dataset
    # subset_size=None ensures we process the full test set
    test_dataset = IceCubeDataset(mode="test", subset_size=None)

    # Create DataLoader
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    all_azimuths = []
    all_zeniths = []

    with torch.no_grad():
        for features, _ in test_loader:
            features = features.to(device)

            # Predict vectors (Batch, 3)
            pred_vectors = model(features)

            # Convert vectors back to angles
            azimuth, zenith = vector_to_azimuth_zenith(pred_vectors)

            all_azimuths.append(azimuth.cpu().numpy())
            all_zeniths.append(zenith.cpu().numpy())

    # Concatenate results
    if len(all_azimuths) > 0:
        all_azimuths = np.concatenate(all_azimuths)
        all_zeniths = np.concatenate(all_zeniths)
    else:
        all_azimuths = np.array([])
        all_zeniths = np.array([])

    # Get event IDs from dataset
    # IceCubeDataset loads all requested data into memory arrays, so indices align with loader
    event_ids = test_dataset.event_ids

    # Create DataFrame
    df = pd.DataFrame(
        {"event_id": event_ids, "azimuth": all_azimuths, "zenith": all_zeniths}
    )

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def predict(weights_path=None):
    """
    Main function to load model and generate submission.

    Args:
        weights_path (str, optional): Path to the model weights.
                                      Defaults to Config.WORKING_DIR/best_model.pth.
    """
    # Set seed for reproducibility
    set_seed(Config.SEED)

    device = torch.device(Config.DEVICE)

    # Initialize Model
    model = SpatiotemporalPointTransformer().to(device)

    # Determine weights path
    if weights_path is None:
        weights_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    print(f"Loading model weights from {weights_path}...")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights file not found at {weights_path}")

    # Load weights
    model.load_state_dict(torch.load(weights_path, map_location=device))

    # Generate Submission
    generate_submission(model, device, Config.SUBMISSION_PATH)
