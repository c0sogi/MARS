import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import set_seed, cartesian_to_spherical
from library.data_processing import IceCubeDataset, collate_fn
from library.model_architecture import DynGTNet


def generate_submission(batch_size=None, num_workers=None):
    """
    Generates predictions for the test set using the trained DynGTNet model
    and saves the submission file to the working directory.

    Args:
        batch_size (int, optional): Batch size for inference. Defaults to Config.BATCH_SIZE.
        num_workers (int, optional): Number of workers for DataLoader. Defaults to Config.NUM_WORKERS.

    Returns:
        pd.DataFrame: The submission dataframe containing event_id, azimuth, and zenith.
    """
    # 1. Setup and Configuration
    Config.setup()
    set_seed(Config.SEED)

    if batch_size is None:
        batch_size = Config.BATCH_SIZE
    if num_workers is None:
        num_workers = Config.NUM_WORKERS

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on device: {device}")

    # 2. Prepare Data
    # IceCubeDataset handles metadata loading, caching, and stratified sampling internally.
    print("Initializing Test Dataset...")
    test_dataset = IceCubeDataset(mode="test")

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # 3. Load Model
    model_path = os.path.join(Config.MODEL_DIR, "best_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model checkpoint not found at {model_path}. Please train the model first."
        )

    print(f"Loading model from {model_path}...")
    model = DynGTNet()

    # Load state dict
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    # 4. Inference Loop
    all_event_ids = []
    all_azimuths = []
    all_zeniths = []

    print(f"Starting inference on {len(test_dataset)} events...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            # Move inputs to device
            x = batch["x"].to(device)
            event_ids = batch["event_ids"]

            # Forward pass: returns (Batch, 3) cartesian vectors
            pred_vecs = model(x)

            # Move to CPU for processing
            pred_vecs = pred_vecs.cpu()

            # Extract components
            pred_x = pred_vecs[:, 0]
            pred_y = pred_vecs[:, 1]
            pred_z = pred_vecs[:, 2]

            # Convert to spherical coordinates (azimuth, zenith)
            # Returns numpy arrays or tensors depending on input; here tensors on CPU
            az, zen = cartesian_to_spherical(pred_x, pred_y, pred_z)

            # Store results
            all_event_ids.extend(event_ids)
            all_azimuths.extend(az.numpy())
            all_zeniths.extend(zen.numpy())

            if (batch_idx + 1) % 100 == 0:
                print(f"Processed batch {batch_idx + 1}/{len(test_loader)}")

    # 5. Create Submission DataFrame
    df_submission = pd.DataFrame(
        {"event_id": all_event_ids, "azimuth": all_azimuths, "zenith": all_zeniths}
    )

    # Ensure event_id is sorted (optional but good practice)
    df_submission = df_submission.sort_values("event_id")

    # 6. Save to CSV
    save_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Ensure directory exists (redundant with Config.setup but safe)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    df_submission.to_csv(save_path, index=False)
    print(f"Submission saved successfully to {save_path}")
    print(f"Total events predicted: {len(df_submission)}")

    return df_submission
