import os
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import seed_everything, get_device
from library.data_processing import prepare_dataloaders
from library.model import SC_GI_BiLSTM


def generate_submission(
    model_path: str = os.path.join(Config.WORKING_DIR, "best_model.pth"),
    batch_size: int = Config.BATCH_SIZE,
    debug: bool = Config.DEBUG,
):
    """
    Generates predictions for the test set and saves the submission file.

    Args:
        model_path (str): Path to the trained model weights.
        batch_size (int): Batch size for inference.
        debug (bool): Whether to run in debug mode (subset of data).
    """
    # 1. Setup
    seed_everything()
    device = get_device()

    print(f"Device: {device}")
    print("Loading test data...")

    # 2. Load Data
    # We use prepare_dataloaders to ensure consistent preprocessing and caching.
    # We only need the test_loader.
    _, _, test_loader = prepare_dataloaders(batch_size=batch_size, debug=debug)

    # 3. Initialize Model
    print("Initializing model architecture...")
    model = SC_GI_BiLSTM().to(device)

    # 4. Load Weights
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

    print(f"Loading model weights from {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 5. Inference Loop
    print("Starting inference...")
    predictions = []

    with torch.no_grad():
        for batch in test_loader:
            # test_loader returns only x tensor
            x = batch.to(device, dtype=torch.float32)

            # Forward pass: (Batch, Seq)
            preds = model(x)

            # Move to CPU, flatten, and store
            # We flatten because the submission format is one row per time step
            preds_flat = preds.cpu().numpy().flatten()
            predictions.extend(preds_flat)

    predictions = np.array(predictions)
    print(f"Generated {len(predictions)} predictions.")

    # 6. Format Submission
    print("Formatting submission...")

    # Load test metadata to map predictions to IDs
    # The model processes data sorted by [breath_id, time_step].
    # We must ensure metadata is sorted identically before assigning predictions.
    if not os.path.exists(Config.TEST_META):
        raise FileNotFoundError(f"Test metadata not found at {Config.TEST_META}")

    test_meta = pd.read_csv(Config.TEST_META)

    # Sort metadata to match model output order: breath_id, then id (proxy for time)
    test_meta = test_meta.sort_values(["breath_id", "id"])

    # Handle potential length mismatch (e.g., if debug mode was used or data truncated)
    if len(predictions) != len(test_meta):
        print(
            f"Note: Prediction count ({len(predictions)}) differs from metadata rows ({len(test_meta)})."
        )
        print(
            "Truncating metadata to match predictions (assuming debug/truncation logic)."
        )
        test_meta = test_meta.iloc[: len(predictions)]

    # Assign predictions
    test_meta["pressure"] = predictions

    # Sort by 'id' as required by the submission format
    submission = test_meta.sort_values("id")[["id", "pressure"]]

    # 7. Save Submission
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission.to_csv(Config.SUBMISSION_PATH, index=False)

    print(f"Submission saved to {Config.SUBMISSION_PATH}")
