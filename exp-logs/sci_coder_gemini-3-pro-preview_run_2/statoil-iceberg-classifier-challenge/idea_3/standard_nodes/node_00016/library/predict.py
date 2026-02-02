import torch
import numpy as np
from library.config import Config
from library.model import RHTN
from library.data_loader import get_dataloaders
from library.utils import load_checkpoint, save_submission


def generate_submission():
    """
    Loads the best trained model, performs inference on the test set,
    and generates a submission CSV file.
    """
    # 1. Setup Device
    device = torch.device(Config.DEVICE)
    print(f"Running inference on device: {device}")

    # 2. Load Data
    # We only need the test_loader and test_ids
    print("Loading test data...")
    _, _, test_loader, test_ids = get_dataloaders(load_cached_data=True)

    # 3. Initialize Model and Load Weights
    print(f"Loading model from {Config.MODEL_SAVE_PATH}...")
    model = RHTN()

    # Load weights onto the CPU first (or directly to device if supported by utils)
    # The utility function handles map_location
    model = load_checkpoint(model, Config.MODEL_SAVE_PATH, device=Config.DEVICE)

    # Move model to the computation device
    model = model.to(device)
    model.eval()

    # 4. Inference Loop
    all_preds = []
    print("Starting inference...")

    with torch.no_grad():
        for inputs, meta in test_loader:
            # Move data to device
            inputs = inputs.to(device)
            meta = meta.to(device)

            # Forward pass
            logits = model(inputs, meta)

            # Apply Sigmoid to get probabilities [0, 1]
            probs = torch.sigmoid(logits)

            # Move to CPU and convert to numpy
            # Flatten to 1D array
            batch_preds = probs.cpu().numpy().flatten()
            all_preds.extend(batch_preds)

    # 5. Save Submission
    all_preds = np.array(all_preds)

    print(f"Generated {len(all_preds)} predictions.")
    print(f"Saving submission to {Config.SUBMISSION_PATH}...")

    save_submission(test_ids, all_preds, Config.SUBMISSION_PATH)
    print("Submission saved successfully.")
