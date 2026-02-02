import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import seed_everything
from library.data import get_dataloaders
from library.model import DualStreamNetwork


def run_inference(
    checkpoint_path=None, save_path=None, debug=False, batch_size=Config.BATCH_SIZE
):
    """
    Runs inference on the test set using the trained DualStreamNetwork.

    Args:
        checkpoint_path (str, optional): Path to the model checkpoint.
                                         Defaults to best_model.pth in Config.OUTPUT_DIR.
        save_path (str, optional): Path to save the submission CSV.
                                   Defaults to Config.SUBMISSION_PATH.
        debug (bool): If True, runs on a small subset of the test data.
        batch_size (int): Batch size for inference.

    Returns:
        pd.DataFrame: The submission dataframe containing predictions.
    """
    # 1. Setup Environment
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Set default paths if not provided
    if checkpoint_path is None:
        checkpoint_path = os.path.join(Config.OUTPUT_DIR, "best_model.pth")

    if save_path is None:
        save_path = Config.SUBMISSION_PATH

    # Ensure output directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")

    print(f"Initializing inference...")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device}")

    # 2. Prepare Data
    # We use get_dataloaders to ensure consistent preprocessing.
    # We only need the test_loader.
    # Note: get_dataloaders handles debug sampling internally if debug=True.
    _, _, test_loader = get_dataloaders(val_batch_size=batch_size, debug=debug)

    # Extract eeg_ids from the dataset metadata to ensure alignment
    # The loader iterates sequentially, so these IDs match the prediction order.
    test_dataset = test_loader.dataset
    test_ids = test_dataset.metadata["eeg_id"].values

    print(f"Loaded test data. Samples: {len(test_dataset)}")

    # 3. Load Model
    model = DualStreamNetwork().to(device)

    # Load weights
    try:
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
    except Exception as e:
        raise RuntimeError(f"Failed to load state dict: {e}")

    model.eval()

    # 4. Inference Loop
    all_probs = []

    print("Starting prediction loop...")
    with torch.no_grad():
        for i, inputs in enumerate(test_loader):
            # Unpack inputs (eeg, spec)
            eeg, spec = inputs
            eeg = eeg.to(device, non_blocking=True)
            spec = spec.to(device, non_blocking=True)

            # Forward pass
            logits = model((eeg, spec))

            # Apply Softmax to get probabilities (sum to 1)
            probs = F.softmax(logits, dim=1)

            # Move to CPU and store
            all_probs.append(probs.cpu().numpy())

    # 5. Post-processing
    # Concatenate all batches
    if len(all_probs) > 0:
        predictions = np.concatenate(all_probs, axis=0)
    else:
        # Handle empty case (unlikely)
        predictions = np.zeros((0, Config.NUM_CLASSES))

    # Verification
    if len(predictions) != len(test_ids):
        print(
            f"Warning: Number of predictions ({len(predictions)}) does not match number of IDs ({len(test_ids)})."
        )

    # Create Submission DataFrame
    submission = pd.DataFrame(predictions, columns=Config.CLASS_NAMES)
    submission.insert(0, "eeg_id", test_ids)

    # 6. Save Submission
    submission.to_csv(save_path, index=False)
    print(f"Submission saved to {save_path}")

    # Print first few rows for verification
    print(submission.head())

    return submission
