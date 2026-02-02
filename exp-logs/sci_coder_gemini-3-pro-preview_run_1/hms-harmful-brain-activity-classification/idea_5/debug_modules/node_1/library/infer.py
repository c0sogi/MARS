import os
import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F

from library.config import Config
from library.utils import set_seed
from library.models import TriViewNet
from library.data import get_loaders


def inference(
    checkpoint_path: str = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth"),
    output_path: str = Config.SUBMISSION_PATH,
    device: str = str(Config.DEVICE),
    debug: bool = False,
    load_cached_data: bool = False,
):
    """
    Runs inference on the test set using the TriViewNet model.

    Args:
        checkpoint_path (str): Path to the trained model weights.
        output_path (str): Path to save the submission CSV.
        device (str): Computation device ('cpu' or 'cuda').
        debug (bool): If True, runs on a subset of the test data.
        load_cached_data (bool): If True, attempts to use cached pre-processed data.
    """
    # 1. Setup
    set_seed(Config.SEED)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Starting inference on device: {device}")

    # 2. Data Loading
    # We only need the test loader. get_loaders returns (train, val, test)
    _, _, test_loader = get_loaders(debug=debug, load_cached_data=load_cached_data)

    # 3. Model Initialization
    model = TriViewNet(num_classes=Config.NUM_CLASSES, pretrained=False)

    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    model.to(device)
    model.eval()

    # 4. Inference Loop
    all_probs = []

    with torch.no_grad():
        for batch in test_loader:
            # Move inputs to device
            micro = batch["micro"].to(device, non_blocking=True)
            meso = batch["meso"].to(device, non_blocking=True)
            macro = batch["macro"].to(device, non_blocking=True)

            # Forward pass
            logits = model(micro, meso, macro)

            # Convert logits to probabilities
            probs = F.softmax(logits, dim=1)

            # Move to CPU and store
            all_probs.append(probs.cpu().numpy())

    # Concatenate all batches
    all_probs = np.concatenate(all_probs, axis=0)

    # 5. Submission Generation
    # Retrieve EEG IDs from the dataset to ensure alignment
    # (DataLoader is not shuffled for test set)
    eeg_ids = test_loader.dataset.eeg_ids

    # Handle debug case where dataset might be smaller than full metadata
    if len(eeg_ids) != len(all_probs):
        # In debug mode, get_loaders samples the dataframe.
        # The dataset.eeg_ids should match the sampled data.
        # Just ensuring consistency.
        print(
            f"Warning: Number of predictions ({len(all_probs)}) differs from ID count ({len(eeg_ids)})."
        )

    # Map probability columns to vote columns required for submission
    # Config.TARGET_COLS are ['seizure_prob', 'lpd_prob', ...]
    # Submission requires ['seizure_vote', 'lpd_vote', ...]
    vote_cols = [col.replace("_prob", "_vote") for col in Config.TARGET_COLS]

    submission_df = pd.DataFrame(all_probs, columns=vote_cols)
    submission_df.insert(0, "eeg_id", eeg_ids)

    # Save to CSV
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
    print(f"Submission shape: {submission_df.shape}")

    # Verification
    # Ensure probabilities sum to 1 (Softmax guarantees this, but good for sanity check)
    row_sums = submission_df[vote_cols].sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-5):
        print("Warning: Not all row probabilities sum to 1.0.")
