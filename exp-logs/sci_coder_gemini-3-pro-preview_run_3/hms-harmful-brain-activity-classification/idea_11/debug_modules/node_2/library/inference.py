import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import seed_everything
from library.data import load_data, EEGDataset
from library.model import BandAdaptiveNet


def predict(debug=False):
    """
    Runs inference on the test set and generates the submission file.

    Args:
        debug (bool): If True, runs on a small subset of data for testing.
    """
    seed_everything(Config.SEED)

    print(f"Starting inference on device: {Config.DEVICE}")

    # 1. Load Test Data
    # load_data handles caching. It returns None for targets in test mode.
    test_eeg, test_spec, _ = load_data(mode="test", load_cached_data=True)

    # Load test metadata to get eeg_ids for the submission file
    df_test = pd.read_csv(Config.TEST_CSV)

    if debug:
        print("DEBUG Mode: Truncating test data.")
        test_eeg = test_eeg[:100]
        test_spec = test_spec[:100]
        df_test = df_test.head(100)

    # 2. Prepare Dataset and Loader
    test_dataset = EEGDataset(test_eeg, test_spec, targets=None, mode="test")

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    # 3. Load Model
    model = BandAdaptiveNet()
    model.to(Config.DEVICE)

    model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model weights not found at {model_path}. Please train the model first."
        )

    print(f"Loading model weights from {model_path}...")
    state_dict = torch.load(model_path, map_location=Config.DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    # 4. Inference Loop
    all_probs = []

    print("Running prediction loop...")
    with torch.no_grad():
        for step, (x_eeg, x_spec) in enumerate(test_loader):
            x_eeg = x_eeg.to(Config.DEVICE)
            x_spec = x_spec.to(Config.DEVICE)

            # Mixed precision inference
            with torch.amp.autocast(device_type="cuda", enabled=Config.USE_AMP):
                logits = model(x_eeg, x_spec)
                probs = F.softmax(logits, dim=1)

            all_probs.append(probs.cpu().numpy())

    # Concatenate predictions
    final_probs = np.concatenate(all_probs, axis=0)

    # 5. Format Submission
    # The submission requires columns ending in _vote, but our targets are _prob.
    # We map them accordingly.
    submission_cols = [col.replace("_prob", "_vote") for col in Config.TARGET_COLS]

    submission_df = pd.DataFrame(final_probs, columns=submission_cols)
    submission_df.insert(0, "eeg_id", df_test["eeg_id"])

    # Ensure output directory exists
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Save submission
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(f"Submission shape: {submission_df.shape}")
    print(submission_df.head())

    return submission_df
