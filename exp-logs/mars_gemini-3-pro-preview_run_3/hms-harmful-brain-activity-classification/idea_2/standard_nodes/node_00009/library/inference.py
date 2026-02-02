import os
import torch
import pandas as pd
import numpy as np
from library import config, model, data, utils


def predict(debug=False):
    """
    Executes the inference pipeline: loads the trained model, generates predictions
    on the test set, and saves the result to a submission CSV file.

    Args:
        debug (bool): If True, runs inference on a small subset of the test data.
    """
    # Ensure reproducibility
    utils.seed_everything(config.SEED)

    # Setup device
    device = torch.device(config.DEVICE)
    print(f"Running inference on device: {device}")

    # Load DataLoaders
    # We only need the test_loader
    _, _, test_loader = data.get_dataloaders(
        train_batch_size=config.BATCH_SIZE,
        val_batch_size=config.BATCH_SIZE,
        debug=debug,
    )

    # Initialize Model architecture
    # pretrained=False is used because we are loading specific trained weights,
    # not the generic ImageNet weights.
    net = model.EEGWaveNet(pretrained=False)

    # Load trained model weights
    if not os.path.exists(config.MODEL_PATH):
        raise FileNotFoundError(f"Model weights not found at {config.MODEL_PATH}")

    state_dict = torch.load(config.MODEL_PATH, map_location=device)
    net.load_state_dict(state_dict)

    net.to(device)
    net.eval()

    all_probs = []

    print("Starting prediction loop...")
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)

            # Forward pass to get logits
            logits = net(inputs)

            # Apply Softmax to convert logits to probabilities
            # dim=1 corresponds to the class dimension
            probs = torch.softmax(logits, dim=1)

            all_probs.append(probs.cpu().numpy())

    # Concatenate all batch results
    if all_probs:
        final_probs = np.concatenate(all_probs)
    else:
        final_probs = np.zeros((0, config.NUM_CLASSES))

    # Load Test Metadata to retrieve EEG IDs
    test_df = pd.read_csv(config.TEST_CSV)

    # Adjust metadata if in debug mode to match the loader's subset
    if debug:
        test_df = test_df.head(config.DEBUG_SIZE)

    # Verify alignment
    if len(test_df) != len(final_probs):
        print(
            f"Warning: Metadata rows ({len(test_df)}) != Predictions ({len(final_probs)})"
        )

    # Construct Submission DataFrame
    submission_df = pd.DataFrame(
        {
            "eeg_id": test_df["eeg_id"],
            "seizure_vote": final_probs[:, 0],
            "lpd_vote": final_probs[:, 1],
            "gpd_vote": final_probs[:, 2],
            "lrda_vote": final_probs[:, 3],
            "grda_vote": final_probs[:, 4],
            "other_vote": final_probs[:, 5],
        }
    )

    # Save to disk
    os.makedirs(config.SUBMISSION_DIR, exist_ok=True)
    submission_df.to_csv(config.SUBMISSION_PATH, index=False)

    print(f"Submission successfully saved to {config.SUBMISSION_PATH}")
    print(submission_df.head())
