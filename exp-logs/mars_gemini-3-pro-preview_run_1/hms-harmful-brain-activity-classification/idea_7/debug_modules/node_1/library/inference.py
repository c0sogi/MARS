import os
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.data import get_dataloaders
from library.model import OffsetGuidedDualStreamModel
from library.utils import seed_everything, get_logger


def predict(model, loader, device):
    """
    Performs inference on the provided data loader using the given model.

    Args:
        model (torch.nn.Module): The trained model.
        loader (torch.utils.data.DataLoader): DataLoader for the test set.
        device (torch.device): Device to run inference on.

    Returns:
        np.ndarray: Array of predicted probabilities with shape (N_samples, N_classes).
    """
    model.eval()
    all_probs = []

    with torch.no_grad():
        for spec, eeg, guidance in loader:
            # Move inputs to device
            spec = spec.to(device)
            eeg = eeg.to(device)
            guidance = guidance.to(device)

            # Forward pass
            logits = model(spec, eeg, guidance)

            # Apply Softmax to get probabilities
            probs = torch.softmax(logits, dim=1)

            # Move to CPU and collect
            all_probs.append(probs.cpu().numpy())

    # Concatenate all batches
    return np.concatenate(all_probs, axis=0)


def run_inference(checkpoint_path=None, output_path=None, debug=False, batch_size=None):
    """
    Orchestrates the full inference pipeline: setup, data loading, model loading,
    prediction, and submission file generation.

    Args:
        checkpoint_path (str, optional): Path to the model checkpoint. Defaults to Config.CHECKPOINT_DIR/best_model.pth.
        output_path (str, optional): Path to save the submission CSV. Defaults to Config.SUBMISSION_DIR/submission.csv.
        debug (bool): Whether to run in debug mode (subset of data).
        batch_size (int, optional): Batch size for inference. Defaults to Config.BATCH_SIZE.
    """
    # 1. Setup
    Config.setup(debug=debug)
    seed_everything(Config.SEED)
    logger = get_logger("inference")
    device = torch.device(Config.DEVICE)

    if batch_size is not None:
        Config.BATCH_SIZE = batch_size

    # Set default paths if not provided
    if checkpoint_path is None:
        checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    if output_path is None:
        output_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    logger.info(f"Starting inference. Device: {device}")
    logger.info(f"Checkpoint: {checkpoint_path}")

    # 2. Data Loading
    # We only need the test loader here. get_dataloaders returns (train, val, test).
    # We enable caching to ensure efficient data loading.
    _, _, test_loader = get_dataloaders(Config, load_cached_data=True)
    logger.info(f"Test loader initialized with {len(test_loader.dataset)} samples.")

    # 3. Model Initialization
    model = OffsetGuidedDualStreamModel(Config).to(device)

    # 4. Load Weights
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        logger.info("Model weights loaded successfully.")
    else:
        logger.error(
            f"Checkpoint not found at {checkpoint_path}. Using random initialization (Expect poor results)."
        )

    # 5. Prediction
    logger.info("Running prediction loop...")
    predictions = predict(model, test_loader, device)

    # 6. Submission Generation
    # Load test metadata to retrieve eeg_ids.
    # The DataLoader preserves the order of the DataFrame it was created from.
    test_df = pd.read_csv(Config.TEST_CSV)

    # If in debug mode, the loader uses a subset, so we must subset the dataframe similarly
    # to match the IDs. The logic must match get_dataloaders exactly.
    if Config.DEBUG:
        test_df = test_df.sample(
            n=min(len(test_df), 100), random_state=Config.SEED
        ).reset_index(drop=True)

    # Verify length alignment
    if len(predictions) != len(test_df):
        logger.error(
            f"Mismatch: Predictions {len(predictions)} vs Metadata {len(test_df)}"
        )
        # In case of mismatch (e.g. drop_last=True in loader, though test usually has drop_last=False),
        # we truncate to the smaller length to avoid crash, but warn heavily.
        min_len = min(len(predictions), len(test_df))
        predictions = predictions[:min_len]
        test_df = test_df.iloc[:min_len]

    # Construct DataFrame
    submission_df = pd.DataFrame(predictions, columns=Config.CLASS_NAMES)
    submission_df.insert(0, "eeg_id", test_df["eeg_id"])

    # Ensure eeg_id is integer
    submission_df["eeg_id"] = submission_df["eeg_id"].astype(int)

    # 7. Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission_df.to_csv(output_path, index=False)
    logger.info(f"Submission saved to {output_path}")

    # Print first few rows for verification
    logger.info("First 5 rows of submission:")
    logger.info(f"\n{submission_df.head().to_string()}")

    return submission_df
