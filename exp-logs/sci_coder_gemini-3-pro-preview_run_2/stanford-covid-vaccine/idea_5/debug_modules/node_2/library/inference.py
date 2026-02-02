import os
import random
import numpy as np
import pandas as pd
import torch
from library.config import Config
from library.model import HybridNet, get_dataloaders


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def predict(load_cached_data=True, batch_size=None):
    """
    Loads the trained model and performs inference on the test set.

    Args:
        load_cached_data (bool): Whether to use cached pre-processed data.
        batch_size (int, optional): Batch size for inference. If None, uses Config.BATCH_SIZE.

    Returns:
        tuple: (all_preds, all_ids)
            - all_preds (np.ndarray): Predictions of shape (Num_Samples, Seq_Len, Num_Targets)
            - all_ids (list): List of sample IDs corresponding to the predictions.
    """
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 1. Load Data
    # We retrieve the test_loader. The get_dataloaders function handles caching internally.
    # Note: get_dataloaders returns (train, val, test). We only need test.
    _, _, test_loader = get_dataloaders(load_cached_data=load_cached_data)

    # If a specific batch size is requested (different from Config), we might need to
    # reconstruct the loader, but since we are using the provided library function
    # which hardcodes Config.BATCH_SIZE, we proceed with the returned loader.
    # The argument is kept for interface flexibility.

    # 2. Load Model
    if not os.path.exists(Config.MODEL_SAVE_PATH):
        raise FileNotFoundError(
            f"Model file not found at {Config.MODEL_SAVE_PATH}. Please run training first."
        )

    model = HybridNet().to(device)
    # Load weights
    state_dict = torch.load(Config.MODEL_SAVE_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    all_preds = []
    all_ids = []

    # 3. Inference Loop
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch["inputs"].to(device)
            ids = batch["ids"]

            # Forward pass
            # Output shape: (Batch, Seq_Len, 5)
            outputs = model(inputs)

            # Move to CPU and numpy
            preds = outputs.cpu().numpy()

            all_preds.append(preds)
            all_ids.extend(ids)

    # Concatenate all batches
    if len(all_preds) > 0:
        all_preds = np.concatenate(all_preds, axis=0)
    else:
        all_preds = np.array([])

    return all_preds, all_ids


def format_submission(preds, ids, save_path=None):
    """
    Formats the raw predictions into the submission CSV format.

    Args:
        preds (np.ndarray): Predictions array of shape (N, Seq_Len, 5).
        ids (list): List of sample IDs.
        save_path (str, optional): Path to save the CSV. Defaults to Config.SUBMISSION_PATH.
    """
    if save_path is None:
        save_path = Config.SUBMISSION_PATH

    # Ensure output directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # The target columns in the order output by the model
    # Config.ALL_TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
    target_cols = Config.ALL_TARGET_COLS

    submission_ids = []
    flat_preds = []

    num_samples, seq_len, num_targets = preds.shape

    # Flatten the predictions: (N * Seq_Len, 5)
    # And generate the id_seqpos keys
    for i in range(num_samples):
        sample_id = ids[i]
        for j in range(seq_len):
            submission_ids.append(f"{sample_id}_{j}")
            flat_preds.append(preds[i, j, :])

    flat_preds = np.array(flat_preds)

    # Create DataFrame
    submission_df = pd.DataFrame(flat_preds, columns=target_cols)
    submission_df.insert(0, "id_seqpos", submission_ids)

    # Save
    submission_df.to_csv(save_path, index=False)
    print(f"Submission saved to {save_path}")


def run_inference(load_cached_data=True):
    """
    Main entry point to run inference and generate submission.
    """
    print("Starting Inference...")

    # Run prediction
    preds, ids = predict(load_cached_data=load_cached_data)

    print(f"Prediction complete. Shape: {preds.shape}")

    # Format and save
    format_submission(preds, ids)
