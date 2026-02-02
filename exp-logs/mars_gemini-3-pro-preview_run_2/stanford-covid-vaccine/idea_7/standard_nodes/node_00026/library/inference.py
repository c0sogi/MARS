import os
import numpy as np
import pandas as pd
import torch
from library.config import Config
from library.data import get_dataloaders
from library.model import HybridNet
from library.utils import set_seed


def predict(model, loader, device):
    """
    Generates predictions for the entire dataset provided by the loader.

    Args:
        model (torch.nn.Module): The trained model.
        loader (torch.utils.data.DataLoader): DataLoader for the test set.
        device (torch.device): Device to run inference on.

    Returns:
        tuple:
            - preds (np.ndarray): Array of shape (NumSamples, SeqLen, NumTargets).
            - ids (list): List of sample IDs corresponding to the predictions.
    """
    model.eval()
    all_preds = []
    all_ids = []

    with torch.no_grad():
        for batch in loader:
            inputs = batch["inputs"].to(device)

            # Forward pass
            # Output shape: (Batch, SeqLen, NumTargets)
            outputs = model(inputs)

            # Move to CPU and numpy
            batch_preds = outputs.cpu().numpy()

            all_preds.append(batch_preds)

            # Collect IDs (if they are in the batch)
            if "id" in batch:
                all_ids.extend(batch["id"])

    # Concatenate all batches
    # Final shape: (TotalSamples, SeqLen, NumTargets)
    preds = np.concatenate(all_preds, axis=0)

    return preds, all_ids


def create_submission_df(ids, preds):
    """
    Formats the predictions into a pandas DataFrame matching the submission format.

    Args:
        ids (list): List of sample IDs.
        preds (np.ndarray): Predictions array of shape (N, SeqLen, 5).

    Returns:
        pd.DataFrame: Formatted submission DataFrame.
    """
    # Prediction array dimensions
    num_samples, seq_len, num_targets = preds.shape

    # Flatten predictions to (N * SeqLen, NumTargets)
    # We flatten by iterating rows (samples) then columns (sequence positions)
    # This corresponds to reshaping with default order='C'
    flat_preds = preds.reshape(-1, num_targets)

    # Generate id_seqpos column
    # We need to repeat each ID for seq_len times, and append the position index
    id_seqpos_list = []
    for sample_id in ids:
        for pos in range(seq_len):
            id_seqpos_list.append(f"{sample_id}_{pos}")

    # Create DataFrame
    # Columns: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
    cols = ["id_seqpos"] + Config.TARGET_COLS

    # Construct data dictionary
    data = {"id_seqpos": id_seqpos_list}

    # Add target columns
    for i, col_name in enumerate(Config.TARGET_COLS):
        data[col_name] = flat_preds[:, i]

    df = pd.DataFrame(data)
    return df


def run_inference(
    model_path=Config.MODEL_CHECKPOINT,
    output_path=Config.SUBMISSION_PATH,
    load_cached_data=True,
):
    """
    Main entry point for inference. Loads data, model, generates predictions,
    and saves the submission file.

    Args:
        model_path (str): Path to the saved model weights.
        output_path (str): Path to save the submission CSV.
        load_cached_data (bool): Whether to use cached preprocessed data.
    """
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on device: {device}")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 2. Load Data
    # We only need the test loader here
    print("Loading test data...")
    _, _, test_loader = get_dataloaders(load_cached_data=load_cached_data)

    # 3. Load Model
    print(f"Loading model from {model_path}...")
    model = HybridNet().to(device)

    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

    # 4. Predict
    print("Generating predictions...")
    preds, ids = predict(model, test_loader, device)

    print(f"Predictions shape: {preds.shape}")
    print(f"Number of IDs: {len(ids)}")

    # 5. Format and Save
    print("Formatting submission...")
    submission_df = create_submission_df(ids, preds)

    print(f"Saving submission to {output_path}...")
    submission_df.to_csv(output_path, index=False)

    print("Inference complete.")
