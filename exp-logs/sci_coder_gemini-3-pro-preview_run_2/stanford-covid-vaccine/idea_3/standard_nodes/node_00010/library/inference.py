import os
import torch
import numpy as np
import pandas as pd
from library.config import Config
from library.data import get_test_dataloader
from library.model import RNA_Net


def predict(
    model_path,
    device=None,
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    load_cached_data=True,
):
    """
    Loads the model and generates predictions for the test set.

    Args:
        model_path (str): Path to the saved model weights (.pth).
        device (torch.device): Device to run inference on.
        batch_size (int): Batch size for the dataloader.
        num_workers (int): Number of workers for data loading.
        load_cached_data (bool): Whether to use cached preprocessed data.

    Returns:
        preds (np.ndarray): Array of shape (N_samples, Seq_Len, Num_Targets).
        test_ids (np.ndarray): Array of shape (N_samples,) containing sample IDs.
    """
    if device is None:
        device = Config.DEVICE

    print(f"Loading model from {model_path}...")
    model = RNA_Net()

    # Load weights
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print("Preparing test dataloader...")
    test_loader, test_ids = get_test_dataloader(
        load_cached_data=load_cached_data,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    all_preds = []

    print("Running inference...")
    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs.to(device)

            # Forward pass
            # Output shape: (Batch, Seq_Len, Num_Targets)
            outputs = model(inputs)

            # Move to CPU and numpy
            all_preds.append(outputs.cpu().numpy())

    # Concatenate all batches
    # Final shape: (Total_Samples, 107, 5)
    preds = np.concatenate(all_preds, axis=0)

    return preds, test_ids


def create_submission(preds, test_ids, output_path):
    """
    Formats predictions into the submission CSV format.

    Args:
        preds (np.ndarray): Predictions of shape (N, 107, 5).
        test_ids (np.ndarray): Sample IDs of shape (N,).
        output_path (str): Path to save the submission CSV.
    """
    print("Formatting submission...")

    # preds shape: (N_samples, Seq_Len, Num_Targets)
    n_samples, seq_len, n_targets = preds.shape

    # We need to flatten the data to (N_samples * Seq_Len, Num_Targets)
    # The submission format requires one row per sequence position.

    # 1. Generate id_seqpos column
    # We repeat each ID 107 times and append _0, _1, ... _106
    ids_repeated = np.repeat(test_ids, seq_len)

    # Create sequence indices tile: [0, 1, ..., 106, 0, 1, ..., 106, ...]
    seq_indices = np.tile(np.arange(seq_len), n_samples)

    # Combine to form strings like "id_00b436dec_0"
    # Using list comprehension for string formatting is usually fast enough
    id_seqpos_list = [f"{id_}_{pos}" for id_, pos in zip(ids_repeated, seq_indices)]

    # 2. Flatten predictions
    # Reshape to (N_samples * Seq_Len, Num_Targets)
    preds_flat = preds.reshape(-1, n_targets)

    # 3. Create DataFrame
    # Columns: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
    df_data = {"id_seqpos": id_seqpos_list}

    # Add target columns
    for i, col_name in enumerate(Config.TARGET_COLS):
        df_data[col_name] = preds_flat[:, i]

    submission_df = pd.DataFrame(df_data)

    # 4. Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
    print(f"Submission shape: {submission_df.shape}")


def run_inference(
    model_path=Config.MODEL_PATH,
    output_path=Config.SUBMISSION_FILE,
    batch_size=Config.BATCH_SIZE,
    load_cached_data=True,
):
    """
    Main function to run the inference pipeline.
    """
    # Run prediction
    preds, test_ids = predict(
        model_path=model_path, batch_size=batch_size, load_cached_data=load_cached_data
    )

    # Create submission file
    create_submission(preds, test_ids, output_path)
