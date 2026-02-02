import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from library import config, model, data, utils


def inference_fn(model_instance, loader, device):
    """
    Runs inference on the provided loader using the given model.

    Args:
        model_instance (nn.Module): The loaded model.
        loader (DataLoader): DataLoader for test data.
        device (str): Device to run on.

    Returns:
        np.ndarray: Predictions of shape (N_samples, Seq_Len, N_Targets)
    """
    model_instance.eval()
    all_preds = []

    with torch.no_grad():
        for inputs in loader:
            inputs = inputs.to(device)
            # Forward pass
            # Output shape: (Batch, 107, 5)
            outputs = model_instance(inputs)
            all_preds.append(outputs.cpu().numpy())

    # Concatenate along batch dimension
    if len(all_preds) > 0:
        return np.concatenate(all_preds, axis=0)
    else:
        return np.array([])


def generate_submission():
    """
    Main function to load data, run inference, and generate the submission CSV.
    """
    # 1. Setup
    utils.seed_everything(config.SEED)
    device = config.DEVICE

    print(f"Running inference on device: {device}")

    # 2. Load Data
    # We use the existing data utility to ensure consistency in preprocessing.
    # load_cached_data=True ensures we use the cache generated during training/setup if available.
    # If cache is missing, it will process from the metadata CSVs.
    test_ids, test_inputs, _ = data.load_or_process_data("test", load_cached_data=True)

    # Create Dataset and Loader
    test_dataset = data.RNADataset(test_inputs, targets=None, ids=test_ids)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=(device == "cuda"),
    )

    # 3. Load Model
    net = model.PartnerAwareHybridNet()
    net.to(device)

    if not os.path.exists(config.MODEL_SAVE_PATH):
        raise FileNotFoundError(
            f"Model weights not found at {config.MODEL_SAVE_PATH}. Train the model first."
        )

    print(f"Loading model weights from {config.MODEL_SAVE_PATH}")
    net.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))

    # 4. Run Inference
    print("Generating predictions...")
    preds = inference_fn(net, test_loader, device)
    # preds shape: (N_samples, 107, 5)

    if preds.size == 0:
        print("No predictions generated. Check data loader.")
        return

    # 5. Format Submission
    print("Formatting submission...")
    n_samples, seq_len, n_targets = preds.shape

    # Flatten predictions to (N_samples * Seq_Len, N_Targets)
    # We flatten the first two dimensions (samples and sequence length)
    preds_flat = preds.reshape(-1, n_targets)

    # Generate id_seqpos column
    # Repeat IDs: [id1, id1, ..., id2, id2, ...]
    ids_repeated = np.repeat(test_ids, seq_len)

    # Tile positions: [0, 1, ..., 106, 0, 1, ..., 106]
    positions_tiled = np.tile(np.arange(seq_len), n_samples)

    # Combine: id_pos
    # Using list comprehension is efficient enough for ~25k rows
    id_seqpos = [f"{id_}_{pos}" for id_, pos in zip(ids_repeated, positions_tiled)]

    # Create DataFrame
    # Columns must match config.TARGET_COLS
    submission_df = pd.DataFrame(preds_flat, columns=config.TARGET_COLS)

    # Insert identifier column at the beginning
    submission_df.insert(0, "id_seqpos", id_seqpos)

    # 6. Save
    os.makedirs(os.path.dirname(config.SUBMISSION_PATH), exist_ok=True)
    submission_df.to_csv(config.SUBMISSION_PATH, index=False)

    print(f"Submission saved to {config.SUBMISSION_PATH}")
    print(f"Rows: {len(submission_df)}, Columns: {submission_df.columns.tolist()}")
