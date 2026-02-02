import os
import torch
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import seed_everything, load_checkpoint
from library.dataset import get_dataloaders
from library.model import HybridCNNBiGRU


def generate_submission_file(ids, preds, output_path):
    """
    Formats predictions into the competition CSV format.

    Args:
        ids (list): List of sample IDs.
        preds (np.array): Prediction tensor of shape (N_samples, 107, 5).
        output_path (str): Path to save the CSV file.
    """
    target_cols = Config.TARGET_COLS

    # preds shape: (N, 107, 5)
    seq_len = preds.shape[1]

    # Flatten predictions: (N * 107, 5)
    flat_preds = preds.reshape(-1, 5)

    # Create id_seqpos strings
    id_seqpos_list = []
    for sample_id in ids:
        for pos in range(seq_len):
            id_seqpos_list.append(f"{sample_id}_{pos}")

    # Create DataFrame
    submission_df = pd.DataFrame(flat_preds, columns=target_cols)
    submission_df.insert(0, "id_seqpos", id_seqpos_list)

    # Save
    print(f"Saving submission to {output_path}...")
    submission_df.to_csv(output_path, index=False)
    print("Submission saved successfully.")


def predict_and_submit(load_cached_data=True):
    """
    Loads the best model, runs inference on the test set, and generates the submission file.

    Args:
        load_cached_data (bool): Whether to load pre-processed data from cache.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 2. Data Loading
    # We only utilize the test_loader for inference
    print("Loading test data...")
    _, _, test_loader = get_dataloaders(load_cached_data=load_cached_data)

    # 3. Model Initialization
    print("Initializing model...")
    model = HybridCNNBiGRU().to(device)

    # 4. Load Weights
    if not os.path.exists(Config.MODEL_SAVE_PATH):
        raise FileNotFoundError(f"Model file not found at {Config.MODEL_SAVE_PATH}")

    start_epoch, best_score = load_checkpoint(model, filename=Config.MODEL_SAVE_PATH)
    print(f"Loaded model checkpoint (Epoch: {start_epoch}, Score: {best_score})")

    # 5. Inference
    model.eval()
    all_ids = []
    all_preds = []

    print("Running inference on test set...")
    with torch.no_grad():
        for batch in test_loader:
            ids = batch["id"]
            sequence = batch["sequence"].to(device)
            structure = batch["structure"].to(device)
            loop_type = batch["predicted_loop_type"].to(device)

            # Forward pass
            # Output shape: (Batch, 107, 5)
            outputs = model(sequence, structure, loop_type)

            all_preds.append(outputs.cpu().numpy())
            all_ids.extend(ids)

    # Concatenate all batches
    # Shape: (Total_Samples, 107, 5)
    final_preds = np.concatenate(all_preds, axis=0)

    # 6. Generate Submission
    generate_submission_file(all_ids, final_preds, Config.SUBMISSION_PATH)
