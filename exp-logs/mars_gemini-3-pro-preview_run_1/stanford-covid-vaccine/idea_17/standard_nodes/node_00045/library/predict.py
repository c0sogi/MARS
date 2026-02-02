import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from library.config import ModelConfig
from library.utils import get_device, set_seed
from library.dataset import load_or_process_data, RNADataset
from library.model import RNARegressor


def generate_predictions(model, loader, device):
    """
    Runs inference on the provided data loader using the specified model.

    Args:
        model (torch.nn.Module): The trained model.
        loader (DataLoader): DataLoader for the test set.
        device (torch.device): Device to run inference on.

    Returns:
        tuple: (predictions numpy array of shape (N, 107, 3), list of sample IDs)
    """
    model.eval()
    all_preds = []
    all_ids = []

    with torch.no_grad():
        for batch in loader:
            seq = batch["seq"].to(device)
            loop = batch["loop"].to(device)
            dist = batch["dist"].to(device)
            mask = batch["mask"].to(device)
            batch_ids = batch["id"]

            # Forward pass
            # Output shape: (Batch, 107, 3)
            preds = model(seq, loop, dist, mask)

            all_preds.append(preds.cpu().numpy())
            all_ids.extend(batch_ids)

    return np.concatenate(all_preds, axis=0), all_ids


def format_submission(preds, ids, output_path):
    """
    Formats raw model predictions into the competition submission CSV format.

    Args:
        preds (np.ndarray): Array of shape (N, 107, 3) containing predictions.
        ids (list): List of N sample IDs.
        output_path (str): Path to save the CSV file.
    """
    data_rows = []
    seq_len = preds.shape[1]  # Should be 107

    for i, sample_id in enumerate(ids):
        sample_preds = preds[i]  # (107, 3)

        for pos in range(seq_len):
            row_id = f"{sample_id}_{pos}"

            # Model outputs correspond to: [reactivity, deg_Mg_pH10, deg_Mg_50C]
            reactivity = float(sample_preds[pos, 0])
            deg_Mg_pH10 = float(sample_preds[pos, 1])
            deg_Mg_50C = float(sample_preds[pos, 2])

            # Unscored columns are filled with 0.0 as per instructions
            deg_pH10 = 0.0
            deg_50C = 0.0

            data_rows.append(
                [row_id, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C]
            )

    columns = [
        "id_seqpos",
        "reactivity",
        "deg_Mg_pH10",
        "deg_pH10",
        "deg_Mg_50C",
        "deg_50C",
    ]
    df = pd.DataFrame(data_rows, columns=columns)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def run_inference():
    """
    Main entry point for the inference pipeline.
    Loads data and model, generates predictions, and saves the submission file.
    """
    # 1. Setup
    set_seed()
    device = get_device()
    print(f"Inference Device: {device}")

    # 2. Load Test Data
    # load_or_process_data returns (train, val, test) dictionaries
    print("Loading test data...")
    _, _, test_data = load_or_process_data(load_cached_data=True)

    test_ds = RNADataset(test_data, mode="test")
    test_loader = DataLoader(
        test_ds,
        batch_size=ModelConfig.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # 3. Load Model Architecture and Weights
    print("Loading model...")
    model = RNARegressor(config=ModelConfig).to(device)

    model_path = os.path.join(ModelConfig.output_dir, "best_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model checkpoint not found at {model_path}. Please ensure training has completed."
        )

    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    # 4. Generate Predictions
    print("Generating predictions...")
    preds, ids = generate_predictions(model, test_loader, device)

    # 5. Save Submission
    print("Formatting submission...")
    format_submission(preds, ids, ModelConfig.submission_file)
