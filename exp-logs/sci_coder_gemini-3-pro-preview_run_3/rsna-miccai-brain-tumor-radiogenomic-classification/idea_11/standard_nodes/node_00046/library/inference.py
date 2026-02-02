import os
import torch
import pandas as pd
import numpy as np
from library.utils import seed_everything, get_device
from library.model import ModalityGroupedEfficientNet
from library.data_loader import get_dataloaders


def predict_submission(
    model_path="./working/idea_11/best_model.pth",
    output_path="./submission.csv",
    batch_size=8,
    load_cached_data=True,
    device=None,
    debug_limit=None,
):
    """
    Generates predictions for the test set and saves a submission file.

    Args:
        model_path (str): Path to the saved model weights.
        output_path (str): Path to save the submission CSV.
        batch_size (int): Batch size for inference.
        load_cached_data (bool): Whether to use cached preprocessed data.
        device (torch.device, optional): Device to run inference on.
        debug_limit (int, optional): Limit number of test samples for debugging.
    """
    # 1. Setup
    seed_everything(42)
    if device is None:
        device = get_device()

    print(f"Starting inference on device: {device}")

    # 2. Data Loading
    # We only need the test loader. get_dataloaders returns (train, val, test)
    print("Loading test data...")
    _, _, test_loader = get_dataloaders(
        batch_size=batch_size,
        load_cached_data=load_cached_data,
        debug_limit=debug_limit,
    )

    # 3. Model Initialization
    print("Initializing model...")
    model = ModalityGroupedEfficientNet()
    model.to(device)

    if os.path.exists(model_path):
        print(f"Loading weights from {model_path}...")
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        raise FileNotFoundError(
            f"Model weights not found at {model_path}. Please train the model first."
        )

    # 4. Inference Loop
    model.eval()
    all_ids = []
    all_probs = []

    print("Running inference...")
    with torch.no_grad():
        for inputs, patient_ids in test_loader:
            inputs = inputs.to(device)

            # Forward pass
            logits = model(inputs)

            # Convert logits to probabilities
            probs = torch.sigmoid(logits)

            # Store results
            # Flatten probs to 1D array
            probs_np = probs.cpu().numpy().flatten()

            all_ids.extend(patient_ids)
            all_probs.extend(probs_np)

    # 5. Result Formatting
    # Convert IDs to integers to match sample_submission format (e.g., "00013" -> 13)
    # The data loader returns IDs as strings based on the metadata.
    processed_ids = [int(pid) for pid in all_ids]

    submission_df = pd.DataFrame({"BraTS21ID": processed_ids, "MGMT_value": all_probs})

    # Sort by ID just to be clean, though not strictly required
    submission_df = submission_df.sort_values("BraTS21ID")

    # 6. Save Submission
    # Ensure output directory exists if it's not in the current dir
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print(f"Saving submission to {output_path}...")
    submission_df.to_csv(output_path, index=False)

    print("Inference complete.")
    print(submission_df.head())

    return submission_df
