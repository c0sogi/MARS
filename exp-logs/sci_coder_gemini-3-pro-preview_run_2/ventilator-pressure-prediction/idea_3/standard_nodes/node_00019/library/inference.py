import os
import numpy as np
import pandas as pd
import torch
from library.utils import seed_everything, get_device
from library.dataset import get_data_loaders
from library.model import HybridModel


def generate_predictions(
    model_path: str = "./working/idea_3_execution/best_model.pth",
    batch_size: int = 256,
    debug: bool = False,
    submission_output_dir: str = "./submission",
    load_cached_data: bool = True,
):
    """
    Generates predictions for the test set using a trained model and saves the submission file.

    Args:
        model_path (str): Path to the trained model weights (.pth file).
        batch_size (int): Batch size for the test data loader.
        debug (bool): If True, uses a subset of data for debugging.
        submission_output_dir (str): Directory where the submission.csv will be saved.
        load_cached_data (bool): Whether to load pre-processed data from cache.
    """
    # 1. Setup
    seed_everything(42)
    device = get_device()
    print(f"Starting inference on device: {device}")

    # 2. Load Data
    # We only need the test_loader, but get_data_loaders returns all three
    print("Loading test data...")
    _, _, test_loader = get_data_loaders(
        batch_size=batch_size, load_cached_data=load_cached_data, debug=debug
    )

    # 3. Initialize Model
    # Must match the architecture used in training (train.py / model.py)
    print("Initializing model architecture...")
    model = HybridModel(input_dim=14, lstm_dim=512, num_lstm_layers=4).to(device)

    # 4. Load Weights
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found at {model_path}")

    print(f"Loading model weights from {model_path}...")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # 5. Inference Loop
    print("Running inference...")
    all_preds = []

    with torch.no_grad():
        for batch in test_loader:
            # batch['input'] shape: (Batch, Seq_Len, Features)
            X = batch["input"].to(device)

            # Forward pass
            # Output shape: (Batch, Seq_Len)
            preds = model(X)

            # Flatten to 1D array immediately to match the row-wise submission format
            preds_flat = preds.cpu().numpy().flatten()
            all_preds.append(preds_flat)

    # Concatenate all batches
    final_preds = np.concatenate(all_preds)
    print(f"Total predictions generated: {len(final_preds)}")

    # 6. Map to IDs and Save Submission
    print("Mapping predictions to IDs...")
    test_meta_path = "./metadata/test_metadata.csv"

    if not os.path.exists(test_meta_path):
        raise FileNotFoundError(
            f"Test metadata not found at {test_meta_path}. Cannot align predictions."
        )

    df_meta = pd.read_csv(test_meta_path)

    # Ensure metadata is sorted exactly as the dataset loader yields data.
    # The dataset loader (via dataset.py) sorts by [breath_id, time_step].
    # We sort metadata by [breath_id, id] assuming id correlates with time_step.
    df_meta = df_meta.sort_values(["breath_id", "id"])

    # Handle Debug/Mismatch cases
    if len(final_preds) != len(df_meta):
        print(
            f"Warning: Prediction count ({len(final_preds)}) != Metadata count ({len(df_meta)})."
        )
        # If debugging, we truncate the metadata to match the predictions
        min_len = min(len(final_preds), len(df_meta))
        df_meta = df_meta.iloc[:min_len]
        final_preds = final_preds[:min_len]

    # Create submission DataFrame
    submission = pd.DataFrame({"id": df_meta["id"], "pressure": final_preds})

    # Save to disk
    os.makedirs(submission_output_dir, exist_ok=True)
    submission_path = os.path.join(submission_output_dir, "submission.csv")

    submission.to_csv(submission_path, index=False)
    print(f"Submission saved successfully to {submission_path}")
