import os
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.model import DFL_GI_BiLSTM
from library.data import get_data_loaders
from library.utils import load_checkpoint, seed_everything


def predict_step(model, dataloader, device):
    """
    Performs inference on the provided dataloader using the given model.

    Args:
        model (torch.nn.Module): The trained model.
        dataloader (torch.utils.data.DataLoader): DataLoader for the test set.
        device (torch.device): Device to run inference on.

    Returns:
        np.ndarray: Flattened array of predictions matching the sequence of the dataloader.
    """
    model.eval()
    predictions = []

    # Disable gradient calculation for inference
    with torch.no_grad():
        for inputs, _, _ in dataloader:
            inputs = inputs.to(device)

            # Forward pass
            # Output shape: (Batch, Seq_Len)
            preds = model(inputs)

            # Flatten the batch predictions to 1D array
            # We flatten because the submission format requires one row per timestep
            # and the dataloader yields batches of sequences.
            preds_flat = preds.view(-1).cpu().numpy()
            predictions.append(preds_flat)

    # Concatenate all batch predictions into a single 1D array
    if len(predictions) > 0:
        return np.concatenate(predictions)
    else:
        return np.array([])


def generate_submission():
    """
    Main routine to generate the submission file.
    Loads data, loads the model, runs inference, and saves the CSV.
    """
    # 1. Initialization
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    print("Initializing inference pipeline...")

    # 2. Data Loading
    # We use the provided data loader function which handles caching and preprocessing.
    # We only need the test_loader.
    _, _, test_loader = get_data_loaders(load_cached_data=True)

    # 3. Model Setup
    print("Setting up model...")
    model = DFL_GI_BiLSTM().to(device)

    # Load the best checkpoint
    # We use the path defined in Config
    if os.path.exists(Config.MODEL_PATH):
        print(f"Loading checkpoint from {Config.MODEL_PATH}...")
        load_checkpoint(Config.MODEL_PATH, model, device=device)
    else:
        print(
            f"WARNING: Checkpoint not found at {Config.MODEL_PATH}. Using random weights."
        )

    # 4. Inference
    print("Running prediction...")
    preds = predict_step(model, test_loader, device)

    # 5. Post-processing and Submission Generation
    print("Generating submission file...")

    # Load test metadata to map predictions to IDs
    # The metadata contains 'id' and 'breath_id'
    test_meta = pd.read_csv(Config.TEST_META)

    # Handle DEBUG mode alignment
    # If Config.DEBUG is True, the test_loader only contains a subset of breaths.
    # We must filter the metadata to match the breaths processed.
    if Config.DEBUG:
        print(
            f"DEBUG mode detected. Filtering metadata to first {Config.DEBUG_SAMPLE_SIZE} breaths."
        )
        # Logic must match library/data.py: test_ids = test_df["breath_id"].unique()[: Config.DEBUG_SAMPLE_SIZE]
        # Note: unique() preserves appearance order in pandas.
        unique_breaths = test_meta["breath_id"].unique()
        selected_breaths = unique_breaths[: Config.DEBUG_SAMPLE_SIZE]
        test_meta = test_meta[test_meta["breath_id"].isin(selected_breaths)].copy()

    # Sort metadata to ensure alignment with Model Output
    # The model processes breaths sequentially (as grouped in the dataset).
    # The VentilatorDataset in library/data.py is built from a dataframe sorted by ['breath_id', 'id'].
    # Therefore, our flattened predictions correspond to rows sorted by breath_id, then id.
    test_meta_sorted = test_meta.sort_values(["breath_id", "id"]).reset_index(drop=True)

    # Integrity Check
    if len(preds) != len(test_meta_sorted):
        raise ValueError(
            f"Shape mismatch: Predictions have {len(preds)} elements, "
            f"but metadata has {len(test_meta_sorted)} rows. "
            "Check Debug settings or Data Loading logic."
        )

    # Assign predictions
    test_meta_sorted["pressure"] = preds

    # Format for submission: id, pressure
    # Sort by 'id' as per standard submission format
    submission = test_meta_sorted[["id", "pressure"]].sort_values("id")

    # 6. Save Output
    # Save to Config.SUBMISSION_PATH
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")

    # Also save to ./submission/submission.csv as requested in the prompt requirements
    specific_submission_dir = "./submission"
    os.makedirs(specific_submission_dir, exist_ok=True)
    specific_submission_path = os.path.join(specific_submission_dir, "submission.csv")
    submission.to_csv(specific_submission_path, index=False)
    print(f"Submission also saved to {specific_submission_path}")
