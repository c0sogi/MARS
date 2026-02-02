import os
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.model import ShallowEmbeddingMLP
from library.dataset import get_dataloaders
from library.utils import load_checkpoint


def run_inference(
    checkpoint_path=os.path.join(Config.WORKING_DIR, "best_model.pth"),
    output_path=os.path.join(Config.SUBMISSION_DIR, "submission.csv"),
    device=Config.DEVICE,
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    debug=Config.DEBUG,
):
    """
    Loads a trained model, generates predictions on the test set, and saves the submission file.

    Args:
        checkpoint_path (str): Path to the saved model checkpoint.
        output_path (str): Path where the submission CSV will be saved.
        device (str): Device to run inference on ('cpu' or 'cuda').
        batch_size (int): Batch size for the data loader.
        num_workers (int): Number of worker threads for data loading.
        debug (bool): If True, runs on a subset of data.
    """
    print(f"Initializing inference on device: {device}")

    # 1. Initialize Model
    model = ShallowEmbeddingMLP()
    model.to(device)

    # 2. Load Weights
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. Ensure training has completed."
        )

    print(f"Loading checkpoint from {checkpoint_path}...")
    load_checkpoint(checkpoint_path, model, optimizer=None, device=device)

    # 3. Get DataLoaders
    # We only need the test loader here.
    # get_dataloaders handles caching and preprocessing internally.
    print("Loading test data...")
    _, _, test_loader = get_dataloaders(
        batch_size=batch_size,
        num_workers=num_workers,
        load_cached_data=True,
        debug=debug,
    )

    # 4. Inference Loop
    print("Starting prediction loop...")
    model.eval()
    all_preds = []

    with torch.no_grad():
        for batch in test_loader:
            cont = batch["continuous"].to(device)
            cat = batch["categorical"].to(device)

            # Forward pass
            outputs = model(cont, cat)

            # Collect results (move to CPU and convert to numpy)
            preds = outputs.cpu().numpy()
            all_preds.append(preds)

    # Concatenate all batches
    predictions = np.concatenate(all_preds)

    # Flatten if shape is (N, 1) to (N,)
    if predictions.ndim > 1:
        predictions = predictions.flatten()

    print(f"Generated {len(predictions)} predictions.")

    # 5. Generate Submission File
    # Load test metadata to ensure IDs match exactly
    test_meta_path = os.path.join(Config.METADATA_DIR, "test_metadata.csv")
    if not os.path.exists(test_meta_path):
        raise FileNotFoundError(f"Test metadata not found at {test_meta_path}")

    # If debugging, we need to subset the metadata similarly to how get_dataloaders does it
    # get_dataloaders calls _load_and_preprocess which subsets metadata if debug=True
    df_test_meta = pd.read_csv(test_meta_path)

    if debug:
        df_test_meta = df_test_meta.head(Config.DEBUG_SUBSET_SIZE)

    test_ids = df_test_meta["id"].values

    # Validation check
    if len(test_ids) != len(predictions):
        raise ValueError(
            f"Mismatch between ID count ({len(test_ids)}) and Prediction count ({len(predictions)}). "
            "Check if debug flags match between data loading and inference."
        )

    # Create DataFrame
    submission_df = pd.DataFrame({"id": test_ids, "target": predictions})

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to CSV
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved successfully to {output_path}")

    return submission_df
