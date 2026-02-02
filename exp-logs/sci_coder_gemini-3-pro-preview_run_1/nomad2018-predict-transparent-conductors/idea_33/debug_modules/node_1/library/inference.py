import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from library.config import Config
from library.dataset import MaterialsDataset, collate_fn
from library.model import GDCC_WDS
from library.data_processing import PreprocessPipeline


def run_inference(
    model_path=Config.BEST_MODEL_PATH,
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    device=Config.DEVICE,
    max_samples=None,
    load_cached_data=True,
):
    """
    Loads the trained model and runs inference on the test dataset.

    Args:
        model_path (str): Path to the saved model checkpoint.
        batch_size (int): Batch size for inference.
        num_workers (int): Number of worker threads for DataLoader.
        device (str): Device to run inference on ('cpu' or 'cuda').
        max_samples (int, optional): Limit number of test samples for debugging.
        load_cached_data (bool): Whether to load pre-processed data from cache.

    Returns:
        ids (np.ndarray): Array of sample IDs.
        predictions (np.ndarray): Array of predicted values (original scale).
    """
    # 1. Initialize Dataset and DataLoader
    print("Initializing Test Dataset...")
    # Mode="test" ensures that the dataset loads the saved scalers from training
    test_dataset = MaterialsDataset(
        metadata_path=Config.TEST_METADATA_PATH,
        mode="test",
        max_samples=max_samples,
        load_cached_data=load_cached_data,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # 2. Load Model
    print(f"Loading model from {model_path}...")
    model = GDCC_WDS()

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 3. Run Inference
    print("Running inference...")
    all_ids = []
    all_preds = []

    with torch.no_grad():
        for batch in test_loader:
            # Move data to device
            atomic_features = batch["atomic_features"].to(device)
            global_features = batch["global_features"].to(device)
            batch_indices = batch["batch_indices"].to(device)

            # Get batch size (num_graphs)
            num_graphs = len(batch["ids"])

            # Forward pass
            outputs = model(atomic_features, global_features, batch_indices, num_graphs)

            # Collect results
            all_ids.append(batch["ids"].cpu().numpy())
            all_preds.append(outputs.cpu().numpy())

    # Concatenate results
    ids = np.concatenate(all_ids)
    predictions_log_scale = np.concatenate(all_preds)

    # 4. Inverse Transform Targets
    # The model predicts log(1+y), so we apply exp(y) - 1 to get original scale.
    # We use the PreprocessPipeline method for consistency.
    pipeline = PreprocessPipeline()
    predictions = pipeline.inverse_transform_targets(predictions_log_scale)

    return ids, predictions


def write_submission(ids, predictions, output_path):
    """
    Writes the predictions to a CSV file in the required format.

    Args:
        ids (np.ndarray): Array of sample IDs.
        predictions (np.ndarray): Array of predicted values (N, 2).
        output_path (str): Path to save the submission CSV.
    """
    # Cite debug_lesson_16: Handle Empty Results from os.path.dirname
    output_path = os.path.abspath(output_path)
    print(f"Writing submission to {output_path}...")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Create DataFrame
    # Columns must match the sample submission: id,formation_energy_ev_natom,bandgap_energy_ev
    df = pd.DataFrame(
        {
            "id": ids,
            "formation_energy_ev_natom": predictions[:, 0],
            "bandgap_energy_ev": predictions[:, 1],
        }
    )

    # Save to CSV
    df.to_csv(output_path, index=False)
    print("Submission saved successfully.")


def generate_submission(
    model_path=Config.BEST_MODEL_PATH,
    output_path=os.path.join(Config.SUBMISSION_DIR, "submission.csv"),
    max_samples=None,
    load_cached_data=True,
):
    """
    Orchestrates the inference and submission generation process.

    Args:
        model_path (str): Path to the trained model checkpoint.
        output_path (str): Path where the submission CSV will be saved.
        max_samples (int, optional): Limit inference to a subset of samples.
        load_cached_data (bool): Whether to use cached pre-processed data.
    """
    # Run inference
    ids, preds = run_inference(
        model_path=model_path,
        max_samples=max_samples,
        load_cached_data=load_cached_data,
    )

    # Write submission
    write_submission(ids, preds, output_path)
