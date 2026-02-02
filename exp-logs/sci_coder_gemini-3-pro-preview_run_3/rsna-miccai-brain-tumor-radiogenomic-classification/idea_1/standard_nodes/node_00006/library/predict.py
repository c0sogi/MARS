import os
import torch
import pandas as pd
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import set_seed
from library.dataset import MGMTDataset
from library.model import MGMTNet


def generate_submission(
    model_path=Config.MODEL_PATH,
    output_path=Config.SUBMISSION_PATH,
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    device_name=Config.DEVICE,
):
    """
    Generates predictions for the test set using the trained model and saves to CSV.

    Args:
        model_path (str): Path to the trained model checkpoint.
        output_path (str): Path to save the submission CSV.
        batch_size (int): Batch size for inference.
        num_workers (int): Number of subprocesses for data loading.
        device_name (str): Device to run inference on.
    """
    # Ensure reproducibility
    set_seed(Config.SEED)
    device = torch.device(device_name)

    # Load Test Metadata
    if not os.path.exists(Config.TEST_METADATA):
        raise FileNotFoundError(
            f"Test metadata file not found at {Config.TEST_METADATA}"
        )

    test_df = pd.read_parquet(Config.TEST_METADATA)

    # Initialize Test Dataset
    # load_cached_data=True allows using pre-processed tensors from disk if available
    test_dataset = MGMTDataset(test_df, split_name="test", load_cached_data=True)

    # Initialize DataLoader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Initialize Model
    model = MGMTNet().to(device)

    # Load Model Weights
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(f"Warning: Model not found at {model_path}. Using random initialization.")

    model.eval()

    predictions = []

    # Inference Loop
    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs.to(device)

            # Forward pass
            outputs = model(inputs)

            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            predictions.extend(probs)

    # Retrieve IDs from dataset
    ids = test_dataset.get_ids()

    # Create Submission DataFrame
    submission_df = pd.DataFrame({"BraTS21ID": ids, "MGMT_value": predictions})

    # Format BraTS21ID as integer (as per sample submission format)
    submission_df["BraTS21ID"] = submission_df["BraTS21ID"].astype(int)

    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
