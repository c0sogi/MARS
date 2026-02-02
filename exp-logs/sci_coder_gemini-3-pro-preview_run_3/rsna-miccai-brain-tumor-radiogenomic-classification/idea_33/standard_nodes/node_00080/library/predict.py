import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from library.utils import seed_everything, get_device
from library.model import VAMSNet
from library.data import process_dataset, VAMSDataset

# Constants
METADATA_DIR = "./metadata"
SUBMISSION_DIR = "./submission"


def generate_submission(
    model_path="./working/idea_33/best_model.pth",
    batch_size=32,
    load_cached_data=True,
    debug_limit=None,
):
    """
    Generates predictions for the test set using the trained VAMSNet model
    and saves the result to a CSV file.

    Args:
        model_path (str): Path to the trained model weights (.pth file).
        batch_size (int): Batch size for inference.
        load_cached_data (bool): Whether to use cached preprocessed data.
        debug_limit (int, optional): Limit the number of test samples for debugging.
    """
    # 1. Setup
    seed_everything(42)
    device = get_device()

    print(f"Generating submission using model: {model_path}")
    print(f"Device: {device}")

    # 2. Load Test Data
    test_meta_path = os.path.join(METADATA_DIR, "test.parquet")
    if not os.path.exists(test_meta_path):
        raise FileNotFoundError(f"Test metadata not found at {test_meta_path}")

    test_df = pd.read_parquet(test_meta_path)

    # Process test data (handles caching internally via library.data)
    # Note: process_dataset returns X, y, ids. For test set, y will be None.
    X_test, _, ids_test = process_dataset(
        test_df, "test", load_cached_data=load_cached_data, debug_limit=debug_limit
    )

    test_dataset = VAMSDataset(X_test, y=None)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # 3. Load Model
    # drop_path_rate=0.0 is used for inference to disable stochastic depth
    model = VAMSNet(drop_path_rate=0.0).to(device)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    # Load state dict
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # 4. Inference
    predictions = []

    print("Starting inference...")
    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs.to(device)

            # Forward pass
            outputs = model(inputs)

            # Apply Sigmoid to get probabilities [0, 1]
            probs = torch.sigmoid(outputs)

            # Flatten and store
            predictions.extend(probs.cpu().numpy().flatten())

    # 5. Save Submission
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    submission_df = pd.DataFrame({"BraTS21ID": ids_test, "MGMT_value": predictions})

    save_path = os.path.join(SUBMISSION_DIR, "submission.csv")
    submission_df.to_csv(save_path, index=False)
    print(f"Submission saved to {save_path}")
    print(submission_df.head())
