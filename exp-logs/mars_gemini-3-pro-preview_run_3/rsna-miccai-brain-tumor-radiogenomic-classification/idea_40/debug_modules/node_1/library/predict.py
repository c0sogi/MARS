import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import from provided library files
from library.utils import get_device, load_data_and_cache, SiameseDataset
from library.model import SiameseNetwork


def generate_submission(
    test_meta_path="./metadata/test.parquet",
    model_path="./working/idea_40/best_model.pth",
    submission_path="./submission/submission.csv",
    cache_dir="./working/idea_40/",
    batch_size=16,
    num_workers=4,
    load_cached_data=True,
):
    """
    Loads the best trained model and generates predictions for the test set.

    Args:
        test_meta_path (str): Path to the test metadata Parquet file.
        model_path (str): Path to the saved model state dict.
        submission_path (str): Path where the submission CSV will be saved.
        cache_dir (str): Directory to store/load cached numpy arrays.
        batch_size (int): Batch size for inference.
        num_workers (int): Number of workers for data loading.
        load_cached_data (bool): Whether to use cached data if available.
    """

    # 1. Setup Device
    device = get_device()
    print(f"Running inference on device: {device}")

    # 2. Check Model Existence
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at {model_path}. Please train the model first."
        )

    # 3. Load Test Data
    # load_data_and_cache returns X_even, X_odd, y (None for test), ids
    print(f"Loading test data from {test_meta_path}...")
    X_test_even, X_test_odd, _, test_ids = load_data_and_cache(
        metadata_path=test_meta_path,
        cache_dir=cache_dir,
        load_cached_data=load_cached_data,
        dataset_name="test",
    )

    # 4. Prepare DataLoader
    # We pass y=None because it's the test set
    test_dataset = SiameseDataset(X_test_even, X_test_odd, y=None)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # Important: Must be False to match IDs
        num_workers=num_workers,
        pin_memory=True,
    )

    # 5. Initialize Model and Load Weights
    print("Initializing model...")
    model = SiameseNetwork(
        model_name="efficientnet_b0",
        pretrained=False,  # Pretrained weights not needed as we load state_dict
        drop_path_rate=0.0,  # No stochastic depth during inference
    )
    model.to(device)

    print(f"Loading weights from {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 6. Inference Loop
    predictions = []
    print("Starting inference...")

    with torch.no_grad():
        for batch in test_loader:
            # SiameseDataset returns (xe, xo) when y is None
            if len(batch) == 2:
                xe, xo = batch
            else:
                xe, xo, _ = batch

            xe = xe.to(device)
            xo = xo.to(device)

            # Forward pass
            logits = model(xe, xo).squeeze(1)
            probs = torch.sigmoid(logits).cpu().numpy()
            predictions.extend(probs)

    # 7. Generate Submission File
    # Ensure IDs and Predictions align
    if len(test_ids) != len(predictions):
        raise ValueError(
            f"Mismatch: {len(test_ids)} IDs vs {len(predictions)} predictions."
        )

    print(f"Saving submission to {submission_path}...")
    os.makedirs(os.path.dirname(submission_path), exist_ok=True)

    submission_df = pd.DataFrame({"BraTS21ID": test_ids, "MGMT_value": predictions})

    submission_df.to_csv(submission_path, index=False)
    print("Submission generation complete.")
