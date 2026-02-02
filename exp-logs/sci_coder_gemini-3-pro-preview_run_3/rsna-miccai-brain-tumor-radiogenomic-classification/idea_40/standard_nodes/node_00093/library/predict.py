import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import from provided library files
from library.utils import get_device, load_data_and_cache, SiameseDataset
from library.model import SiameseNetwork
from torch.utils.data import DataLoader


def generate_submission(
    test_meta_path="./metadata/test.parquet",
    model_path="./working/idea_41/best_model.pth",
    submission_path="./submission/submission.csv",
    cache_dir="./working/idea_41/",
    batch_size=16,
    num_workers=4,
    load_cached_data=True,
):
    """
    Loads the best trained model and generates predictions for the test set.
    """

    device = get_device()
    print(f"Running inference on device: {device}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at {model_path}. Please train the model first."
        )

    print(f"Loading test data from {test_meta_path}...")
    X_even, X_odd, _, test_ids = load_data_and_cache(
        metadata_path=test_meta_path,
        cache_dir=cache_dir,
        load_cached_data=load_cached_data,
        dataset_name="test",
    )

    test_dataset = SiameseDataset(X_even, X_odd, y=None)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print("Initializing model...")
    model = SiameseNetwork(
        model_name="efficientnet_b0",
        pretrained=False,
        drop_path_rate=0.0,
    )
    model.to(device)

    print(f"Loading weights from {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    predictions = []
    print("Starting inference...")

    with torch.no_grad():
        for batch in test_loader:
            xe, xo = batch
            xe, xo = xe.to(device), xo.to(device)
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
