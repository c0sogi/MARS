import os
import torch
import pandas as pd
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import set_seed, rle_encode
from library.model import ContextEnhancedUNet
from library.dataset import ContrailsDataset


def predict_and_submit(
    model_path=Config.BEST_MODEL_PATH,
    metadata_path=Config.TEST_METADATA_PATH,
    output_path=Config.SUBMISSION_PATH,
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    device=None,
    debug=False,
    debug_sample_size=Config.DEBUG_SAMPLE_SIZE,
):
    """
    Runs inference on the test dataset using the trained model and generates
    a submission CSV file in Run-Length Encoding (RLE) format.

    Args:
        model_path (str): Path to the trained model weights.
        metadata_path (str): Path to the test metadata CSV.
        output_path (str): Path where the submission CSV will be saved.
        batch_size (int): Batch size for inference.
        num_workers (int): Number of worker threads for data loading.
        device (torch.device, optional): Device to run inference on. If None, detects automatically.
        debug (bool): If True, runs on a subset of the data.
        debug_sample_size (int): Number of samples to use in debug mode.
    """
    # 1. Setup
    set_seed(Config.SEED)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Starting inference on device: {device}")

    # 2. Load Metadata
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Test metadata not found at {metadata_path}")

    test_df = pd.read_csv(metadata_path)

    if debug:
        print(f"Debug mode: Subsetting test data to {debug_sample_size} samples.")
        test_df = test_df.head(debug_sample_size)

    print(f"Total test samples: {len(test_df)}")

    # 3. Prepare Data Loader
    # train=False ensures no augmentations and returns dummy masks
    test_dataset = ContrailsDataset(test_df, train=False)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # 4. Load Model
    model = ContextEnhancedUNet()

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found at {model_path}")

    # Load weights
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    # 5. Inference Loop
    submission_data = []

    print("Running prediction loop...")

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            record_ids = batch["record_id"]

            # Forward pass
            logits = model(images)

            # Apply Sigmoid to get probabilities
            probs = torch.sigmoid(logits)

            # Thresholding (Standard 0.5 threshold as per strategy)
            # Output shape: (B, 1, H, W)
            preds = (probs > Config.THRESHOLD).float()

            # Move to CPU for encoding
            preds_cpu = preds.cpu().numpy()

            # Iterate through batch to encode
            for i in range(len(record_ids)):
                # Extract single mask: (1, H, W) -> (H, W)
                mask = preds_cpu[i, 0, :, :]
                r_id = record_ids[i]

                # Run-Length Encode
                encoded_string = rle_encode(mask)

                submission_data.append(
                    {"record_id": str(r_id), "encoded_pixels": encoded_string}
                )

    # 6. Generate Submission File
    submission_df = pd.DataFrame(submission_data)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    submission_df.to_csv(output_path, index=False)

    print(f"Submission saved to {output_path}")
    print(f"Generated predictions for {len(submission_df)} records.")
