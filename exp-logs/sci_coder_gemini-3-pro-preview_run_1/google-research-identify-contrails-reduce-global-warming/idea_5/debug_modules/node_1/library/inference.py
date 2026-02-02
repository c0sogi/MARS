import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from library.config import Config
from library.dataset import ContrailDataset
from library.model import TemporalAshNet
from library.utils import rle_encode


def set_seed(seed):
    """
    Sets the seed for reproducibility.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_inference(
    checkpoint_path=None,
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    load_cached_data=False,
):
    """
    Runs inference on the test set using the trained TemporalAshNet model.

    Args:
        checkpoint_path (str, optional): Path to the model checkpoint.
                                         Defaults to Config.CHECKPOINT_DIR/best_model.pth.
        batch_size (int): Batch size for the dataloader.
        num_workers (int): Number of workers for data loading.
        load_cached_data (bool): If True, attempts to load predictions from a cached parquet file
                                 instead of running the model.

    Returns:
        pd.DataFrame: The submission dataframe containing record_id and encoded_pixels.
    """
    # 1. Setup
    Config.setup()
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if checkpoint_path is None:
        checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    cache_file = os.path.join(Config.PREDICTION_DIR, "raw_predictions.parquet")

    # 2. Caching Mechanism
    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading cached predictions from {cache_file}...")
        df_submission = pd.read_parquet(cache_file)
        # Ensure submission file is also generated from cache
        df_submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
        return df_submission

    # 3. Initialize Data
    print("Initializing Test Dataset...")
    test_dataset = ContrailDataset(Config.TEST_METADATA_PATH, stage="test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # 4. Initialize Model
    print(f"Loading model from {checkpoint_path}...")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    model = TemporalAshNet()
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 5. Inference Loop
    print("Starting Inference...")
    results = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            record_ids = batch["record_id"]

            # Forward pass
            logits = model(images)

            # Apply activation and threshold
            probs = torch.sigmoid(logits)
            preds = (probs > Config.THRESHOLD).float()

            # Move to CPU for encoding
            preds_np = preds.detach().cpu().numpy()

            # Iterate over batch to encode
            for i, record_id in enumerate(record_ids):
                # preds_np shape is (B, 1, H, W), we need (H, W)
                mask = preds_np[i, 0, :, :]

                # Run-Length Encoding
                encoded_pixels = rle_encode(mask)

                results.append(
                    {"record_id": record_id, "encoded_pixels": encoded_pixels}
                )

    # 6. Create DataFrame
    df_submission = pd.DataFrame(results)

    # 7. Save to Cache
    print(f"Saving raw predictions to cache: {cache_file}")
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    df_submission.to_parquet(cache_file, index=False)

    # 8. Save Submission
    print(f"Saving submission to {Config.SUBMISSION_PATH}")
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
    df_submission.to_csv(Config.SUBMISSION_PATH, index=False)

    print("Inference completed successfully.")
    return df_submission
