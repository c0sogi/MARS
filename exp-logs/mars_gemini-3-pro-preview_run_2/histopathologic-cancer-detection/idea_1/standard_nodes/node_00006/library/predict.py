import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from library.config import Config
from library.dataset import PathologyDataset, get_transforms
from library.model import TumorClassifier
from library.utils import set_seed


def generate_submission(
    checkpoint_path: str = None,
    output_path: str = Config.PREDICTION_FILE,
    batch_size: int = Config.BATCH_SIZE,
    num_workers: int = Config.NUM_WORKERS,
    device: str = Config.DEVICE,
    debug: bool = Config.DEBUG,
):
    """
    Generates predictions for the test set and saves them to a CSV file.

    Args:
        checkpoint_path (str, optional): Path to the trained model checkpoint.
                                         Defaults to Config.CHECKPOINT_DIR/best_model.pth.
        output_path (str): Path where the submission CSV will be saved.
        batch_size (int): Batch size for inference.
        num_workers (int): Number of workers for the dataloader.
        device (str): Device to run inference on ('cuda' or 'cpu').
        debug (bool): If True, runs on a subset of the test data.
    """
    # 1. Setup
    set_seed(Config.SEED)

    if checkpoint_path is None:
        checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")

    print(f"Loading model from {checkpoint_path}...")
    print(f"Running inference on device: {device}")

    # 2. Data Loading
    # We use the test metadata and test transforms
    test_dataset = PathologyDataset(
        metadata_path=Config.TEST_METADATA_PATH,
        transform=get_transforms("test"),
        debug=debug,
    )

    # Important: shuffle=False to maintain order matching the IDs
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Extract IDs from the dataset dataframe to ensure alignment
    # The dataset class slices the dataframe in __init__ if debug is True,
    # so this correctly reflects the data in the loader.
    test_ids = test_dataset.df["id"].values

    # 3. Model Initialization
    model = TumorClassifier(
        pretrained=False
    )  # Pretrained weights not needed as we load checkpoint

    # Load state dict
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    # 4. Inference Loop
    all_probs = []

    print("Starting inference with 8-view TTA (Cite Lesson 2)...")
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)

            # Accumulate probabilities from 8 views (D4 Dihedral Group)
            # 1. Original
            probs_sum = torch.sigmoid(model(images))

            # 2. Rotate 90
            probs_sum += torch.sigmoid(model(torch.rot90(images, 1, [2, 3])))

            # 3. Rotate 180
            probs_sum += torch.sigmoid(model(torch.rot90(images, 2, [2, 3])))

            # 4. Rotate 270
            probs_sum += torch.sigmoid(model(torch.rot90(images, 3, [2, 3])))

            # 5. Horizontal Flip
            img_h = torch.flip(images, [3])
            probs_sum += torch.sigmoid(model(img_h))

            # 6. Horizontal Flip + Rot 90
            probs_sum += torch.sigmoid(model(torch.rot90(img_h, 1, [2, 3])))

            # 7. Horizontal Flip + Rot 180
            probs_sum += torch.sigmoid(model(torch.rot90(img_h, 2, [2, 3])))

            # 8. Horizontal Flip + Rot 270
            probs_sum += torch.sigmoid(model(torch.rot90(img_h, 3, [2, 3])))

            # Average
            avg_probs = probs_sum / 8.0

            # Move to CPU and flatten
            all_probs.extend(avg_probs.cpu().numpy().flatten())

    # 5. Generate Submission File
    all_probs = np.array(all_probs)

    # Sanity check
    if len(all_probs) != len(test_ids):
        raise ValueError(
            f"Mismatch between number of predictions ({len(all_probs)}) "
            f"and number of test IDs ({len(test_ids)})."
        )

    # Create DataFrame
    submission_df = pd.DataFrame({"id": test_ids, "label": all_probs})

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to CSV
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
    print(f"Total predictions: {len(submission_df)}")
