import os
import torch
import pandas as pd
import numpy as np
from torch.cuda.amp import autocast

from library.config import Config
from library.dataset import get_dataloaders
from library.model import CatheterModel, set_seed


def create_submission(
    model_path=None, batch_size=Config.BATCH_SIZE, debug=Config.DEBUG
):
    """
    Loads the trained model, performs inference on the test set,
    and generates the submission CSV file.

    Args:
        model_path (str, optional): Path to the trained model weights.
                                    Defaults to best_model.pth in working dir.
        batch_size (int): Batch size for inference.
        debug (bool): If True, runs on a subset of data.
    """
    # 1. Setup
    set_seed(Config.SEED)
    device = Config.DEVICE

    if model_path is None:
        model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found at {model_path}")

    # 2. Data Loading
    # get_dataloaders automatically loads metadata from Config paths if dfs are None
    loaders = get_dataloaders(
        train_df=None, val_df=None, test_df=None, batch_size=batch_size, debug=debug
    )
    test_loader = loaders["test"]

    # 3. Model Loading
    # We set pretrained=False because we are loading our own fine-tuned weights
    model = CatheterModel(num_classes=Config.NUM_CLASSES, pretrained=False)

    # Load state dict
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    # 4. Inference Loop
    all_probs = []
    all_uids = []

    print(f"Starting inference on device: {device}")
    print(f"Model path: {model_path}")
    print(f"Test batches: {len(test_loader)}")

    with torch.no_grad():
        for images, uids in test_loader:
            images = images.to(device, non_blocking=True)

            # Use autocast for mixed precision inference (faster on compatible GPUs)
            with autocast():
                logits = model(images)
                probs = torch.sigmoid(logits)

            all_probs.append(probs.cpu().numpy())
            all_uids.extend(uids)

    # Concatenate all batches
    all_probs = np.concatenate(all_probs)

    # 5. Submission Generation
    # Create dictionary for DataFrame construction
    submission_data = {"StudyInstanceUID": all_uids}

    # Map probabilities to target columns
    for i, col_name in enumerate(Config.TARGET_COLS):
        submission_data[col_name] = all_probs[:, i]

    df_sub = pd.DataFrame(submission_data)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    # Save to CSV
    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)

    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(f"Submission shape: {df_sub.shape}")
    print("First 5 rows of submission:")
    print(df_sub.head())
