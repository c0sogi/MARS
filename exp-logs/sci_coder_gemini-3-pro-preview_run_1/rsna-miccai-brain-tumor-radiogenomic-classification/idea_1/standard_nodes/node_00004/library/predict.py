import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import seed_everything
from library.dataset import MGMTDataset, get_transforms
from library.model import MGMTClassifier


def generate_submission(
    batch_size=Config.BATCH_SIZE,
    device=Config.DEVICE,
    num_workers=Config.NUM_WORKERS,
    load_cached_data=True,
):
    """
    Generates the submission file for the test set.

    Args:
        batch_size (int): Batch size for inference.
        device (str): Device to run inference on ('cuda' or 'cpu').
        num_workers (int): Number of workers for data loading.
        load_cached_data (bool): Whether to use cached preprocessed data.

    Returns:
        pd.DataFrame: The generated submission dataframe.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    print(f"Initializing prediction on device: {device}")

    # 2. Data Loading
    # We use the 'val' transforms for testing (Resize + Normalize, no augmentation)
    test_dataset = MGMTDataset(
        metadata_path=Config.TEST_METADATA_PATH,
        split="test",
        transform=get_transforms("val"),
        load_cached_data=load_cached_data,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # 3. Model Setup
    # Initialize model architecture
    # We set pretrained=False to avoid downloading weights, as we will load our own checkpoint
    model = MGMTClassifier(
        model_name=Config.BACKBONE,
        pretrained=False,
        num_classes=Config.NUM_CLASSES,
        dropout_rate=Config.DROPOUT_RATE,
    )

    # Load trained weights
    if not os.path.exists(Config.MODEL_PATH):
        raise FileNotFoundError(
            f"Model checkpoint not found at {Config.MODEL_PATH}. "
            "Please ensure the model has been trained before running prediction."
        )

    print(f"Loading model weights from {Config.MODEL_PATH}...")
    state_dict = torch.load(Config.MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 4. Inference
    all_ids = []
    all_probs = []

    print(f"Starting inference on {len(test_dataset)} samples...")

    with torch.no_grad():
        for images, _, subject_ids in test_loader:
            images = images.to(device)

            # Forward pass: (B, 3, H, W) -> (B, 1)
            logits = model(images)

            # Apply sigmoid to get probabilities [0, 1]
            probs = torch.sigmoid(logits)

            # Move to CPU and flatten
            probs_np = probs.cpu().numpy().flatten()
            ids_np = subject_ids.numpy().flatten()

            all_ids.extend(ids_np)
            all_probs.extend(probs_np)

    # 5. Create Submission DataFrame
    df_submission = pd.DataFrame({"BraTS21ID": all_ids, "MGMT_value": all_probs})

    # Ensure BraTS21ID is integer type
    df_submission["BraTS21ID"] = df_submission["BraTS21ID"].astype(int)

    # 6. Save Submission
    df_submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print("First 5 predictions:")
    print(df_submission.head())

    return df_submission
