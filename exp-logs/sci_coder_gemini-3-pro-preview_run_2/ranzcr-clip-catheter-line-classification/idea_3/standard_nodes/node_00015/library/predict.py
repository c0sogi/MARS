import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import seed_everything
from library.dataset import CatheterDataset, get_transforms
from library.model import CatheterModel


def predict(
    batch_size=Config.BATCH_SIZE,
    device=Config.DEVICE,
    metadata_path=Config.TEST_METADATA,
    model_path=Config.BEST_MODEL_PATH,
    output_path=Config.SUBMISSION_FILE,
):
    """
    Performs inference on the test dataset and generates the submission file.

    Args:
        batch_size (int): Batch size for inference.
        device (torch.device): Device to run inference on.
        metadata_path (str): Path to the test metadata CSV.
        model_path (str): Path to the trained model checkpoint.
        output_path (str): Path to save the submission CSV.
    """
    # 1. Setup
    seed_everything(Config.SEED)

    print(f"Loading test metadata from {metadata_path}...")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Test metadata not found at {metadata_path}")

    df_test = pd.read_csv(metadata_path)
    print(f"Test samples: {len(df_test)}")

    # 2. Dataset and Loader
    # We use the same transforms pipeline (Resize -> CLAHE -> Pad) to ensure consistency
    test_dataset = CatheterDataset(
        df_test, transforms=get_transforms(data="test"), mode="test"
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    # 3. Model Initialization
    # Initialize model structure without downloading pretrained weights
    print(f"Initializing model {Config.MODEL_NAME}...")
    model = CatheterModel(pretrained=False)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

    print(f"Loading weights from {model_path}...")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 4. Inference Loop
    all_preds = []
    print("Starting inference...")

    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)

            # Forward pass
            logits = model(images)

            # Apply Sigmoid to get probabilities
            probs = torch.sigmoid(logits)

            all_preds.append(probs.cpu().numpy())

    # Concatenate all batches
    all_preds = np.concatenate(all_preds, axis=0)

    # 5. Generate Submission
    # Create DataFrame with StudyInstanceUID and predicted probabilities
    submission_df = pd.DataFrame(all_preds, columns=Config.TARGET_COLS)
    submission_df.insert(0, "StudyInstanceUID", df_test["StudyInstanceUID"])

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to CSV
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved successfully to {output_path}")

    return submission_df
