import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from library.config import Config
from library.dataset import AnimalDataset, get_transforms
from library.model import AnimalModel


def run_inference(
    test_metadata_path=Config.TEST_METADATA_PATH,
    model_checkpoint=Config.MODEL_CHECKPOINT_PATH,
    submission_path=Config.SUBMISSION_PATH,
    batch_size=Config.BATCH_SIZE,
    device=Config.DEVICE,
):
    """
    Runs inference on the test set using the trained model with TTA and saves the submission file.

    Args:
        test_metadata_path (str): Path to the test metadata CSV.
        model_checkpoint (str): Path to the trained model weights.
        submission_path (str): Path to save the submission CSV.
        batch_size (int): Batch size for inference.
        device (str): Device to run inference on.
    """

    # 1. Load Test Metadata
    if not os.path.exists(test_metadata_path):
        raise FileNotFoundError(f"Test metadata file not found: {test_metadata_path}")

    df_test = pd.read_csv(test_metadata_path)
    print(f"Loaded test metadata: {len(df_test)} samples.")

    # 2. Prepare Dataset and DataLoader
    # Mode 'test' ensures __getitem__ returns only the image tensor
    test_dataset = AnimalDataset(
        df_test, transforms=get_transforms("test"), mode="test"
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Load Model
    print(f"Loading model architecture: {Config.MODEL_NAME}")
    # pretrained=False because we are loading our own weights
    model = AnimalModel(pretrained=False)

    if not os.path.exists(model_checkpoint):
        raise FileNotFoundError(f"Model checkpoint not found: {model_checkpoint}")

    print(f"Loading weights from {model_checkpoint}")
    state_dict = torch.load(model_checkpoint, map_location=device)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    # 4. Inference Loop with TTA
    print("Starting inference with Test-Time Augmentation (Horizontal Flip)...")
    all_preds = []

    # Use mixed precision for inference speed on A100
    with torch.no_grad():
        for images in test_loader:
            images = images.to(device, non_blocking=True)

            with torch.amp.autocast("cuda"):
                # Original Prediction
                logits = model(images)

                # TTA: Horizontal Flip
                # Flip along width dimension (dim 3 for NCHW)
                images_flipped = torch.flip(images, dims=[3])
                logits_flipped = model(images_flipped)

                # Average Logits
                avg_logits = (logits + logits_flipped) / 2.0

                # Get Class Predictions
                preds = torch.argmax(avg_logits, dim=1).cpu().numpy()
                all_preds.extend(preds)

    # 5. Generate Submission
    # Ensure submission directory exists
    os.makedirs(os.path.dirname(submission_path), exist_ok=True)

    # Create DataFrame
    # Using 'Predicted' as per Task Description "Submission Format"
    submission_df = pd.DataFrame({"Id": df_test["Id"], "Predicted": all_preds})

    # Save to CSV
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
    print("Head of submission:")
    print(submission_df.head())
