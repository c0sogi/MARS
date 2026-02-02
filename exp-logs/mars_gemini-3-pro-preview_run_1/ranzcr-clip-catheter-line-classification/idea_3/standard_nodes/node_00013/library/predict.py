import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import seed_everything
from library.dataset import CatheterDataset, get_transforms
from library.model import CatheterModel


def inference_fn(
    test_metadata_path=Config.test_metadata_path,
    model_path=Config.model_save_path,
    submission_output_dir="./submission",
    batch_size=Config.batch_size,
    debug=Config.debug,
    device=Config.device,
):
    """
    Runs inference on the test set using the trained model and generates a submission file.

    Args:
        test_metadata_path (str): Path to the test metadata CSV file.
        model_path (str): Path to the trained model weights (.pth file).
        submission_output_dir (str): Directory to save the submission.csv file.
        batch_size (int): Batch size for inference.
        debug (bool): If True, runs inference on a small subset of the test data.
        device (torch.device): Device to run inference on.
    """
    seed_everything(Config.seed)

    print(f"Inference Configuration:")
    print(f"  Test Metadata: {test_metadata_path}")
    print(f"  Model Path: {model_path}")
    print(f"  Output Dir: {submission_output_dir}")
    print(f"  Device: {device}")
    print(f"  Debug Mode: {debug}")

    # 1. Load Metadata
    if not os.path.exists(test_metadata_path):
        raise FileNotFoundError(f"Test metadata file not found at {test_metadata_path}")

    test_df = pd.read_csv(test_metadata_path)

    if debug:
        print(
            f"Debug mode enabled: Sampling {Config.debug_sample_size} rows from test set."
        )
        test_df = test_df.iloc[: Config.debug_sample_size].copy()

    print(f"Test set size: {len(test_df)}")

    # 2. Prepare Dataset and DataLoader
    # We use 'valid' transforms for testing (Resize + Normalize)
    test_transforms = get_transforms(data="valid")

    test_dataset = CatheterDataset(
        df=test_df, transforms=test_transforms, input_dir=Config.input_dir
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # 3. Initialize Model and Load Weights
    print(f"Initializing model: {Config.model_name}")
    # We set pretrained=False because we are loading custom weights
    model = CatheterModel(
        model_name=Config.model_name,
        pretrained=False,
        num_classes=Config.num_classes,
        in_channels=Config.in_channels,
    )

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights file not found at {model_path}")

    print(f"Loading weights from {model_path}...")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    # 4. Inference Loop with Test Time Augmentation (TTA)
    # TTA Strategy: Average predictions of original image and horizontally flipped image
    print("Starting inference with Horizontal Flip TTA...")

    all_probs = []

    with torch.no_grad():
        for i, (images, _) in enumerate(test_loader):
            images = images.to(device)

            # Forward pass 1: Original images
            logits_orig = model(images)
            probs_orig = torch.sigmoid(logits_orig)

            # Forward pass 2: Horizontally flipped images
            # Flip along width dimension (dim 3 for NCHW format)
            images_flipped = torch.flip(images, dims=[3])
            logits_flipped = model(images_flipped)
            probs_flipped = torch.sigmoid(logits_flipped)

            # Average probabilities
            avg_probs = (probs_orig + probs_flipped) / 2.0

            all_probs.append(avg_probs.cpu().numpy())

            if (i + 1) % 10 == 0:
                print(f"Processed batch {i + 1}/{len(test_loader)}")

    # Concatenate all batch predictions
    final_probs = np.concatenate(all_probs, axis=0)

    # 5. Generate Submission File
    print("Constructing submission DataFrame...")

    # Create DataFrame with target columns
    submission_df = pd.DataFrame(final_probs, columns=Config.target_cols)

    # Insert StudyInstanceUID as the first column
    submission_df.insert(0, "StudyInstanceUID", test_df["StudyInstanceUID"].values)

    # Ensure output directory exists
    os.makedirs(submission_output_dir, exist_ok=True)
    output_path = os.path.join(submission_output_dir, "submission.csv")

    # Save to CSV
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved successfully to {output_path}")

    return submission_df
