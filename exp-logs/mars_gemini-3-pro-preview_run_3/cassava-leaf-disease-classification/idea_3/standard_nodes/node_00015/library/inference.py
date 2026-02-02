import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import CFG, seed_everything
from library.model import CassavaViT
from library.dataset import CassavaDataset, get_transforms


def predict_tta(model, dataloader, device):
    """
    Performs inference with Test-Time Augmentation (TTA).
    Averages predictions from:
    1. Original Image
    2. Horizontal Flip
    3. Vertical Flip

    Args:
        model (nn.Module): The trained model.
        dataloader (DataLoader): DataLoader for the test set.
        device (str): Device to run inference on.

    Returns:
        np.ndarray: Array of predicted labels.
    """
    model.eval()
    final_preds = []

    # Disable gradients for inference
    with torch.no_grad():
        for images in dataloader:
            images = images.to(device)

            # 1. Forward pass - Original
            logits_orig = model(images)
            probs_orig = F.softmax(logits_orig, dim=1)

            # 2. Forward pass - Horizontal Flip
            # images shape: [Batch, Channels, Height, Width]
            # Flip along Width (dim 3)
            images_hflip = torch.flip(images, dims=[3])
            logits_hflip = model(images_hflip)
            probs_hflip = F.softmax(logits_hflip, dim=1)

            # 3. Forward pass - Vertical Flip
            # Flip along Height (dim 2)
            images_vflip = torch.flip(images, dims=[2])
            logits_vflip = model(images_vflip)
            probs_vflip = F.softmax(logits_vflip, dim=1)

            # Average probabilities
            avg_probs = (probs_orig + probs_hflip + probs_vflip) / 3.0

            # Get predicted class (argmax)
            preds = torch.argmax(avg_probs, dim=1).cpu().numpy()
            final_preds.extend(preds)

    return np.array(final_preds)


def run_inference(model_checkpoint_path, output_csv_path="./submission/submission.csv"):
    """
    Main driver function to run inference and generate submission file.

    Args:
        model_checkpoint_path (str): Path to the .pth model file.
        output_csv_path (str): Path where the submission CSV will be saved.
    """
    print(f"Starting inference using checkpoint: {model_checkpoint_path}")

    # Set seed for reproducibility
    seed_everything(CFG.seed)

    # 1. Load Data
    # Using the generated metadata for test set
    if not os.path.exists(CFG.test_csv):
        raise FileNotFoundError(f"Test metadata not found at {CFG.test_csv}")

    df_test = pd.read_csv(CFG.test_csv)

    # Dataset and DataLoader
    # output_label=False because test set labels are placeholders
    test_dataset = CassavaDataset(
        df_test, transform=get_transforms(data="test"), output_label=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
    )

    # 2. Load Model
    device = torch.device(CFG.device)
    # Initialize model architecture
    # Note: We set pretrained=False to avoid loading ImageNet weights since we load our own checkpoint
    model = CassavaViT(
        model_name=CFG.model_name, pretrained=False, num_classes=CFG.num_classes
    )
    model.to(device)

    # Load trained weights
    if not os.path.exists(model_checkpoint_path):
        raise FileNotFoundError(
            f"Model checkpoint not found at {model_checkpoint_path}"
        )

    checkpoint = torch.load(model_checkpoint_path, map_location=device)

    # Handle case where checkpoint is a dict with 'model_state_dict' key or just the state dict
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    print("Model weights loaded successfully.")

    # 3. Run Prediction with TTA
    print("Running prediction with TTA...")
    predictions = predict_tta(model, test_loader, device)

    # 4. Save Submission
    # Ensure output directory exists
    output_dir = os.path.dirname(output_csv_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    df_test["label"] = predictions

    # Keep only required columns
    submission_df = df_test[["image_id", "label"]]
    submission_df.to_csv(output_csv_path, index=False)

    print(f"Submission saved to {output_csv_path}")
    print(submission_df.head())
