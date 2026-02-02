import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import CFG
from library.dataset import CassavaDataset, get_transforms, load_metadata
from library.model import CassavaClassifier
from library.utils import seed_everything


def inference_fn(model, data_loader, device):
    """
    Runs inference on the test loader using Test-Time Augmentation (TTA).
    Averages predictions across:
    1. Original image
    2. Horizontally flipped image
    3. Vertically flipped image

    Args:
        model (nn.Module): The trained model.
        data_loader (DataLoader): Test data loader.
        device (torch.device): Computation device.

    Returns:
        np.array: Predicted labels.
    """
    model.eval()
    final_preds = []

    with torch.no_grad():
        for images in data_loader:
            images = images.to(device)

            # 1. Original
            output_orig = model(images)
            probs_orig = F.softmax(output_orig, dim=1)

            # 2. Horizontal Flip
            images_h = torch.flip(images, dims=[-1])
            output_h = model(images_h)
            probs_h = F.softmax(output_h, dim=1)

            # 3. Vertical Flip
            images_v = torch.flip(images, dims=[-2])
            output_v = model(images_v)
            probs_v = F.softmax(output_v, dim=1)

            # Average probabilities
            avg_probs = (probs_orig + probs_h + probs_v) / 3.0

            # Get predicted class
            preds = torch.argmax(avg_probs, dim=1)
            final_preds.extend(preds.cpu().numpy())

    return np.array(final_preds)


def generate_submission(load_cached_data=True):
    """
    Main function to generate the submission file.

    Args:
        load_cached_data (bool): Whether to use cached metadata.
    """
    # Set seed for reproducibility
    seed_everything(CFG.seed)

    # Setup device
    device = CFG.device

    # Load Test Metadata
    # load_metadata handles caching logic internally as per library.dataset
    test_df = load_metadata("test", load_cached_data=load_cached_data)

    # Create Dataset and DataLoader
    # We use 'valid' transforms for inference (Resize + Normalize)
    test_dataset = CassavaDataset(
        df=test_df,
        file_root=CFG.test_root,
        transform=get_transforms("valid", CFG.img_size),
        output_label=False,
        seed=CFG.seed,
        is_training=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=CFG.valid_batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
    )

    # Initialize Model
    model = CassavaClassifier(
        model_name=CFG.model_name,
        pretrained=False,  # We load our own weights
        num_classes=CFG.num_classes,
        img_size=CFG.img_size,
    )

    # Load Best Weights
    weights_path = os.path.join(CFG.output_dir, "best_model.pth")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights not found at {weights_path}")

    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)

    print(f"Loaded model weights from {weights_path}")

    # Run Inference
    print("Starting inference with TTA...")
    predictions = inference_fn(model, test_loader, device)

    # Create Submission DataFrame
    test_df["label"] = predictions
    submission_df = test_df[["image_id", "label"]]

    # Ensure submission directory exists
    submission_dir = "./submission"
    os.makedirs(submission_dir, exist_ok=True)

    submission_path = os.path.join(submission_dir, "submission.csv")
    submission_df.to_csv(submission_path, index=False)

    print(f"Submission saved to {submission_path}")
    print(submission_df.head())
