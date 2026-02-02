import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import seed_everything
from library.model import StegoNet
from library.dataset import AlaskaDataset, get_transforms


def predict_tta(model, images, device):
    """
    Applies 5-View Test Time Augmentation to a batch of images.
    Views: Original, Horizontal Flip, Vertical Flip, Rot90, Rot270.

    Args:
        model (nn.Module): The trained model.
        images (torch.Tensor): Batch of images (B, C, H, W).
        device (torch.device): Device to run inference on.

    Returns:
        np.array: Averaged probabilities for the batch (B,).
    """
    # images: (B, C, H, W)
    x = images.to(device)

    # Create 5 views
    # 1. Original
    # 2. Horizontal Flip (W is dim 3)
    x_h = torch.flip(x, [3])
    # 3. Vertical Flip (H is dim 2)
    x_v = torch.flip(x, [2])
    # 4. Rotate 90 deg (k=1)
    x_r90 = torch.rot90(x, 1, [2, 3])
    # 5. Rotate 270 deg (k=3)
    x_r270 = torch.rot90(x, 3, [2, 3])

    # Stack along batch dimension: (B*5, C, H, W)
    # Order: [Batch_Orig, Batch_H, Batch_V, Batch_R90, Batch_R270]
    batch_stack = torch.cat([x, x_h, x_v, x_r90, x_r270], dim=0)

    # Forward pass
    with torch.no_grad():
        logits = model(batch_stack).view(-1)
        probs = torch.sigmoid(logits)

    # Split back into views to average per image
    # Split into 5 chunks of size B (original batch size)
    chunks = torch.chunk(probs, 5, dim=0)
    # Stack to (B, 5)
    probs_stacked = torch.stack(chunks, dim=1)

    # Average probability across views
    avg_probs = probs_stacked.mean(dim=1).cpu().numpy()

    return avg_probs


def generate_submission(checkpoint_path=None, debug=Config.debug):
    """
    Generates predictions for the test set using the best model and TTA.
    Saves the result to submission.csv.

    Args:
        checkpoint_path (str, optional): Path to model weights. Defaults to best_model.pth.
        debug (bool): If True, runs on a subset defined in Config.
    """
    # 1. Setup
    seed_everything(Config.seed)
    device = Config.device

    if checkpoint_path is None:
        checkpoint_path = os.path.join(Config.checkpoint_dir, "best_model.pth")

    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}. Cannot generate submission.")
        return

    print(f"Generating submission using checkpoint: {checkpoint_path}")
    print(f"Debug mode: {debug}")

    # 2. Data Loading
    # Temporarily override Config.debug to control dataset size via argument
    original_debug = Config.debug
    Config.debug = debug

    try:
        test_dataset = AlaskaDataset("test", transform=get_transforms("test"))
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.val_batch_size,
            shuffle=False,
            num_workers=Config.num_workers,
            pin_memory=True,
        )

        # 3. Model Initialization
        # Initialize model architecture without downloading pretrained weights
        # since we are loading a full state dictionary immediately after.
        model = StegoNet(
            backbone_name=Config.backbone_name,
            pretrained=False,
            num_classes=Config.num_classes,
        )
        model = model.to(device)

        # Load weights
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()

        # 4. Inference Loop
        results_id = []
        results_score = []

        print("Starting inference on test set...")

        for images, ids in test_loader:
            # images: (B, C, H, W)
            # ids: tuple of image IDs (e.g., '0001.jpg')

            avg_probs = predict_tta(model, images, device)

            results_id.extend(ids)
            results_score.extend(avg_probs)

        # 5. Save Submission
        df_sub = pd.DataFrame({"Id": results_id, "Label": results_score})

        # Ensure output directory exists
        os.makedirs(os.path.dirname(Config.submission_path), exist_ok=True)

        df_sub.to_csv(Config.submission_path, index=False)
        print(f"Submission saved to {Config.submission_path}")
        print(f"Total predictions: {len(df_sub)}")

    finally:
        # Restore original config state
        Config.debug = original_debug
