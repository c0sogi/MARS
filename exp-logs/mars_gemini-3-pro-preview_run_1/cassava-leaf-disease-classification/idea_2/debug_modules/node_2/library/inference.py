import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from tqdm import tqdm

from library.config import CFG
from library.utils import seed_everything
from library.dataset import CassavaDataset, get_transforms
from library.model import CassavaModel


def predict_tta(model, loader, device):
    """
    Performs inference with Test Time Augmentation (TTA).
    Strategies: Original, Horizontal Flip, Vertical Flip.
    """
    model.eval()
    final_preds = []

    # Disable gradient calculation for inference
    with torch.no_grad():
        for images in tqdm(loader, desc="Inference", disable=True):
            images = images.to(device)

            # 1. Original
            logits_orig = model(images)

            # 2. Horizontal Flip (dim 3 is width for NCHW)
            images_hflip = torch.flip(images, dims=[3])
            logits_hflip = model(images_hflip)

            # 3. Vertical Flip (dim 2 is height for NCHW)
            images_vflip = torch.flip(images, dims=[2])
            logits_vflip = model(images_vflip)

            # Average logits
            # We average logits before softmax. Alternatively, one could average probabilities.
            # Averaging logits is standard for ensembling same-model TTA.
            avg_logits = (logits_orig + logits_hflip + logits_vflip) / 3.0

            # Get predicted class
            preds = torch.argmax(avg_logits, dim=1).cpu().numpy()
            final_preds.extend(preds)

    return final_preds


def run_inference(load_cached_data=False):
    """
    Main function to load model, data, and generate submission.

    Args:
        load_cached_data (bool): Placeholder for consistency with requirements.
                                 Inference usually runs fresh.
    """
    # 1. Setup
    seed_everything(CFG.seed)
    device = CFG.device

    # 2. Load Data
    # Using metadata file as source of truth
    test_df = pd.read_csv(CFG.test_csv)

    # Create Dataset and Loader
    # output_label=False because we don't need targets for inference
    test_dataset = CassavaDataset(
        test_df, transform=get_transforms("valid"), output_label=False
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
    )

    # 3. Load Model
    model = CassavaModel(model_name=CFG.model_name, pretrained=False)

    # Construct model path
    model_path = os.path.join(CFG.output_dir, CFG.model_save_name)

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model weights not found at {model_path}. Please train the model first."
        )

    print(f"Loading model from {model_path}...")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)

    # 4. Run Inference
    print("Starting inference with TTA (Original + HFlip + VFlip)...")
    predictions = predict_tta(model, test_loader, device)

    # 5. Create Submission
    test_df["label"] = predictions

    # Select only required columns
    submission_df = test_df[["image_id", "label"]]

    # Ensure submission directory exists
    submission_dir = "./submission"
    os.makedirs(submission_dir, exist_ok=True)

    submission_path = os.path.join(submission_dir, "submission.csv")
    submission_df.to_csv(submission_path, index=False)

    print(f"Submission saved to {submission_path}")
    print(submission_df.head())
