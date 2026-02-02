import os
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from library.config import Config
from library.models import AppleNet
from library.dataset import get_loaders, load_data
from library.utils import seed_everything


def predict_tta(model, loader, device):
    """
    Generates predictions with Test-Time Augmentation (TTA).
    Strategy: Original, Horizontal Flip, Vertical Flip, Transpose.

    Args:
        model (nn.Module): The trained model.
        loader (DataLoader): Test data loader.
        device (str): Device to run inference on.

    Returns:
        np.ndarray: Averaged probability predictions (N, Num_Classes).
    """
    model.eval()
    preds_list = []

    with torch.no_grad():
        for images in loader:
            images = images.to(device)

            # 1. Original
            out_1 = model(images)
            p1 = torch.softmax(out_1, dim=1)

            # 2. Horizontal Flip (Flip W dimension)
            out_2 = model(torch.flip(images, dims=[3]))
            p2 = torch.softmax(out_2, dim=1)

            # 3. Vertical Flip (Flip H dimension)
            out_3 = model(torch.flip(images, dims=[2]))
            p3 = torch.softmax(out_3, dim=1)

            # 4. Transpose (Swap H and W dimensions)
            out_4 = model(torch.transpose(images, 2, 3))
            p4 = torch.softmax(out_4, dim=1)

            # Average probabilities
            avg_probs = (p1 + p2 + p3 + p4) / 4.0
            preds_list.append(avg_probs.cpu())

    return torch.cat(preds_list, dim=0).numpy()


def run_inference():
    """
    Main inference orchestration.
    Loads models, performs TTA inference, ensembles predictions, and saves submission.
    """
    # Ensure reproducibility for consistent data sampling in DEBUG mode
    seed_everything(Config.SEED)

    # 1. Load Data
    # get_loaders handles caching and debug sampling internally for the loader
    _, _, test_loader = get_loaders(load_cached_data=True)

    # Load raw dataframe to get image_ids
    _, _, test_df = load_data(load_cached_data=True)

    # Synchronize dataframe with loader if in DEBUG mode
    if Config.DEBUG:
        test_df = test_df.sample(
            min(len(test_df), Config.DEBUG_SAMPLE_SIZE)
        ).reset_index(drop=True)

    ensemble_preds = []

    # 2. Iterate over Backbones (Heterogeneous Ensemble)
    for backbone_name in Config.BACKBONES:
        print(f"Running inference for backbone: {backbone_name}")

        # Initialize Model
        # pretrained=False because we are loading our own fine-tuned weights
        model = AppleNet(backbone_name, Config.NUM_CLASSES, pretrained=False)

        # Construct path to the best weights
        weights_path = os.path.join(Config.WORKING_DIR, f"{backbone_name}_best.pth")

        if not os.path.exists(weights_path):
            print(f"  -> Weights not found at {weights_path}. Skipping this model.")
            continue

        # Load Weights
        # These weights are already the EMA weights if USE_EMA was enabled during training
        state_dict = torch.load(weights_path, map_location=Config.DEVICE)
        model.load_state_dict(state_dict)
        model.to(Config.DEVICE)

        # Predict using TTA
        preds = predict_tta(model, test_loader, Config.DEVICE)
        ensemble_preds.append(preds)

        # Cleanup to free GPU memory
        del model, state_dict
        gc.collect()
        torch.cuda.empty_cache()

    if not ensemble_preds:
        raise RuntimeError(
            "No predictions generated. Check if model weights exist in the working directory."
        )

    # 3. Ensemble (Simple Average)
    # Shape: (N_samples, N_classes)
    final_preds = np.mean(ensemble_preds, axis=0)

    # 4. Create Submission DataFrame
    submission = pd.DataFrame()
    submission["image_id"] = test_df["image_id"]

    for i, label in enumerate(Config.LABELS):
        submission[label] = final_preds[:, i]

    # 5. Save Submission
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(submission.head())
