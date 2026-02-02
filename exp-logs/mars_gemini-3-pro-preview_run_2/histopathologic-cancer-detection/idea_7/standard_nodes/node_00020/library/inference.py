import os
import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F

from library.config import Config
from library.utils import set_seed
from library.dataset import get_test_dataloader
from library.models import get_model


def load_ensemble(device):
    """
    Loads the heterogeneous ensemble of models (ConvNeXt-Tiny + EfficientNetV2-S).
    Loads 5 folds for each architecture, totaling 10 models.
    Uses EMA weights if available, as they provide better generalization.

    Args:
        device (torch.device): The device to load models onto.

    Returns:
        list[nn.Module]: A list of loaded PyTorch models in evaluation mode.
    """
    models = []

    print(f"Loading ensemble models from {Config.CHECKPOINT_DIR}...")

    for arch in Config.MODEL_ARCHS:
        for fold in range(Config.N_FOLDS):
            # Construct expected filename for the best model of this fold
            # Logic matches save_checkpoint in library/utils.py
            filename = f"best_model_{arch}_fold_{fold}.pth"
            checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, filename)

            if not os.path.exists(checkpoint_path):
                print(f"Warning: Checkpoint not found at {checkpoint_path}. Skipping.")
                continue

            # Initialize model architecture
            # num_classes=1 for binary classification
            model = get_model(arch, pretrained=False, num_classes=1)
            model.to(device)
            model.eval()

            # Load checkpoint
            try:
                checkpoint = torch.load(checkpoint_path, map_location=device)

                # Prioritize EMA weights if they exist
                if "ema_state_dict" in checkpoint:
                    state_dict = checkpoint["ema_state_dict"]
                else:
                    state_dict = checkpoint["state_dict"]

                # Handle DataParallel prefix 'module.' if present
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith("module."):
                        new_state_dict[k[7:]] = v
                    else:
                        new_state_dict[k] = v

                model.load_state_dict(new_state_dict)
                models.append(model)

            except Exception as e:
                print(f"Error loading {filename}: {e}")

    print(f"Successfully loaded {len(models)} models.")
    return models


def get_tta_views(images):
    """
    Generates 8 Dihedral views (D4 group) for Test Time Augmentation.
    This exploits the rotational invariance of the pathology patches.

    Args:
        images (torch.Tensor): Input batch of shape (B, C, H, W).

    Returns:
        list[torch.Tensor]: List of 8 tensors representing the augmented views.
    """
    views = []

    # 1. Identity
    views.append(images)
    # 2. Rotate 90
    views.append(torch.rot90(images, 1, [2, 3]))
    # 3. Rotate 180
    views.append(torch.rot90(images, 2, [2, 3]))
    # 4. Rotate 270
    views.append(torch.rot90(images, 3, [2, 3]))

    # 5. Horizontal Flip
    hflip = torch.flip(images, [3])
    views.append(hflip)
    # 6. HFlip + Rotate 90 (equivalent to Vertical Flip + Transpose variants)
    views.append(torch.rot90(hflip, 1, [2, 3]))
    # 7. HFlip + Rotate 180
    views.append(torch.rot90(hflip, 2, [2, 3]))
    # 8. HFlip + Rotate 270
    views.append(torch.rot90(hflip, 3, [2, 3]))

    return views


def run_inference(load_cached_data=True):
    """
    Orchestrates the inference process.
    1. Loads the test dataset.
    2. Loads the ensemble of models.
    3. Iterates through the data, applying TTA and averaging predictions.
    4. Saves the results to a submission CSV.

    Args:
        load_cached_data (bool): Whether to use cached .npy files for the dataset.
    """
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 1. Load Data
    print("Initializing Test DataLoader...")
    # get_test_dataloader handles caching internally via library.dataset logic
    test_loader = get_test_dataloader(load_cached_data=load_cached_data)

    # 2. Load Models
    models = load_ensemble(device)
    if not models:
        print("Error: No models loaded. Cannot proceed with inference.")
        return

    # 3. Inference Loop
    print(f"Starting Inference on {len(test_loader.dataset)} images...")
    print(f"Strategy: Ensemble of {len(models)} models x {Config.TTA_VIEWS} TTA views.")

    all_ids = []
    all_preds = []

    with torch.no_grad():
        for i, (images, ids) in enumerate(test_loader):
            images = images.to(device)
            batch_size = images.size(0)

            # Generate TTA views: List of 8 tensors
            views = get_tta_views(images)

            # Accumulator for this batch: (B, 1)
            # We sum probabilities across all models and all views
            batch_accum = torch.zeros((batch_size, 1), device=device)

            # Iterate over all models
            for model in models:
                # Iterate over all views
                for view in views:
                    logits = model(view)
                    probs = torch.sigmoid(logits)
                    batch_accum += probs

            # Average: divide by (num_models * num_views)
            num_predictions = len(models) * len(views)
            batch_avg = batch_accum / num_predictions

            # Store results
            # Flatten to 1D array
            all_preds.extend(batch_avg.cpu().numpy().flatten())
            all_ids.extend(ids)

            if (i + 1) % 50 == 0:
                print(f"Processed batch {i + 1}/{len(test_loader)}")

    # 4. Save Submission
    print("Generating submission file...")

    # Create DataFrame
    df_sub = pd.DataFrame({"id": all_ids, "label": all_preds})

    # Ensure output directory exists
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Save to CSV
    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)

    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(f"Total predictions: {len(df_sub)}")
    print("First 5 rows:")
    print(df_sub.head())
