import os
import torch
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import set_seed, rle_encode
from library.dataset import get_loaders
from library.model import DeepResUNet


def load_model(checkpoint_path, device):
    """
    Loads the DeepResUNet model from a checkpoint.

    Args:
        checkpoint_path (str): Path to the .pth file.
        device (torch.device): Device to load the model onto.

    Returns:
        model (nn.Module): Loaded model in eval mode, or None if path doesn't exist.
    """
    model = DeepResUNet()
    model = model.to(device)

    if os.path.exists(checkpoint_path):
        print(f"Loading weights from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(f"Warning: Checkpoint {checkpoint_path} not found.")
        return None

    model.eval()
    return model


def predict(limit_batches=None, model_paths=None):
    """
    Generates predictions for the test set using specified models.
    Applies Test-Time Augmentation (Horizontal Flip) and generates a submission file.

    Args:
        limit_batches (int, optional): If set, limits inference to N batches (for debugging).
        model_paths (list, optional): List of checkpoint paths to use. If None, auto-selects based on default logic.
    """
    # 1. Setup
    Config.setup()
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    print("Initializing Inference...")

    # 2. Data Loading
    # We only need the test loader. load_cached_data=True ensures we use pre-processed npy files if available.
    _, _, test_loader = get_loaders(load_cached_data=True)

    # 3. Model Loading
    models = []

    if model_paths:
        print(f"Loading models from provided paths: {model_paths}")
        for path in model_paths:
            m = load_model(path, device)
            if m is not None:
                models.append(m)
    else:
        # Default fallback logic if no paths provided
        path_c2 = os.path.join(Config.CHECKPOINT_DIR, "best_cycle_2.pth")
        path_c3 = os.path.join(Config.CHECKPOINT_DIR, "best_cycle_3.pth")

        model_c2 = load_model(path_c2, device)
        if model_c2 is not None:
            models.append(model_c2)

        model_c3 = load_model(path_c3, device)
        if model_c3 is not None:
            models.append(model_c3)

        if not models:
            print(
                "Cycle-specific checkpoints not found. Attempting to load global best model."
            )
            path_best = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
            model_best = load_model(path_best, device)
            if model_best is not None:
                models.append(model_best)

    if not models:
        raise FileNotFoundError(
            f"No valid checkpoints found in {Config.CHECKPOINT_DIR}"
        )

    print(f"Ensembling {len(models)} model(s).")

    # 4. Inference Loop
    submission_data = []

    # Calculate cropping indices to revert padding
    # Padding was added to reach (128, 128) from (101, 101)
    # dataset.py uses reflection padding: top=delta_h//2, left=delta_w//2
    pad_h = Config.IMG_HEIGHT - Config.ORIG_HEIGHT
    pad_w = Config.IMG_WIDTH - Config.ORIG_WIDTH
    top = pad_h // 2
    left = pad_w // 2

    print("Starting prediction loop...")

    with torch.no_grad():
        for batch_idx, (images, _, ids) in enumerate(test_loader):
            if limit_batches is not None and batch_idx >= limit_batches:
                break

            images = images.to(device)
            batch_size = images.size(0)

            # Accumulate probabilities from all models
            avg_probs = torch.zeros(
                (batch_size, 1, Config.IMG_HEIGHT, Config.IMG_WIDTH), device=device
            )

            for model in models:
                # Standard Prediction
                logits = model(images)
                probs = torch.sigmoid(logits)

                # Test-Time Augmentation (Horizontal Flip)
                if Config.TTA_FLIP:
                    # Flip input horizontally (dim 3: W)
                    images_flipped = torch.flip(images, dims=[3])
                    logits_flipped = model(images_flipped)
                    probs_flipped = torch.sigmoid(logits_flipped)

                    # Flip output back
                    probs_flipped_back = torch.flip(probs_flipped, dims=[3])

                    # Average original and flipped
                    probs = 0.5 * (probs + probs_flipped_back)

                avg_probs += probs

            # Average across ensemble members
            avg_probs /= len(models)

            # Move to CPU for post-processing
            avg_probs = avg_probs.cpu().numpy()

            # Process each image in the batch
            for i in range(batch_size):
                img_id = ids[i]
                prob_map = avg_probs[i, 0]  # (128, 128)

                # Crop back to original size (101, 101)
                # We extract the center region corresponding to the original image
                prob_map_cropped = prob_map[
                    top : top + Config.ORIG_HEIGHT, left : left + Config.ORIG_WIDTH
                ]

                # Threshold to binary mask
                # Using 0.5 as the standard decision boundary
                binary_mask = (prob_map_cropped > 0.5).astype(np.uint8)

                # Run-Length Encoding
                rle = rle_encode(binary_mask)

                submission_data.append([img_id, rle])

    # 5. Save Submission
    df_sub = pd.DataFrame(submission_data, columns=["id", "rle_mask"])

    # Ensure submission directory exists
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    save_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    df_sub.to_csv(save_path, index=False)
    print(f"Inference complete. Submission saved to {save_path}")
    print(f"Total predictions: {len(df_sub)}")
