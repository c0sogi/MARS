import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from library.config import (
    WORKING_DIR,
    TEST_METADATA_PATH,
    STREAM_A_CONFIG,
    STREAM_B_CONFIG,
    DEVICE,
    TTA_ENABLED,
    TTA_VIEWS,
    SUBMISSION_FILE,
    NUM_WORKERS,
)
from library.utils import get_device, create_submission
from library.model import ResolutionPreservedUNet
from library.dataset import DenoisingDataset


def apply_tta(x, k):
    """
    Applies the k-th geometric transformation from the D4 group.

    Mapping:
    0: Identity
    1: Rot90
    2: Rot180
    3: Rot270
    4: HFlip
    5: HFlip + Rot90
    6: HFlip + Rot180
    7: HFlip + Rot270

    Args:
        x (torch.Tensor): Input tensor of shape (B, C, H, W).
        k (int): Transformation index (0-7).

    Returns:
        torch.Tensor: Transformed tensor.
    """
    # 1. Flip (if k >= 4)
    if k >= 4:
        x = torch.flip(x, dims=[3])  # Horizontal flip (dim 3 is width)

    # 2. Rotate
    rot_k = k % 4
    if rot_k > 0:
        x = torch.rot90(x, k=rot_k, dims=[2, 3])

    return x


def reverse_tta(x, k):
    """
    Reverses the k-th geometric transformation.
    Order is reversed: Inverse Rotation -> Inverse Flip.

    Args:
        x (torch.Tensor): Transformed tensor of shape (B, C, H, W).
        k (int): Transformation index (0-7).

    Returns:
        torch.Tensor: Original orientation tensor.
    """
    # 1. Reverse Rotation
    rot_k = k % 4
    if rot_k > 0:
        x = torch.rot90(x, k=-rot_k, dims=[2, 3])

    # 2. Reverse Flip
    if k >= 4:
        x = torch.flip(x, dims=[3])

    return x


def load_ensemble_models(device):
    """
    Loads all available trained models from Stream A and Stream B.

    Args:
        device (torch.device): Device to load models onto.

    Returns:
        list: A list of loaded PyTorch models.
    """
    models = []
    configs = [STREAM_A_CONFIG, STREAM_B_CONFIG]

    print("Loading ensemble models...")

    for config in configs:
        stream_name = config["name"]
        seeds = config["seeds"]

        for seed in seeds:
            model_path = os.path.join(WORKING_DIR, f"{stream_name}_seed_{seed}.pth")

            if os.path.exists(model_path):
                try:
                    model = ResolutionPreservedUNet()
                    model.load_state_dict(torch.load(model_path, map_location=device))
                    model.to(device)
                    model.eval()
                    models.append(model)
                    print(f"Loaded: {os.path.basename(model_path)}")
                except Exception as e:
                    print(f"Error loading {model_path}: {e}")
            else:
                print(f"Warning: Model checkpoint not found: {model_path}")

    if not models:
        raise RuntimeError("No models were loaded. Ensure training has completed.")

    print(f"Total models loaded: {len(models)}")
    return models


def predict_test_set(debug_max_samples=None):
    """
    Runs the inference pipeline on the test set using the ensemble of models.
    Generates the submission CSV file.

    Args:
        debug_max_samples (int, optional): Limit the number of test samples for debugging.
    """
    device = get_device()

    # 1. Load Metadata
    if not os.path.exists(TEST_METADATA_PATH):
        raise FileNotFoundError(f"Test metadata not found at {TEST_METADATA_PATH}")

    test_df = pd.read_csv(TEST_METADATA_PATH)

    if debug_max_samples is not None:
        test_df = test_df.head(debug_max_samples)

    # 2. Initialize Dataset and Loader
    # Note: We use a specific cache name for test to avoid conflicts
    dataset = DenoisingDataset(
        test_df,
        img_size=None,  # None triggers inference padding logic
        augment=False,
        cache_name="test_cache",
        load_cached_data=True,
    )

    # Batch size 1 is required because images have varying aspect ratios/sizes
    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True
    )

    # 3. Load Ensemble
    models = load_ensemble_models(device)

    # 4. Inference Loop
    predictions = {}
    print(f"Starting inference on {len(dataset)} images...")

    with torch.no_grad():
        for i, (noisy_t, _, img_ids) in enumerate(dataloader):
            img_id = img_ids[0]  # Unpack batch of size 1
            noisy_t = noisy_t.to(device)

            # Accumulator for ensemble averaging
            # Shape will be determined by the first prediction
            ensemble_accum = None
            count = 0

            # Iterate over all models in the ensemble
            for model in models:

                # Determine TTA views
                views = range(TTA_VIEWS) if TTA_ENABLED else [0]

                for k in views:
                    # Apply TTA
                    inputs = apply_tta(noisy_t, k)

                    # Forward Pass
                    outputs = model(inputs)

                    # Reverse TTA
                    outputs = reverse_tta(outputs, k)

                    # Accumulate
                    if ensemble_accum is None:
                        ensemble_accum = outputs
                    else:
                        ensemble_accum += outputs
                    count += 1

            # Average predictions
            avg_pred = ensemble_accum / count

            # Post-processing
            # 1. Convert to numpy
            pred_np = (
                avg_pred.squeeze(0).squeeze(0).cpu().numpy()
            )  # (H_padded, W_padded)

            # 2. Crop to original size
            # Retrieve original image to get dimensions
            # dataset.noisy_imgs is a dict {id: np.array}
            original_img = dataset.noisy_imgs[str(img_id)]
            orig_h, orig_w = original_img.shape

            # Center crop (since padding was symmetric reflection)
            curr_h, curr_w = pred_np.shape

            pad_h = curr_h - orig_h
            pad_w = curr_w - orig_w

            pt = pad_h // 2
            pl = pad_w // 2

            # Crop
            final_pred = pred_np[pt : pt + orig_h, pl : pl + orig_w]

            # Store prediction
            predictions[str(img_id)] = final_pred

            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(dataset)} images")

    # 5. Generate Submission
    create_submission(predictions, SUBMISSION_FILE)
