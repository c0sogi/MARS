import os
import torch
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.model import NarrowSEResNet
from library.dataset import CactusDataset, get_transforms
from library.utils import get_device, save_submission, load_checkpoint


def predict_with_tta(model, images):
    """
    Performs Test Time Augmentation (TTA) on a batch of images.
    Strategies: Original, Horizontal Flip, Vertical Flip.

    Args:
        model (nn.Module): The trained model.
        images (torch.Tensor): Batch of images [B, C, H, W].

    Returns:
        torch.Tensor: Averaged probabilities [B, 1].
    """
    # 1. Original
    logits_orig = model(images)
    probs_orig = torch.sigmoid(logits_orig)

    # 2. Horizontal Flip (dim 3 is width for [B, C, H, W])
    images_hflip = torch.flip(images, dims=[3])
    logits_hflip = model(images_hflip)
    probs_hflip = torch.sigmoid(logits_hflip)

    # 3. Vertical Flip (dim 2 is height for [B, C, H, W])
    images_vflip = torch.flip(images, dims=[2])
    logits_vflip = model(images_vflip)
    probs_vflip = torch.sigmoid(logits_vflip)

    # Average probabilities
    avg_probs = (probs_orig + probs_hflip + probs_vflip) / 3.0
    return avg_probs


def run_inference(debug=False):
    """
    Runs the inference pipeline:
    1. Loads test data.
    2. Iterates over all trained seeds.
    3. Performs TTA inference.
    4. Averages predictions across seeds.
    5. Saves submission file.

    Args:
        debug (bool): If True, runs on a subset of data.
    """
    device = get_device()
    print(f"Starting Inference on device: {device}")

    # 1. Prepare Test Data
    # We use 'val' transforms (ToTensor only) because TTA flips are applied manually
    test_dataset = CactusDataset(
        metadata_path=Config.TEST_METADATA_PATH,
        split="test",
        transform=get_transforms("val"),
        load_cached_data=True,
        debug=debug,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 2. Initialize Accumulators
    num_samples = len(test_dataset)
    accumulated_probs = np.zeros((num_samples, 1), dtype=np.float32)
    stored_ids = []
    processed_seeds_count = 0

    seeds = Config.SEEDS

    # 3. Ensemble Loop
    for seed_idx, seed in enumerate(seeds):
        model_path = os.path.join(Config.WORK_DIR, f"model_seed_{seed}.pth")

        if not os.path.exists(model_path):
            print(
                f"Warning: Model for seed {seed} not found at {model_path}. Skipping."
            )
            continue

        print(f"Processing Seed {seed} ({seed_idx + 1}/{len(seeds)})...")

        # Initialize Model
        model = NarrowSEResNet().to(device)
        load_checkpoint(model_path, model, device=device)
        model.eval()

        current_seed_probs = []
        current_seed_ids = []

        # Inference Loop
        with torch.no_grad():
            for images, _, ids in test_loader:
                images = images.to(device)

                # Predict with TTA
                probs = predict_with_tta(model, images)

                # Store results
                current_seed_probs.append(probs.cpu().numpy())

                # Only collect IDs on the first successful pass to ensure alignment
                if processed_seeds_count == 0:
                    current_seed_ids.extend(ids)

        # Concatenate batch results
        full_seed_probs = np.concatenate(current_seed_probs, axis=0)

        # Validate shape consistency
        if accumulated_probs.shape != full_seed_probs.shape:
            # This might happen if debug flag changed or dataset size mismatch
            print("Error: Shape mismatch in predictions. Resetting accumulator.")
            accumulated_probs = np.zeros_like(full_seed_probs)

        # Accumulate
        accumulated_probs += full_seed_probs

        if processed_seeds_count == 0:
            stored_ids = current_seed_ids

        processed_seeds_count += 1

    # 4. Finalize Predictions
    if processed_seeds_count > 0:
        final_probs = accumulated_probs / processed_seeds_count
    else:
        print("Error: No models were processed. Generating zero predictions.")
        final_probs = accumulated_probs

    # Flatten for CSV format
    final_probs = final_probs.flatten()

    # 5. Save Submission
    print(f"Saving submission to {Config.SUBMISSION_PATH}...")
    save_submission(stored_ids, final_probs, Config.SUBMISSION_PATH)
    print("Inference complete.")
