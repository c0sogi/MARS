import os
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import get_device, load_checkpoint
from library.model import get_model
from library.dataset import get_test_dataloader, process_and_cache_data


def predict_with_tta(model, images, device):
    """
    Performs inference using Test-Time Augmentation (TTA).
    Averages predictions from the original image and a horizontally flipped version.

    Args:
        model (nn.Module): The trained model.
        images (torch.Tensor): Batch of images (B, C, H, W).
        device (torch.device): The device to run inference on.

    Returns:
        torch.Tensor: Softmax probabilities averaged across views (B, Num_Classes).
    """
    # 1. Forward pass on original images
    output_orig = model(images)
    probs_orig = torch.softmax(output_orig, dim=1)

    # 2. Forward pass on horizontally flipped images
    # Image tensor is (Batch, Channel, Height, Width). Flip on Width (dim 3).
    images_flipped = torch.flip(images, dims=[3])
    output_flipped = model(images_flipped)
    probs_flipped = torch.softmax(output_flipped, dim=1)

    # 3. Average predictions
    avg_probs = (probs_orig + probs_flipped) / 2.0
    return avg_probs


def generate_ensemble_submission():
    """
    Generates the final submission file by ensembling predictions from all trained fold models.
    Uses TTA for each model and averages the results.
    Saves the output to ./submission/submission.csv.
    """
    device = get_device()
    print(f"Starting Ensemble Inference on device: {device}")

    # 1. Get Class Names
    # We load cached data to ensure we have the exact class ordering used during training
    _, class_names = process_and_cache_data(load_cached_data=True)

    # 2. Prepare Data Loader
    test_loader = get_test_dataloader()

    # 3. Initialize Accumulators
    ensemble_probs = None
    all_ids = []
    models_used_count = 0

    # 4. Iterate through each fold's model
    for fold_idx in range(Config.N_FOLDS):
        ckpt_path = os.path.join(Config.OUTPUT_DIR, f"model_fold_{fold_idx}.pth")

        if not os.path.exists(ckpt_path):
            print(f"Checkpoint not found for Fold {fold_idx} at {ckpt_path}. Skipping.")
            continue

        print(f"Processing Fold {fold_idx}...")

        # Initialize model and load weights
        model = get_model(Config.MODEL_NAME, Config.NUM_CLASSES, pretrained=False)
        model.to(device)
        load_checkpoint(model, None, ckpt_path)
        model.eval()

        fold_probs_list = []
        collect_ids = len(all_ids) == 0  # Only collect IDs once

        with torch.no_grad():
            for images, ids in test_loader:
                images = images.to(device)

                # Get TTA predictions
                probs = predict_with_tta(model, images, device)
                fold_probs_list.append(probs.cpu())

                if collect_ids:
                    all_ids.extend(ids)

        # Concatenate all batches for this fold
        fold_probs_tensor = torch.cat(fold_probs_list, dim=0)

        # Accumulate into ensemble
        if ensemble_probs is None:
            ensemble_probs = fold_probs_tensor
        else:
            ensemble_probs += fold_probs_tensor

        models_used_count += 1

        # Cleanup to save memory
        del model
        torch.cuda.empty_cache()

    # 5. Finalize Predictions
    if models_used_count == 0:
        raise RuntimeError("No models were found for inference!")

    # Compute arithmetic mean
    avg_probs = ensemble_probs / models_used_count
    avg_probs_np = avg_probs.numpy()

    # 6. Create Submission DataFrame
    print("Constructing submission DataFrame...")
    df_submission = pd.DataFrame(avg_probs_np, columns=class_names)
    df_submission.insert(0, "id", all_ids)

    # 7. Save to Disk
    submission_dir = "./submission"
    os.makedirs(submission_dir, exist_ok=True)
    save_path = os.path.join(submission_dir, "submission.csv")

    df_submission.to_csv(save_path, index=False)
    print(f"Submission saved successfully to {save_path}")
    print(f"Ensemble composed of {models_used_count} models.")
