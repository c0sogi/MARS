import os
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F

from library.config import CFG
from library.model import AppleClassifier
from library.dataset import prepare_test_loader
from library.utils import seed_everything


def predict_tta(model, loader, device):
    """
    Performs Test-Time Augmentation (TTA) inference.
    Averages predictions from: Original, Horizontal Flip, Vertical Flip.

    Args:
        model (nn.Module): The trained model.
        loader (DataLoader): Test data loader.
        device (torch.device): Computation device.

    Returns:
        np.array: Array of predicted probabilities (N_samples, N_classes).
    """
    model.eval()
    preds = []

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)

            # 1. Original
            out_orig = model(images)
            prob_orig = torch.softmax(out_orig, dim=1)

            # 2. Horizontal Flip (dim 3 is width)
            images_h = torch.flip(images, dims=[3])
            out_h = model(images_h)
            prob_h = torch.softmax(out_h, dim=1)

            # 3. Vertical Flip (dim 2 is height)
            images_v = torch.flip(images, dims=[2])
            out_v = model(images_v)
            prob_v = torch.softmax(out_v, dim=1)

            # Average probabilities
            avg_prob = (prob_orig + prob_h + prob_v) / 3.0
            preds.append(avg_prob.cpu().numpy())

    return np.concatenate(preds)


def generate_submission(load_cached_data=False):
    """
    Generates the final submission file by ensembling predictions from all trained models.
    Implements caching for individual model predictions.

    Args:
        load_cached_data (bool): If True, attempts to load predictions from cache.
                                 If False or cache missing, computes and saves cache.
    """
    seed_everything(CFG.seed)
    device = CFG.device

    # Ensure output directory for cache exists
    os.makedirs(CFG.output_dir, exist_ok=True)

    final_preds = None
    model_count = 0
    test_image_ids = None

    print(
        f"Starting inference ensemble. Backbones: {CFG.backbones}, Folds: {CFG.n_folds}"
    )

    for backbone in CFG.backbones:
        # Prepare loader for this backbone (resolution dependent)
        loader, ids = prepare_test_loader(backbone)

        # Store image_ids from the first successful loader
        if test_image_ids is None:
            test_image_ids = ids

        for fold in range(CFG.n_folds):
            # Define paths
            model_path = os.path.join(CFG.output_dir, f"{backbone}_fold{fold}_best.pth")
            cache_path = os.path.join(
                CFG.output_dir, f"preds_{backbone}_fold{fold}.npy"
            )

            preds = None

            # 1. Try to load from cache
            if load_cached_data:
                if os.path.exists(cache_path):
                    try:
                        print(
                            f"Loading cached predictions for {backbone} fold {fold}..."
                        )
                        preds = np.load(cache_path)
                    except Exception as e:
                        print(f"Failed to load cache: {e}. Re-computing.")
                        preds = None
                else:
                    print(f"Cache not found for {backbone} fold {fold}. Re-computing.")

            # 2. Compute if not loaded
            if preds is None:
                if not os.path.exists(model_path):
                    print(
                        f"Model checkpoint not found: {model_path}. Skipping this model."
                    )
                    continue

                print(f"Running inference for {backbone} fold {fold}...")

                # Load Model
                model = AppleClassifier(backbone, pretrained=False)
                state_dict = torch.load(model_path, map_location=device)
                model.load_state_dict(state_dict)
                model.to(device)

                # Predict with TTA
                preds = predict_tta(model, loader, device)

                # Save to cache
                np.save(cache_path, preds)

                # Clean up
                del model
                torch.cuda.empty_cache()

            # Accumulate
            if final_preds is None:
                final_preds = np.zeros_like(preds)

            final_preds += preds
            model_count += 1

    if model_count == 0:
        print("Error: No models were found or processed. Cannot generate submission.")
        return

    # Average predictions
    avg_preds = final_preds / model_count
    print(f"Ensembled {model_count} models.")

    # Create Submission DataFrame
    # Ensure columns are in the correct order as defined in config
    sub_df = pd.DataFrame({"image_id": test_image_ids})

    for i, label in enumerate(CFG.class_labels):
        sub_df[label] = avg_preds[:, i]

    # Save Submission
    save_path = os.path.join(CFG.submission_dir, "submission.csv")
    sub_df.to_csv(save_path, index=False)

    print(f"Submission saved to {save_path}")
    print("Head of submission:")
    print(sub_df.head())
