import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from library.config import Config
from library.dataset import DogCatDataset, get_transforms
from library.models import create_model


def predict_with_tta(model, dataloader, device):
    """
    Generates predictions for a single model using Test Time Augmentation (TTA).
    Averages predictions from original and horizontally flipped images.

    Args:
        model (nn.Module): The trained PyTorch model.
        dataloader (DataLoader): DataLoader for the test set.
        device (str): Device to run inference on.

    Returns:
        tuple: (ids, predictions) where ids is a list of image IDs and
               predictions is a list of probabilities.
    """
    model.eval()
    all_preds = []
    all_ids = []

    with torch.no_grad():
        for images, ids in dataloader:
            images = images.to(device)

            # 1. Forward pass on original images
            logits_orig = model(images).squeeze(1)
            probs_orig = torch.sigmoid(logits_orig)

            # 2. Forward pass on horizontally flipped images (TTA)
            images_flipped = torch.flip(
                images, dims=[3]
            )  # Flip along width (N, C, H, W)
            logits_flip = model(images_flipped).squeeze(1)
            probs_flip = torch.sigmoid(logits_flip)

            # 3. Average probabilities
            probs_avg = (probs_orig + probs_flip) / 2.0

            all_preds.extend(probs_avg.cpu().numpy())
            all_ids.extend(ids.numpy())

    return all_ids, all_preds


def ensemble_predictions(device=Config.DEVICE, load_cached_data=True):
    """
    Performs ensemble inference using all models defined in Config.MODEL_SPECS.
    Implements caching to avoid re-computing predictions for models that have
    already been processed.

    Args:
        device (str): Device to run inference on.
        load_cached_data (bool): If True, attempts to load predictions from
                                 parquet cache before running inference.
    """
    # Ensure directories exist
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    print("\nStarting Ensemble Inference...")

    ensemble_preds = None
    test_ids = None
    valid_models_count = 0

    # Iterate over all defined models
    for model_key, spec in Config.MODEL_SPECS.items():
        cache_path = os.path.join(Config.WORKING_DIR, f"preds_{model_key}.parquet")

        current_ids = None
        current_preds = None

        # --- Strategy: Cache Lookup ---
        if load_cached_data and os.path.exists(cache_path):
            print(f"[{model_key}] Loading cached predictions from {cache_path}...")
            try:
                df_cache = pd.read_parquet(cache_path)
                current_ids = df_cache["id"].values
                current_preds = df_cache["prob"].values
            except Exception as e:
                print(f"[{model_key}] Failed to load cache: {e}. Re-computing...")
                current_preds = None

        # --- Strategy: Compute if not cached ---
        if current_preds is None:
            checkpoint_path = os.path.join(Config.WORKING_DIR, f"{model_key}_best.pth")

            if not os.path.exists(checkpoint_path):
                print(
                    f"[{model_key}] Checkpoint not found at {checkpoint_path}. Skipping model."
                )
                continue

            print(f"[{model_key}] Running inference...")

            # Setup Data (Resolution Specific)
            img_size = spec["img_size"]
            test_dataset = DogCatDataset(
                split="test",
                img_size=img_size,
                transform=get_transforms(img_size, is_train=False),
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=Config.BATCH_SIZE,
                shuffle=False,
                num_workers=Config.NUM_WORKERS,
                pin_memory=True,
            )

            # Setup Model
            model = create_model(model_key, pretrained=False)
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            model = model.to(device)

            # Predict with TTA
            current_ids, current_preds = predict_with_tta(model, test_loader, device)

            # Convert to numpy
            current_ids = np.array(current_ids)
            current_preds = np.array(current_preds)

            # Save to Cache
            try:
                df_cache = pd.DataFrame({"id": current_ids, "prob": current_preds})
                df_cache.to_parquet(cache_path)
                print(f"[{model_key}] Predictions cached to {cache_path}.")
            except Exception as e:
                print(f"[{model_key}] Warning: Failed to save cache: {e}")

        # --- Aggregation ---
        if ensemble_preds is None:
            ensemble_preds = current_preds
            test_ids = current_ids
        else:
            # Sanity check for ID alignment
            if not np.array_equal(test_ids, current_ids):
                raise ValueError(
                    f"ID mismatch detected for model {model_key}. Ensemble alignment failed."
                )
            ensemble_preds += current_preds

        valid_models_count += 1

    # --- Final Submission Generation ---
    if valid_models_count == 0:
        print(
            "Error: No valid models were found for inference. Cannot generate submission."
        )
        return

    print(f"\nAggregating predictions from {valid_models_count} models...")
    final_preds = ensemble_preds / valid_models_count

    submission_df = pd.DataFrame({"id": test_ids, "label": final_preds})

    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved successfully to {Config.SUBMISSION_PATH}")
