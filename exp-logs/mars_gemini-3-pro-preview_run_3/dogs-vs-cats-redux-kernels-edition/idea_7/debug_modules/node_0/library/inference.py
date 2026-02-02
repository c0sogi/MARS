import os
import numpy as np
import pandas as pd
import torch
from library.config import Config
from library.utils import get_device
from library.data import get_dataloaders
from library.model_factory import create_model


def predict_one_epoch(model, loader, device, use_tta=False):
    """
    Generates predictions for a single model on a dataloader.

    Args:
        model: The PyTorch model.
        loader: The DataLoader.
        device: The device to run on.
        use_tta (bool): Whether to use Test Time Augmentation (Horizontal Flip).

    Returns:
        tuple: (ids, probabilities)
    """
    model.eval()
    all_preds = []
    all_ids = []

    with torch.no_grad():
        for images, img_ids in loader:
            images = images.to(device)

            # Forward pass (Original)
            logits = model(images)
            probs = torch.sigmoid(logits)

            if use_tta:
                # Forward pass (Horizontal Flip)
                # NCHW format: dim 3 is width
                images_flipped = torch.flip(images, dims=[3])
                logits_flipped = model(images_flipped)
                probs_flipped = torch.sigmoid(logits_flipped)

                # Average probabilities
                probs = (probs + probs_flipped) / 2.0

            # Store results
            # Flatten to ensure 1D arrays
            all_preds.append(probs.cpu().numpy().flatten())
            all_ids.append(img_ids.numpy().flatten())

    # Concatenate all batches
    if len(all_preds) > 0:
        return np.concatenate(all_ids), np.concatenate(all_preds)
    else:
        return np.array([]), np.array([])


def run_inference(
    load_cached_data: bool = True,
    debug: bool = Config.DEBUG,
    debug_subset_size: int = Config.DEBUG_SUBSET_SIZE,
):
    """
    Runs the inference pipeline: loads models, predicts on test set, ensembles results, and saves submission.

    Args:
        load_cached_data (bool): Whether to use cached metadata.
        debug (bool): Whether to run in debug mode (subset of data).
        debug_subset_size (int): Size of subset for debug mode.
    """
    device = get_device()
    print(f"Starting inference on device: {device}")

    # Load Test DataLoader
    # Note: We only need the test loader here.
    _, _, test_loader = get_dataloaders(
        load_cached_data=load_cached_data,
        debug=debug,
        debug_subset_size=debug_subset_size,
    )

    model_backbones = Config.MODEL_BACKBONES
    accumulated_preds = None
    sample_ids = None
    models_used_count = 0

    print(f"Ensembling models: {model_backbones}")

    for model_name in model_backbones:
        # Construct checkpoint path
        checkpoint_path = os.path.join(Config.WORKING_DIR, f"{model_name}_best.pth")

        if not os.path.exists(checkpoint_path):
            print(
                f"Warning: Checkpoint not found for {model_name} at {checkpoint_path}. Skipping this model."
            )
            continue

        print(f"Processing model: {model_name}...")

        # Initialize model
        # pretrained=False because we load specific weights immediately after
        try:
            model = create_model(model_name, pretrained=False, num_classes=1)
            model.to(device)

            # Load weights
            state_dict = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(state_dict)

            # Generate predictions
            ids, preds = predict_one_epoch(
                model, test_loader, device, use_tta=Config.USE_TTA
            )

            # Initialize or Accumulate
            if accumulated_preds is None:
                accumulated_preds = preds
                sample_ids = ids
            else:
                # Ensure alignment
                if not np.array_equal(sample_ids, ids):
                    raise ValueError(f"ID mismatch encountered for model {model_name}")
                accumulated_preds += preds

            models_used_count += 1
            print(f"Model {model_name} processed successfully.")

        except Exception as e:
            print(f"Error processing model {model_name}: {e}")
            continue

    if models_used_count == 0:
        print(
            "Error: No models were successfully processed. Cannot generate submission."
        )
        return

    # Average predictions
    final_preds = accumulated_preds / models_used_count

    # Create Submission DataFrame
    submission_df = pd.DataFrame({"id": sample_ids, "label": final_preds})

    # Ensure IDs are integers (as per sample submission)
    submission_df["id"] = submission_df["id"].astype(int)

    # Sort by ID
    submission_df = submission_df.sort_values("id")

    # Save to CSV
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)

    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(f"Total models ensembled: {models_used_count}")
    print(submission_df.head())
