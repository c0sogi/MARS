import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from library.config import Config
from library.dataset import get_dataloaders
from library.models import get_model
from library.utils import optimize_threshold, load_checkpoint


def predict_with_tta(model, dataloader, device, mode="val"):
    """
    Generates predictions using Test Time Augmentation (Horizontal Flip).

    Args:
        model (nn.Module): The trained model.
        dataloader (DataLoader): DataLoader for validation or test set.
        device (torch.device): Device to run inference on.
        mode (str): 'val' or 'test'.

    Returns:
        tuple: (all_probs, all_targets_or_ids)
            - all_probs: np.ndarray of shape (N, num_classes)
            - all_targets_or_ids: np.ndarray of targets (if val) or list of ids (if test)
    """
    model.eval()
    all_probs = []
    all_targets_or_ids = []

    # Determine if we are collecting targets (val) or IDs (test)
    # We check the first batch to see what the 3rd element is
    collecting_ids = False

    with torch.no_grad():
        for batch in dataloader:
            images, targets, ids = batch
            images = images.to(device)

            # 1. Forward pass original
            out_orig = model(images)
            probs_orig = torch.sigmoid(out_orig)

            # 2. Forward pass flipped (TTA)
            # Flip along width dimension (dim 3 for NCHW)
            images_flipped = torch.flip(images, dims=[3])
            out_flipped = model(images_flipped)
            probs_flipped = torch.sigmoid(out_flipped)

            # 3. Average probabilities
            avg_probs = (probs_orig + probs_flipped) / 2.0

            all_probs.append(avg_probs.cpu().numpy())

            # Collect targets or IDs
            # Note: targets in dataloader are tensors, ids are tuples of strings
            if isinstance(ids[0], str):
                collecting_ids = True
                all_targets_or_ids.extend(ids)
            else:
                all_targets_or_ids.append(targets.numpy())

    all_probs = np.concatenate(all_probs, axis=0)

    if not collecting_ids:
        all_targets_or_ids = np.concatenate(all_targets_or_ids, axis=0)

    return all_probs, all_targets_or_ids


def get_model_predictions(model_name, mode, dataloader, device, load_cached_data=True):
    """
    Gets predictions for a specific model and mode (val/test).
    Implements caching of the raw probability arrays.

    Args:
        model_name (str): Name of the model architecture.
        mode (str): 'val' or 'test'.
        dataloader (DataLoader): The data loader.
        device (torch.device): Compute device.
        load_cached_data (bool): Whether to use cached .npy files.

    Returns:
        tuple: (probs, targets_or_ids)
    """
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    probs_path = os.path.join(cache_dir, f"{model_name}_{mode}_probs.npy")
    meta_path = os.path.join(
        cache_dir, f"{model_name}_{mode}_meta.npy"
    )  # targets or ids

    # Try to load from cache
    if load_cached_data and os.path.exists(probs_path) and os.path.exists(meta_path):
        print(f"Loading cached predictions for {model_name} ({mode})...")
        try:
            probs = np.load(probs_path)
            meta = np.load(meta_path, allow_pickle=True)
            return probs, meta
        except Exception as e:
            print(f"Failed to load cache: {e}. Re-running inference.")

    # Run inference
    print(f"Running inference for {model_name} ({mode})...")

    # Load model and checkpoint
    model = get_model(model_name, num_classes=Config.NUM_CLASSES, pretrained=False)
    model = model.to(device)

    checkpoint_filename = f"{model_name}_best.pth"
    try:
        load_checkpoint(model, checkpoint_filename, device=device)
    except FileNotFoundError:
        print(
            f"Warning: Checkpoint {checkpoint_filename} not found. Using random weights (for debugging only)."
        )

    probs, meta = predict_with_tta(model, dataloader, device)

    # Save to cache
    np.save(probs_path, probs)
    np.save(meta_path, meta)

    # Clean up
    del model
    torch.cuda.empty_cache()

    return probs, meta


def ensemble_predictions(probs_list):
    """
    Averages a list of probability arrays.
    """
    if not probs_list:
        return None
    return np.mean(probs_list, axis=0)


def generate_submission(debug=Config.DEBUG, load_cached_data=True):
    """
    Main function to generate the submission file.

    1. Loads Val and Test loaders.
    2. Gets predictions for Model A and Model B (with TTA).
    3. Ensembles predictions.
    4. Optimizes threshold on Validation set.
    5. Applies threshold to Test set.
    6. Saves submission.csv.
    """
    device = torch.device(Config.DEVICE)

    # 1. Get DataLoaders
    # We don't need the train loader here
    _, val_loader, test_loader = get_dataloaders(
        debug=debug,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=load_cached_data,
    )

    models = [Config.MODEL_A_NAME, Config.MODEL_B_NAME]

    # 2. Validation Inference (for Threshold Calibration)
    print("--- Processing Validation Set ---")
    val_probs_list = []
    val_targets = None

    for model_name in models:
        probs, targets = get_model_predictions(
            model_name, "val", val_loader, device, load_cached_data
        )
        val_probs_list.append(probs)
        val_targets = targets  # Targets should be same for all models

    # Ensemble Validation
    val_ensemble_probs = ensemble_predictions(val_probs_list)

    # Optimize Threshold
    print("Optimizing threshold on ensemble...")
    best_threshold, best_score = optimize_threshold(val_targets, val_ensemble_probs)
    print(f"Optimal Threshold: {best_threshold}")
    print(f"Validation Micro-F1 Score with Optimal Threshold: {best_score}")

    # 3. Test Inference
    print("\n--- Processing Test Set ---")
    test_probs_list = []
    test_ids = None

    for model_name in models:
        probs, ids = get_model_predictions(
            model_name, "test", test_loader, device, load_cached_data
        )
        test_probs_list.append(probs)
        test_ids = ids  # IDs should be same for all models

    # Ensemble Test
    test_ensemble_probs = ensemble_predictions(test_probs_list)

    # 4. Generate Submission CSV
    print(f"Generating submission with threshold {best_threshold}...")

    # Binarize predictions
    predictions_bin = (test_ensemble_probs > best_threshold).astype(int)

    submission_rows = []
    for idx, image_id in enumerate(test_ids):
        # Get indices of positive classes
        pred_indices = np.where(predictions_bin[idx] == 1)[0]

        # Format as space-separated string
        pred_str = " ".join(map(str, pred_indices))

        submission_rows.append({"id": image_id, "attribute_ids": pred_str})

    submission_df = pd.DataFrame(submission_rows)

    # Save
    out_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    submission_df.to_csv(out_path, index=False)
    print(f"Submission saved to {out_path}")

    return best_score
