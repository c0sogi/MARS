import os
import torch
import numpy as np
import pandas as pd
from library.models import ModelFactory
from library.utils import load_checkpoint
from library.config import CACHE_DIR


def predict_with_tta(model, loader, device):
    """
    Generates predictions for 4 TTA views (Original, H-Flip, V-Flip, 180-Rot).

    Args:
        model (nn.Module): The trained model.
        loader (DataLoader): DataLoader for the dataset.
        device (str): Device to run inference on.

    Returns:
        tuple: (ids, raw_predictions, targets)
               raw_predictions is a numpy array of shape (N, 4).
    """
    model.eval()
    all_ids = []
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            ids = batch["id"]
            targets = batch.get("target", None)

            # Generate 4 views
            # View 0: Original
            v0 = images
            # View 1: Horizontal Flip
            v1 = torch.flip(images, dims=[3])
            # View 2: Vertical Flip
            v2 = torch.flip(images, dims=[2])
            # View 3: 180 Degree Rotation (Horizontal + Vertical Flip)
            v3 = torch.flip(images, dims=[2, 3])

            # Stack views along batch dimension: (4*B, C, H, W)
            # This allows a single forward pass for efficiency
            batch_stack = torch.cat([v0, v1, v2, v3], dim=0)

            # Forward pass
            outputs = model(batch_stack)

            # Handle models that return tuples (e.g., aux heads)
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            # Convert logits to probabilities
            probs = torch.sigmoid(outputs)

            # Unstack predictions
            # Shape: (4*B, 1) -> (4, B, 1)
            batch_size = images.size(0)
            p0, p1, p2, p3 = torch.split(probs, batch_size, dim=0)

            # Concatenate views per image: (B, 4)
            # Columns: [Original, H-Flip, V-Flip, 180-Rot]
            batch_preds = torch.cat([p0, p1, p2, p3], dim=1)

            all_preds.append(batch_preds.cpu().numpy())
            all_ids.extend(ids)

            if targets is not None:
                all_targets.append(targets.cpu().numpy())

    # Concatenate all batches
    all_preds = np.concatenate(all_preds, axis=0)

    if all_targets:
        all_targets = np.concatenate(all_targets, axis=0)
    else:
        all_targets = None

    return all_ids, all_preds, all_targets


def _process_predictions(ids, raw_preds, targets=None):
    """
    Computes Intrinsic Stability Features (Mean and Std) from TTA predictions.
    """
    # Calculate Mean and Standard Deviation across the 4 views
    mu = np.mean(raw_preds, axis=1)
    sigma = np.std(raw_preds, axis=1)

    data = {"id": ids, "pred_mean": mu, "pred_std": sigma}

    if targets is not None:
        data["target"] = targets

    return pd.DataFrame(data)


def generate_fold_predictions(
    model_name,
    fold_idx,
    checkpoint_path,
    val_loader,
    test_loader,
    device,
    load_cached_data=True,
):
    """
    Generates Out-Of-Fold (Val) and Test predictions for a specific model fold.
    Handles loading, re-parameterization, inference, and caching.

    Args:
        model_name (str): Name of the architecture (e.g., 'RepVGG').
        fold_idx (int): Fold index.
        checkpoint_path (str): Path to the model checkpoint.
        val_loader (DataLoader): Validation data loader.
        test_loader (DataLoader): Test data loader.
        device (str): Device.
        load_cached_data (bool): Whether to use cached predictions.

    Returns:
        tuple: (val_df, test_df) containing predictions and stability features.
    """
    # Define cache files
    val_cache_file = os.path.join(
        CACHE_DIR, f"preds_{model_name}_fold{fold_idx}_val.parquet"
    )
    test_cache_file = os.path.join(
        CACHE_DIR, f"preds_{model_name}_fold{fold_idx}_test.parquet"
    )

    # 1. Try loading from cache
    if load_cached_data:
        if os.path.exists(val_cache_file) and os.path.exists(test_cache_file):
            print(f"Loading cached predictions for {model_name} Fold {fold_idx}...")
            try:
                val_df = pd.read_parquet(val_cache_file)
                test_df = pd.read_parquet(test_cache_file)
                return val_df, test_df
            except Exception as e:
                print(f"Cache load failed ({e}), re-computing...")

    print(f"Generating predictions for {model_name} Fold {fold_idx}...")

    # 2. Initialize Model
    model = ModelFactory.get_model(model_name).to(device)

    # 3. Load Checkpoint
    try:
        load_checkpoint(checkpoint_path, model, device=device)
    except FileNotFoundError:
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    # 4. Structural Re-parameterization (for RepVGG)
    # Must be done in eval mode, after loading weights
    model.eval()
    if hasattr(model, "reparameterize"):
        print(f"Reparameterizing {model_name}...")
        model.reparameterize()

    # 5. Inference on Validation Set (OOF)
    val_ids, val_raw, val_targets = predict_with_tta(model, val_loader, device)
    val_df = _process_predictions(val_ids, val_raw, val_targets)

    # 6. Inference on Test Set
    test_ids, test_raw, test_targets = predict_with_tta(model, test_loader, device)
    test_df = _process_predictions(test_ids, test_raw, test_targets)

    # 7. Save to Cache
    os.makedirs(CACHE_DIR, exist_ok=True)
    val_df.to_parquet(val_cache_file, index=False)
    test_df.to_parquet(test_cache_file, index=False)

    return val_df, test_df
