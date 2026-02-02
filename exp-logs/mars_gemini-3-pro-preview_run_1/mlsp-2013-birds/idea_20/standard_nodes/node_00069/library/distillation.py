import os
import torch
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import load_checkpoint, set_seed
from library.data import get_dataloaders
from library.models import create_model


def generate_pseudo_labels(load_cached_data=True):
    """
    Generates pseudo-labels for the test set using the heterogeneous teacher ensemble.
    Applies Test-Time Augmentation (TTA) and ensemble averaging.

    Args:
        load_cached_data (bool): If True, attempts to load previously generated
                                 pseudo-labels from disk.

    Returns:
        pd.DataFrame: DataFrame containing 'rec_id' and soft probability columns
                      (species_0 to species_18).
    """
    set_seed(Config.SEED)

    # 1. Caching Logic
    if load_cached_data and os.path.exists(Config.PSEUDO_LABELS_PATH):
        print(f"Loading cached pseudo-labels from {Config.PSEUDO_LABELS_PATH}")
        try:
            return pd.read_parquet(Config.PSEUDO_LABELS_PATH)
        except Exception as e:
            print(f"Failed to load cache: {e}. Regenerating...")

    print("Generating pseudo-labels with Heterogeneous Teacher Ensemble...")

    # 2. Setup Data
    # We only need the test loader here.
    # Passing pseudo_labels_path=None because we are creating them now.
    dataloaders = get_dataloaders(pseudo_labels_path=None)
    test_loader = dataloaders["test"]

    # Prepare storage for ensemble predictions
    # We need to know the number of samples. The loader might be batched.
    # We'll collect predictions list-wise and concatenate.
    ensemble_probs = None
    rec_ids_list = []

    # Collect rec_ids once (order is deterministic with shuffle=False)
    for batch in test_loader:
        rec_ids_list.append(batch["rec_id"].numpy())
    rec_ids = np.concatenate(rec_ids_list)
    num_samples = len(rec_ids)

    # Initialize accumulator
    ensemble_accumulator = np.zeros((num_samples, Config.NUM_CLASSES), dtype=np.float32)

    # 3. Ensemble Inference Loop
    device = Config.DEVICE

    for i, arch in enumerate(Config.TEACHER_ARCHS):
        print(f"Inference with Teacher {i+1}/{len(Config.TEACHER_ARCHS)}: {arch}")

        # Instantiate model (pretrained=False because we load custom weights)
        model = create_model(arch, num_classes=Config.NUM_CLASSES, pretrained=False)

        # Load SWA weights
        checkpoint_path = Config.get_teacher_path(i, arch)
        try:
            load_checkpoint(checkpoint_path, model, device=device)
        except FileNotFoundError:
            print(
                f"Error: Checkpoint not found at {checkpoint_path}. Ensure teachers are trained."
            )
            raise

        model.to(device)
        model.eval()

        model_preds = []

        with torch.no_grad():
            for batch in test_loader:
                images = batch["image"].to(device)

                # --- Test-Time Augmentation (TTA) ---
                # 1. Original Forward Pass
                logits_orig = model(images)

                # 2. Horizontal Flip Forward Pass
                images_flipped = torch.flip(images, dims=[3])  # [B, C, H, W]
                logits_flip = model(images_flipped)

                # Average Logits
                avg_logits = (logits_orig + logits_flip) / 2.0

                # Convert to Probabilities
                probs = torch.sigmoid(avg_logits)
                model_preds.append(probs.cpu().numpy())

        # Concatenate predictions for this model
        model_probs = np.concatenate(model_preds, axis=0)

        # Accumulate to ensemble
        ensemble_accumulator += model_probs

    # 4. Ensemble Averaging
    final_probs = ensemble_accumulator / len(Config.TEACHER_ARCHS)

    # 5. Sanitization
    if np.isnan(final_probs).any():
        print("Warning: NaN values detected in pseudo-labels. Replacing with zeros.")
        final_probs = np.nan_to_num(final_probs, nan=0.0)

    # 6. Formatting Output
    # Create DataFrame
    cols = [f"species_{k}" for k in range(Config.NUM_CLASSES)]
    df_pseudo = pd.DataFrame(final_probs, columns=cols)
    df_pseudo.insert(0, "rec_id", rec_ids)

    # 7. Save to Cache
    os.makedirs(os.path.dirname(Config.PSEUDO_LABELS_PATH), exist_ok=True)
    df_pseudo.to_parquet(Config.PSEUDO_LABELS_PATH, index=False)
    print(f"Pseudo-labels saved to {Config.PSEUDO_LABELS_PATH}")

    return df_pseudo
