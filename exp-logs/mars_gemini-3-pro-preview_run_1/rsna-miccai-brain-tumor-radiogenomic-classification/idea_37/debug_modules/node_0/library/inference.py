import os
import glob
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.model import ARVSNet
from library.dataset import get_dataloader


def load_models(device):
    """
    Scans the cache directory for model checkpoints.
    Prioritizes 'best_model_fold*.pth' for ensembling.
    Falls back to 'best_model.pth' if no folds are found.
    """
    models = []

    # Pattern for fold models
    fold_pattern = os.path.join(Config.CACHE_DIR, "best_model_fold*.pth")
    fold_paths = sorted(glob.glob(fold_pattern))

    if len(fold_paths) > 0:
        print(f"Found {len(fold_paths)} fold models for ensemble inference.")
        checkpoint_paths = fold_paths
    else:
        # Fallback to single model
        single_model_path = os.path.join(Config.CACHE_DIR, "best_model.pth")
        if os.path.exists(single_model_path):
            print("Found single model checkpoint.")
            checkpoint_paths = [single_model_path]
        else:
            print(f"WARNING: No model checkpoints found in {Config.CACHE_DIR}.")
            return []

    # Load each model
    for path in checkpoint_paths:
        try:
            model = ARVSNet()
            state_dict = torch.load(path, map_location=device)
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            models.append(model)
            print(f"Loaded model from {path}")
        except Exception as e:
            print(f"Error loading model from {path}: {e}")

    return models


def predict_test_set(load_cached_geometry=True):
    """
    Main inference function.
    1. Loads trained models.
    2. Prepares test dataloader (calculating geometry if needed).
    3. Generates predictions.
    4. Saves submission.csv.
    """
    device = torch.device(Config.DEVICE)
    print(f"Running inference on device: {device}")

    # 1. Load Models
    models = load_models(device)
    if not models:
        raise RuntimeError("No models loaded. Cannot perform inference.")

    # 2. Prepare DataLoader
    # This handles geometry calculation (CoM, Depth) and caching internally
    try:
        test_loader = get_dataloader(
            split="test", load_cached_geometry=load_cached_geometry
        )
    except Exception as e:
        print(f"Error creating test dataloader: {e}")
        return

    # 3. Inference Loop
    all_ids = []
    all_probs = []

    print(f"Starting inference on {len(test_loader.dataset)} subjects...")

    with torch.no_grad():
        for i, (images, subject_ids) in enumerate(test_loader):
            images = images.to(device)

            batch_probs = []

            # Ensemble prediction
            for model in models:
                logits = model(images)
                probs = torch.sigmoid(logits)
                batch_probs.append(probs.cpu().numpy())

            # Average predictions across models: (Num_Models, Batch, 1) -> (Batch, 1)
            avg_probs = np.mean(batch_probs, axis=0)

            # Flatten to 1D array
            avg_probs = avg_probs.flatten()

            all_probs.extend(avg_probs.tolist())
            all_ids.extend(subject_ids)

    # 4. Create Submission
    submission_df = pd.DataFrame({"BraTS21ID": all_ids, "MGMT_value": all_probs})

    # Ensure output directory exists
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Save
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print("Head of submission:")
    print(submission_df.head())

    return submission_df
