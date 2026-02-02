import os
import torch
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import seed_everything
from library.data import get_test_loader
from library.model import AppleClassifier


def predict_model(model_name, img_size, fold_idx, device):
    """
    Performs inference for a specific model and fold using TTA.
    Returns a dictionary mapping image_id to predicted probabilities [rust, scab].
    """
    # Sanitize model name to match training save format
    safe_model_name = model_name.replace(".", "_")
    weights_path = os.path.join(
        Config.WORKING_DIR, f"best_model_{safe_model_name}_fold_{fold_idx}.pth"
    )

    if not os.path.exists(weights_path):
        print(
            f"Warning: Model weights not found at {weights_path}. Skipping this model."
        )
        return None

    # Load Model
    model = AppleClassifier(model_name=model_name, pretrained=False)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Load Data
    loader = get_test_loader(img_size=img_size, load_cached_data=True)

    preds_dict = {}

    with torch.no_grad():
        for images, image_ids in loader:
            images = images.to(device)

            # Test Time Augmentation (Horizontal Flip)
            # Original
            logits_orig = model(images)
            probs_orig = torch.sigmoid(logits_orig)

            # Flipped
            images_flipped = torch.flip(images, dims=[3])
            logits_flip = model(images_flipped)
            probs_flip = torch.sigmoid(logits_flip)

            # Average
            avg_probs = (probs_orig + probs_flip) / 2.0
            avg_probs = avg_probs.cpu().numpy()

            for idx, img_id in enumerate(image_ids):
                preds_dict[img_id] = avg_probs[idx]

    return preds_dict


def generate_submission():
    """
    Main inference routine.
    Aggregates predictions from all models/folds, reconstructs class probabilities,
    and saves the submission file.
    """
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Define the ensemble configuration
    # (Model Name, Image Size)
    model_configs = [
        (Config.MODEL_A_NAME, Config.IMG_SIZE_EFFNET),
        (Config.MODEL_B_NAME, Config.IMG_SIZE_CONVNEXT),
    ]

    # Storage for aggregated predictions
    # Map: image_id -> np.array([prob_rust, prob_scab])
    ensemble_preds = {}
    models_count = 0

    print("Starting Inference and Ensemble Aggregation...")

    for model_name, img_size in model_configs:
        for fold in range(Config.N_FOLDS):
            print(f"Processing {model_name} | Fold {fold}...")

            fold_preds = predict_model(model_name, img_size, fold, device)

            if fold_preds is None:
                continue

            models_count += 1

            for img_id, probs in fold_preds.items():
                if img_id not in ensemble_preds:
                    ensemble_preds[img_id] = np.zeros_like(probs)
                ensemble_preds[img_id] += probs

    if models_count == 0:
        print("Error: No models were loaded. Cannot generate submission.")
        return

    print(f"Aggregating predictions from {models_count} models...")

    # Prepare final data
    final_data = []

    # Process aggregated predictions
    # Sort by image_id to ensure consistent order (though not strictly required by dict)
    sorted_ids = sorted(ensemble_preds.keys())

    for img_id in sorted_ids:
        # Average the accumulated probabilities
        avg_probs = ensemble_preds[img_id] / models_count

        p_r = avg_probs[0]  # Probability of Rust
        p_s = avg_probs[1]  # Probability of Scab

        # Reconstruct 4-class probabilities
        # Healthy: Neither Rust nor Scab
        p_healthy = (1 - p_r) * (1 - p_s)

        # Multiple: Both Rust and Scab
        p_multiple = p_r * p_s

        # Rust (Only): Rust but not Scab
        p_rust_only = p_r * (1 - p_s)

        # Scab (Only): Scab but not Rust
        p_scab_only = (1 - p_r) * p_s

        final_data.append(
            {
                "image_id": img_id,
                "healthy": p_healthy,
                "multiple_diseases": p_multiple,
                "rust": p_rust_only,
                "scab": p_scab_only,
            }
        )

    # Create DataFrame
    submission_df = pd.DataFrame(final_data)

    # Ensure column order matches sample submission
    cols = ["image_id", "healthy", "multiple_diseases", "rust", "scab"]
    submission_df = submission_df[cols]

    # Save
    save_path = Config.SUBMISSION_FILE
    submission_df.to_csv(save_path, index=False)
    print(f"Submission saved to {save_path}")
    print(submission_df.head())
