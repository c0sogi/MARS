import os
import numpy as np
import pandas as pd
import torch
from library.config import Config
from library.dataset import get_test_loader
from library.models import get_model


def predict_batch_tta(model, images, device):
    """
    Predicts probabilities for a batch of images using Test-Time Augmentation (Horizontal Flip).

    Args:
        model (nn.Module): The trained model.
        images (torch.Tensor): Batch of images (B, C, H, W).
        device (torch.device): The computation device.

    Returns:
        torch.Tensor: Averaged probabilities for the batch.
    """
    # 1. Forward pass on original images
    logits = model(images)
    probs = torch.sigmoid(logits)

    # 2. Forward pass on horizontally flipped images
    # Flip along the width dimension (dim=3 for NCHW format)
    images_flipped = torch.flip(images, dims=[3])
    logits_flipped = model(images_flipped)
    probs_flipped = torch.sigmoid(logits_flipped)

    # 3. Average the probabilities
    avg_probs = (probs + probs_flipped) / 2.0
    return avg_probs


def run_inference(
    checkpoint_dir=Config.CHECKPOINT_DIR, submission_dir=Config.SUBMISSION_DIR
):
    """
    Runs inference on the test set using an ensemble of all available checkpoints.
    Applies Test-Time Augmentation (TTA) and averages predictions across models.
    Saves the final submission file.

    Args:
        checkpoint_dir (str): Directory containing model checkpoints.
        submission_dir (str): Directory to save the submission CSV.
    """
    device = torch.device(Config.DEVICE)
    test_loader = get_test_loader()

    # Accumulator for predictions: pd.Series indexed by ID
    ensemble_preds = None
    model_count = 0

    print(f"Starting inference on device: {device}")

    # Iterate over all architectures and folds defined in Config
    for model_name in Config.MODEL_ARCHS:
        for fold_idx in range(Config.N_FOLDS):
            ckpt_filename = f"{model_name}_fold_{fold_idx}.pth"
            ckpt_path = os.path.join(checkpoint_dir, ckpt_filename)

            # Skip if checkpoint does not exist (e.g., if training was interrupted or partial)
            if not os.path.exists(ckpt_path):
                print(f"Checkpoint not found: {ckpt_path}. Skipping.")
                continue

            print(f"Processing Model: {model_name} | Fold: {fold_idx}")

            # Load Model Architecture
            # pretrained=False because we are loading our own weights
            model = get_model(model_name, pretrained=False)

            # Load Weights
            state_dict = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()

            # Storage for this specific model's predictions
            current_model_ids = []
            current_model_probs = []

            with torch.no_grad():
                for images, ids in test_loader:
                    images = images.to(device)

                    # Get TTA predictions
                    probs = predict_batch_tta(model, images, device)

                    # Store results (move to CPU to save GPU memory)
                    current_model_probs.append(probs.cpu().numpy().flatten())
                    current_model_ids.append(ids.numpy().flatten())

            # Concatenate results for the full dataset
            full_probs = np.concatenate(current_model_probs)
            full_ids = np.concatenate(current_model_ids)

            # Create a Series for this model, indexed by ID
            # This ensures that even if dataloader order varied (unlikely), alignment is correct
            preds_series = pd.Series(full_probs, index=full_ids).sort_index()

            # Add to ensemble accumulator
            if ensemble_preds is None:
                ensemble_preds = preds_series
            else:
                ensemble_preds = ensemble_preds.add(preds_series, fill_value=0)

            model_count += 1

            # Cleanup to free VRAM
            del model
            torch.cuda.empty_cache()

    if model_count == 0:
        print("Error: No models were loaded. Cannot generate submission.")
        return

    # Compute Arithmetic Mean of Probabilities
    final_preds = ensemble_preds / model_count

    # Create Submission DataFrame
    submission_df = pd.DataFrame({"id": final_preds.index, "label": final_preds.values})

    # Ensure 'id' is integer
    submission_df["id"] = submission_df["id"].astype(int)

    # Sort by ID (standard submission requirement)
    submission_df = submission_df.sort_values("id")

    # Save to CSV
    os.makedirs(submission_dir, exist_ok=True)
    save_path = os.path.join(submission_dir, "submission.csv")
    submission_df.to_csv(save_path, index=False)

    print(f"Ensemble inference completed using {model_count} models.")
    print(f"Submission saved to {save_path}")
