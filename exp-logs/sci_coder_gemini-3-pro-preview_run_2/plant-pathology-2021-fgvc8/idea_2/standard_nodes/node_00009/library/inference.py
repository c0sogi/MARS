import os
import torch
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

from library.config import Config
from library.utils import seed_everything
from library.dataset import get_loaders
from library.model import AppleDiseaseModel


def predict_fn(checkpoint_path: str = None, limit_batches: int = None):
    """
    Generates predictions for the test set using the trained model.
    Applies Test Time Augmentation (TTA) by averaging predictions on original
    and horizontally flipped images.

    Args:
        checkpoint_path (str, optional): Path to the model checkpoint.
                                         Defaults to 'best_model.pth' in the working directory.
        limit_batches (int, optional): Limit the number of batches for debugging purposes.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    if checkpoint_path is None:
        checkpoint_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    print(f"Initializing inference on device: {device}")

    # 2. Load Model
    # We set pretrained=False because we are about to load our own fine-tuned weights
    model = AppleDiseaseModel(
        model_name=Config.MODEL_NAME, pretrained=False, num_classes=Config.NUM_CLASSES
    )

    if os.path.exists(checkpoint_path):
        print(f"Loading weights from {checkpoint_path}...")
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(
            f"WARNING: Checkpoint {checkpoint_path} not found. Using random initialization."
        )

    model.to(device)
    model.eval()

    # 3. Get Data Loader
    # get_loaders returns (train, val, test). We only need test.
    _, _, test_loader = get_loaders()

    # 4. Inference Loop
    image_ids = []
    predictions = []

    print(f"Starting inference (TTA={'Enabled' if Config.USE_TTA else 'Disabled'})...")

    with torch.no_grad():
        for i, (images, ids) in enumerate(tqdm(test_loader, desc="Predicting")):
            if limit_batches is not None and i >= limit_batches:
                break

            images = images.to(device, dtype=torch.float)

            # Forward pass 1: Original
            logits_orig = model(images)
            probs_orig = torch.sigmoid(logits_orig)

            if Config.USE_TTA:
                # Forward pass 2: Horizontal Flip
                # Flip along width dimension (dim 3 for NCHW tensor)
                images_flipped = torch.flip(images, dims=[3])
                logits_flip = model(images_flipped)
                probs_flip = torch.sigmoid(logits_flip)

                # Average probabilities
                probs = (probs_orig + probs_flip) / 2.0
            else:
                probs = probs_orig

            predictions.append(probs.cpu().numpy())
            image_ids.extend(ids)

    # 5. Process Predictions
    if predictions:
        predictions = np.concatenate(predictions, axis=0)

        # Apply threshold to get binary matrix
        binary_preds = (predictions > Config.THRESHOLD).astype(int)

        final_labels = []
        classes = np.array(Config.CLASSES)

        for row in binary_preds:
            # Get indices of positive classes
            indices = np.where(row == 1)[0]

            if len(indices) > 0:
                # Map indices to class names and join with space
                labels_list = classes[indices]
                final_labels.append(" ".join(labels_list))
            else:
                # Fallback: if no class exceeds threshold, default to 'healthy'
                # This is a common heuristic for this specific dataset structure
                final_labels.append("healthy")

        # 6. Generate Submission
        submission_df = pd.DataFrame({"image": image_ids, "labels": final_labels})

        # Ensure output directory exists
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

        # Save to CSV
        save_path = Config.SUBMISSION_PATH
        submission_df.to_csv(save_path, index=False)

        print(f"Inference complete. Submission saved to {save_path}")
        print("Sample predictions:")
        print(submission_df.head())

    else:
        print("No predictions generated (dataset might be empty).")
