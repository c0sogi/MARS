import os
import torch
import pandas as pd
import numpy as np

from library.config import Config
from library.utils import seed_everything
from library.data import get_test_loader
from library.model import AppleEfficientNet


def predict_tta(model, loader, device):
    """
    Performs Test Time Augmentation (TTA) inference.
    Averages predictions from original image and horizontally flipped image.

    Args:
        model (nn.Module): The trained PyTorch model.
        loader (DataLoader): DataLoader for test data.
        device (torch.device): Computation device.

    Returns:
        tuple: (list of image_ids, np.ndarray of probabilities)
    """
    model.eval()
    all_probs = []
    all_image_ids = []

    with torch.no_grad():
        for images, image_ids in loader:
            images = images.to(device)

            # 1. Forward pass: Original images
            logits = model(images)
            probs = torch.softmax(logits, dim=1)

            # 2. Forward pass: Horizontally flipped images
            # Input shape is (Batch, Channels, Height, Width). Flip on Width (dim 3).
            images_flipped = torch.flip(images, dims=[3])
            logits_flipped = model(images_flipped)
            probs_flipped = torch.softmax(logits_flipped, dim=1)

            # 3. Average predictions
            avg_probs = (probs + probs_flipped) / 2.0

            all_probs.append(avg_probs.cpu().numpy())
            all_image_ids.extend(image_ids)

    final_probs = np.concatenate(all_probs, axis=0)
    return all_image_ids, final_probs


def run_inference(load_cached_data=True):
    """
    Main inference function.
    Loads model, performs TTA on test set, and saves submission.

    Args:
        load_cached_data (bool): Whether to use cached dataframes for the data loader.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Ensure submission directory exists
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # 2. Load Data
    # get_test_loader handles metadata loading and dataset creation
    test_loader = get_test_loader(load_cached_data=load_cached_data)

    # 3. Load Model
    # Initialize model structure
    # We set pretrained=False because we are about to load our own trained weights
    model = AppleEfficientNet(
        model_name=Config.MODEL_NAME,
        pretrained=False,
        num_classes=Config.NUM_CLASSES,
        dropout_rate=Config.DROPOUT_RATE,
    )

    # Load trained weights
    if not os.path.exists(Config.BEST_MODEL_PATH):
        raise FileNotFoundError(
            f"Model weights not found at {Config.BEST_MODEL_PATH}. "
            "Please ensure the model has been trained and saved."
        )

    print(f"Loading model weights from {Config.BEST_MODEL_PATH}...")
    state_dict = torch.load(Config.BEST_MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)

    # 4. Predict with TTA
    print(
        f"Starting inference on device: {device} using Test Time Augmentation (Horizontal Flip)..."
    )
    image_ids, probs = predict_tta(model, test_loader, device)

    # 5. Create Submission DataFrame
    # Config.CLASS_LABELS order: ["healthy", "multiple_diseases", "rust", "scab"]
    # This matches the model output order and the required submission format.
    submission_df = pd.DataFrame(probs, columns=Config.CLASS_LABELS)
    submission_df.insert(0, "image_id", image_ids)

    # 6. Save Submission
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)

    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(f"Submission shape: {submission_df.shape}")
    print("First 5 rows of submission:")
    print(submission_df.head().to_string())
