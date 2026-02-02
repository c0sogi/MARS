import os
import random
import numpy as np
import pandas as pd
import torch

from library.config import Config
from library.data import get_dataloader
from library.model import AsymmetricEfficientNet


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def predict_submission(
    model_path=Config.MODEL_SAVE_PATH,
    output_path=Config.SUBMISSION_FILE,
    batch_size=Config.BATCH_SIZE,
    device=Config.DEVICE,
):
    """
    Generates predictions for the test set using the trained model and saves them to a CSV file.
    Implements Test-Time Augmentation (TTA) by averaging predictions from:
    1. Original Image
    2. Horizontal Flip
    3. Vertical Flip

    Args:
        model_path (str): Path to the trained model weights.
        output_path (str): Path where the submission CSV will be saved.
        batch_size (int): Batch size for inference.
        device (str): Computation device ('cpu' or 'cuda').
    """
    # 1. Setup
    set_seed(Config.SEED)
    print(f"Running inference on device: {device}")

    # 2. Load Test Data
    # We rely on the metadata generated previously
    if not os.path.exists(Config.TEST_METADATA):
        raise FileNotFoundError(f"Test metadata not found at {Config.TEST_METADATA}")

    test_df = pd.read_csv(Config.TEST_METADATA)

    # Use the library's dataloader factory
    # MRIDataset handles the ROI selection and stacking internally
    test_loader = get_dataloader(
        test_df, phase="test", batch_size=batch_size, shuffle=False
    )

    # 3. Load Model
    # Initialize architecture
    model = AsymmetricEfficientNet(num_classes=1)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found at {model_path}")

    # Load weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()

    # 4. Inference Loop with TTA
    results = []
    print(f"Generating predictions for {len(test_df)} subjects using TTA...")

    with torch.no_grad():
        for images, subject_ids in test_loader:
            images = images.to(device)  # Shape: (B, 12, H, W)

            # --- TTA 1: Original ---
            logits_1 = model(images)
            probs_1 = torch.sigmoid(logits_1)

            # --- TTA 2: Horizontal Flip ---
            # Flip along width dimension (dim 3)
            images_h = torch.flip(images, [3])
            logits_2 = model(images_h)
            probs_2 = torch.sigmoid(logits_2)

            # --- TTA 3: Vertical Flip ---
            # Flip along height dimension (dim 2)
            images_v = torch.flip(images, [2])
            logits_3 = model(images_v)
            probs_3 = torch.sigmoid(logits_3)

            # --- Average Predictions ---
            avg_probs = (probs_1 + probs_2 + probs_3) / 3.0
            avg_probs_np = avg_probs.cpu().numpy().flatten()

            # Store results
            for sub_id, prob in zip(subject_ids, avg_probs_np):
                results.append({"BraTS21ID": sub_id, "MGMT_value": prob})

    # 5. Save Submission
    submission_df = pd.DataFrame(results)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved successfully to {output_path}")
