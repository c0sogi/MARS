import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from library.config import Config
from library.dataset import AppleDataset, get_transforms
from library.model import AppleConvNeXt
from library.utils import seed_everything


def predict_with_tta(model, images):
    """
    Perform Test Time Augmentation (TTA) by averaging logits from:
    1. Original Image
    2. Horizontal Flip
    3. Vertical Flip

    Args:
        model: The PyTorch model.
        images: Batch of images (B, C, H, W).

    Returns:
        torch.Tensor: Averaged logits (B, Num_Classes).
    """
    # 1. Original
    logits_orig = model(images)

    # 2. Horizontal Flip (dim 3 is width)
    images_hflip = torch.flip(images, dims=[3])
    logits_hflip = model(images_hflip)

    # 3. Vertical Flip (dim 2 is height)
    images_vflip = torch.flip(images, dims=[2])
    logits_vflip = model(images_vflip)

    # Average logits
    avg_logits = (logits_orig + logits_hflip + logits_vflip) / 3.0
    return avg_logits


def run_inference(max_samples=None):
    """
    Main inference function.
    Loads the best model, performs inference on the test set with TTA,
    and saves the submission file.

    Args:
        max_samples (int, optional): Limit the number of samples for debugging.
    """
    seed_everything(Config.SEED)

    # Ensure submission directory exists
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Load Test Metadata
    if not os.path.exists(Config.TEST_CSV):
        raise FileNotFoundError(f"Test metadata not found at {Config.TEST_CSV}")

    df_test = pd.read_csv(Config.TEST_CSV)

    if max_samples is not None:
        df_test = df_test.iloc[:max_samples]

    # Initialize Dataset and DataLoader
    # Use 'valid' transforms for deterministic resizing and normalization
    test_dataset = AppleDataset(df_test, transforms=get_transforms(data="valid"))

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Initialize Model
    device = Config.DEVICE
    model = AppleConvNeXt(pretrained=False)

    # Load Checkpoint
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    all_image_ids = df_test["image"].tolist()
    all_labels = []

    print(f"Starting inference on {len(df_test)} images...")

    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)

            # Inference with TTA
            if Config.TTA_FLIP:
                logits = predict_with_tta(model, images)
            else:
                logits = model(images)

            # Apply Sigmoid
            probs = torch.sigmoid(logits)

            # Thresholding
            preds = (probs > Config.CONF_THRESHOLD).cpu().numpy()

            # Decode predictions
            for row_idx in range(preds.shape[0]):
                row_preds = preds[row_idx]
                # Get indices where prediction is 1
                label_indices = np.where(row_preds == 1)[0]

                if len(label_indices) > 0:
                    predicted_labels = [Config.CLASSES[idx] for idx in label_indices]
                    label_str = " ".join(predicted_labels)
                else:
                    # Fallback strategy:
                    # If no class exceeds threshold, default to 'healthy'
                    label_str = "healthy"

                all_labels.append(label_str)

    # Create Submission DataFrame
    submission_df = pd.DataFrame({"image": all_image_ids, "labels": all_labels})

    # Save to CSV
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved successfully to {Config.SUBMISSION_PATH}")
    print(submission_df.head())
