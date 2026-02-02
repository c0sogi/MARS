import os
import cv2
import torch
import numpy as np
import pandas as pd
import torchvision.transforms as T

from library.config import Config, seed_everything, get_label_map
from library.models import SegmentationUNet, CharacterClassifier
from library.utils import load_image

# Define normalization transform matching training configuration
NORMALIZE = T.Compose(
    [
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def load_models(device):
    """
    Loads the segmentation and classification models with trained weights.

    Args:
        device (torch.device): The device to load models onto.

    Returns:
        tuple: (seg_model, cls_model, id2label)
    """
    # Get label mapping
    label2id, id2label = get_label_map(load_cached_data=True)
    num_classes = len(label2id)

    # Initialize models
    seg_model = SegmentationUNet(n_classes=1)
    cls_model = CharacterClassifier(num_classes=num_classes)

    # Define weight paths
    seg_weights_path = os.path.join(Config.CACHE_DIR, "seg_model.pth")
    cls_weights_path = os.path.join(Config.CACHE_DIR, "cls_model.pth")

    # Load weights if available
    if os.path.exists(seg_weights_path):
        seg_model.load_state_dict(torch.load(seg_weights_path, map_location=device))
        print(f"Loaded segmentation weights from {seg_weights_path}")
    else:
        print(
            f"Warning: Segmentation weights not found at {seg_weights_path}. Using random initialization."
        )

    if os.path.exists(cls_weights_path):
        cls_model.load_state_dict(torch.load(cls_weights_path, map_location=device))
        print(f"Loaded classification weights from {cls_weights_path}")
    else:
        print(
            f"Warning: Classification weights not found at {cls_weights_path}. Using random initialization."
        )

    # Move to device and set to eval mode
    seg_model.to(device)
    cls_model.to(device)
    seg_model.eval()
    cls_model.eval()

    return seg_model, cls_model, id2label


def process_page(image, seg_model, cls_model, device, id2label):
    """
    Runs the inference pipeline on a single page image.

    Args:
        image (np.ndarray): RGB image.
        seg_model (nn.Module): Segmentation model.
        cls_model (nn.Module): Classification model.
        device (torch.device): Compute device.
        id2label (dict): Mapping from class ID to Unicode label.

    Returns:
        list: List of prediction strings "Label X Y".
    """
    orig_h, orig_w = image.shape[:2]

    # ---------------------------------------------------------
    # 1. Segmentation (Localization)
    # ---------------------------------------------------------
    # Resize for segmentation model input
    seg_input = cv2.resize(image, (Config.SEG_IMG_SIZE[1], Config.SEG_IMG_SIZE[0]))

    # Preprocess
    seg_tensor = NORMALIZE(seg_input).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        seg_logits = seg_model(seg_tensor)
        seg_probs = torch.sigmoid(seg_logits)

    # Post-process mask
    # Remove batch and channel dims -> (H, W)
    mask_np = seg_probs.squeeze().cpu().numpy()

    # Resize mask back to original image dimensions
    mask_resized = cv2.resize(mask_np, (orig_w, orig_h))

    # Thresholding to create binary mask
    binary_mask = (mask_resized > Config.CONF_THRESHOLD).astype(np.uint8)

    # ---------------------------------------------------------
    # 2. Component Analysis (Extraction)
    # ---------------------------------------------------------
    # Connected Components Analysis
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
        binary_mask, connectivity=8
    )

    crops = []
    coords = []  # List of (center_x, center_y)

    # Iterate over components (skip background at index 0)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]

        # Filter noise
        if area < Config.MIN_AREA_THRESHOLD:
            continue

        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]

        # Calculate center point for submission
        center_x = int(x + w / 2)
        center_y = int(y + h / 2)

        # Extract crop from original image
        # Clamp coordinates to image boundaries
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(orig_w, x + w)
        y2 = min(orig_h, y + h)

        if x2 <= x1 or y2 <= y1:
            continue

        crop = image[y1:y2, x1:x2]

        # Resize for classification model
        crop_resized = cv2.resize(
            crop, (Config.CLS_CROP_SIZE[1], Config.CLS_CROP_SIZE[0])
        )

        crops.append(crop_resized)
        coords.append((center_x, center_y))

    # Limit predictions per page to meet competition requirements
    if len(crops) > 1200:
        crops = crops[:1200]
        coords = coords[:1200]

    predictions = []

    # ---------------------------------------------------------
    # 3. Classification (Recognition)
    # ---------------------------------------------------------
    if crops:
        # Prepare batch
        crop_tensors = [NORMALIZE(c) for c in crops]
        crop_batch = torch.stack(crop_tensors).to(device)

        batch_size = Config.CLS_BATCH_SIZE
        num_crops = len(crops)
        pred_indices = []

        # Run classification in batches
        with torch.no_grad():
            for i in range(0, num_crops, batch_size):
                batch = crop_batch[i : i + batch_size]
                outputs = cls_model(batch)
                _, preds = torch.max(outputs, 1)
                pred_indices.extend(preds.cpu().numpy())

        # Format predictions
        for (cx, cy), label_idx in zip(coords, pred_indices):
            if label_idx in id2label:
                char_unicode = id2label[label_idx]
                predictions.append(f"{char_unicode} {cx} {cy}")

    return predictions


def generate_submission(
    test_csv_path=Config.TEST_CSV, submission_path=Config.SUBMISSION_PATH
):
    """
    Generates the submission file for the test set.

    Args:
        test_csv_path (str): Path to the test metadata CSV.
        submission_path (str): Path to save the submission CSV.
    """
    seed_everything(Config.SEED)
    device = Config.DEVICE

    print("Starting submission generation...")

    # Load Metadata
    if not os.path.exists(test_csv_path):
        raise FileNotFoundError(f"Test metadata not found at {test_csv_path}")

    test_df = pd.read_csv(test_csv_path)
    print(f"Found {len(test_df)} test images.")

    # Load Models
    seg_model, cls_model, id2label = load_models(device)

    results = []

    for idx, row in test_df.iterrows():
        image_id = row["image_id"]
        file_path = row["file_path"]

        try:
            # Load Image
            image = load_image(file_path)

            # Run Pipeline
            preds_list = process_page(image, seg_model, cls_model, device, id2label)

            # Join into space-separated string
            labels_str = " ".join(preds_list)

        except Exception as e:
            print(f"Error processing image {image_id}: {e}")
            labels_str = ""

        results.append({"image_id": image_id, "labels": labels_str})

        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1}/{len(test_df)} images.")

    # Create DataFrame
    submission_df = pd.DataFrame(results)

    # Save
    os.makedirs(os.path.dirname(submission_path), exist_ok=True)
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved successfully to {submission_path}")
