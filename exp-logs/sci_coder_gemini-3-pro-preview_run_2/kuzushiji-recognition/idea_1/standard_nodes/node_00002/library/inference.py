import os
import cv2
import torch
import torch.nn.functional as F
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
    # 1. Segmentation (Localization via Heatmap)
    # ---------------------------------------------------------
    # Resize for segmentation model input
    seg_input = cv2.resize(image, (Config.SEG_IMG_SIZE[1], Config.SEG_IMG_SIZE[0]))

    # Preprocess
    seg_tensor = NORMALIZE(seg_input).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        seg_logits = seg_model(seg_tensor)
        seg_probs = torch.sigmoid(seg_logits)

    # ---------------------------------------------------------
    # 2. Peak Detection (Extraction)
    # Cite solution_lesson_node_00001: Finding local maxima instead of connected components
    # ---------------------------------------------------------
    # Perform Max Pooling to find local maxima
    kernel_size = 5
    padding = kernel_size // 2
    hmax = F.max_pool2d(seg_probs, kernel_size=kernel_size, stride=1, padding=padding)

    # Peaks are where value equals max_pool value and is above threshold
    # Note: seg_probs is (1, 1, H, W)
    peaks = (hmax == seg_probs) & (seg_probs > Config.CONF_THRESHOLD)

    # Get coordinates
    # peaks is boolean tensor. nonzero returns indices [batch, channel, y, x]
    peak_coords = peaks.nonzero()

    crops = []
    coords = []

    # Scale factors
    scale_x = orig_w / Config.SEG_IMG_SIZE[1]
    scale_y = orig_h / Config.SEG_IMG_SIZE[0]

    crop_h, crop_w = Config.CLS_CROP_SIZE

    # Iterate over peaks
    for p in peak_coords:
        # p is [0, 0, y, x]
        py, px = p[2].item(), p[3].item()

        # Map to original image
        center_x = int(px * scale_x)
        center_y = int(py * scale_y)

        # Define crop box centered at (center_x, center_y)
        # Using a fixed size crop for classification input
        # We use a slightly larger context than the target size to allow resizing
        # But here we just take a region and resize it to CLS_CROP_SIZE

        # Let's take a crop that roughly matches the average character size + context
        # Avg char is ~76x92. Let's take 128x128 context
        context_size = 128
        half_size = context_size // 2

        x1 = max(0, center_x - half_size)
        y1 = max(0, center_y - half_size)
        x2 = min(orig_w, center_x + half_size)
        y2 = min(orig_h, center_y + half_size)

        if x2 <= x1 or y2 <= y1:
            continue

        crop = image[y1:y2, x1:x2]

        # Resize for classification model
        crop_resized = cv2.resize(crop, (crop_w, crop_h))

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
