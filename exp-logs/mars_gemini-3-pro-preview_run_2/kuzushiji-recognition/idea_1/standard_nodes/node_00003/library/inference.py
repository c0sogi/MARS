import os
import cv2
import torch
import numpy as np
import pandas as pd
import torchvision.transforms as T

from library.config import Config, seed_everything, get_label_map
from library.models import get_detection_model
from library.utils import load_image


def load_models(device):
    """
    Loads the detection model with trained weights.
    """
    # Get label mapping
    label2id, id2label = get_label_map(load_cached_data=True)
    # +1 for background
    num_classes = len(label2id) + 1

    # Initialize model
    model = get_detection_model(num_classes)

    # Define weight paths
    weights_path = os.path.join(Config.CACHE_DIR, "det_model.pth")

    # Load weights if available
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"Loaded detection weights from {weights_path}")
    else:
        print(
            f"Warning: Weights not found at {weights_path}. Using random initialization."
        )

    # Move to device and set to eval mode
    model.to(device)
    model.eval()

    return model, id2label


def process_page(image, model, device, id2label):
    """
    Runs the detection inference on a single page image.
    """
    # Preprocess
    # Convert to tensor [0, 1]
    img_tensor = T.ToTensor()(image).to(device)

    # Inference
    with torch.no_grad():
        # Input must be a list of tensors
        predictions = model([img_tensor])

    # Process outputs
    pred = predictions[0]
    boxes = pred["boxes"].cpu().numpy()
    labels = pred["labels"].cpu().numpy()
    scores = pred["scores"].cpu().numpy()

    results = []

    # Filter by score
    valid_indices = scores > Config.DET_SCORE_THRESHOLD

    valid_boxes = boxes[valid_indices]
    valid_labels = labels[valid_indices]
    valid_scores = scores[valid_indices]

    # Sort by score descending
    sorted_idx = np.argsort(valid_scores)[::-1]

    # Limit to 1200 predictions
    if len(sorted_idx) > 1200:
        sorted_idx = sorted_idx[:1200]

    for idx in sorted_idx:
        box = valid_boxes[idx]
        label_id = valid_labels[idx]

        # Calculate center
        x1, y1, x2, y2 = box
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        # Map label ID back to unicode
        # Note: model label 0 is background, so our classes start at 1.
        # id2label is 0-indexed for the characters.
        # So char_id = model_label - 1
        char_id = label_id - 1

        if char_id in id2label:
            char_unicode = id2label[char_id]
            results.append(f"{char_unicode} {cx} {cy}")

    return results


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

    # Also save to runfile location required by competition (demo_submission.csv is in working/ but instructions say 'designated location')
    # The prompt says "as long as your best submission is stored at the designated location at the end of your run"
    # Usually this implies submission.csv in working directory.
    # We will copy it to ./working/submission.csv just in case.
    try:
        submission_df.to_csv("./working/submission.csv", index=False)
    except:
        pass
