import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import torchvision.transforms as T

from library.config import Config, seed_everything, get_label_map
from library.models import get_detection_model
from library.utils import load_image


def load_models(device):
    """
    Loads the detection model.
    """
    # Get label mapping
    label2id, id2label = get_label_map(load_cached_data=True)
    num_classes = len(label2id) + 1  # +1 for background

    # Initialize model
    model = get_detection_model(num_classes=num_classes)

    # Define weight paths
    weights_path = os.path.join(Config.CACHE_DIR, "det_model.pth")

    # Load weights
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
    Runs inference on a single page.
    """
    # Preprocess
    # Faster R-CNN expects 0-1 float tensors
    img_tensor = T.functional.to_tensor(image).to(device)

    with torch.no_grad():
        # Input to model must be a list of tensors
        outputs = model([img_tensor])

    # Outputs is a list of dicts
    output = outputs[0]

    boxes = output["boxes"].cpu().numpy()
    labels = output["labels"].cpu().numpy()
    scores = output["scores"].cpu().numpy()

    predictions = []

    for box, label, score in zip(boxes, labels, scores):
        if score > Config.CONF_THRESHOLD:
            # Convert box to center point
            x1, y1, x2, y2 = box
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            if label in id2label:
                char_unicode = id2label[label]
                predictions.append(f"{char_unicode} {cx} {cy}")

    return predictions


def generate_submission(
    test_csv_path=Config.TEST_CSV, submission_path=Config.SUBMISSION_PATH
):
    """
    Generates the submission file for the test set.
    """
    seed_everything(Config.SEED)
    device = Config.DEVICE

    print("Starting submission generation...")

    if not os.path.exists(test_csv_path):
        raise FileNotFoundError(f"Test metadata not found at {test_csv_path}")

    test_df = pd.read_csv(test_csv_path)
    print(f"Found {len(test_df)} test images.")

    # Load Model
    model, id2label = load_models(device)

    results = []

    for idx, row in test_df.iterrows():
        image_id = row["image_id"]
        file_path = row["file_path"]

        try:
            image = load_image(file_path)
            preds_list = process_page(image, model, device, id2label)
            labels_str = " ".join(preds_list)
        except Exception as e:
            print(f"Error processing image {image_id}: {e}")
            labels_str = ""

        results.append({"image_id": image_id, "labels": labels_str})

        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1}/{len(test_df)} images.")

    submission_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(submission_path), exist_ok=True)
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved successfully to {submission_path}")
