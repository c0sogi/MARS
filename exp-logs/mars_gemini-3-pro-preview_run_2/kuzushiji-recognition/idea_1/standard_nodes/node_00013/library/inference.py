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
    num_classes = len(label2id) + 1

    # Initialize model
    model = get_detection_model(num_classes)

    # Load weights
    weights_path = os.path.join(Config.CACHE_DIR, "det_model.pth")
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"Loaded detection weights from {weights_path}")
    else:
        print(f"Warning: Weights not found at {weights_path}")

    model.to(device)
    model.eval()

    return model, id2label


def process_page(image, model, device, id2label):
    """
    Runs inference on a single page.
    """
    # Prepare input
    # Convert to tensor (0-1)
    img_tensor = T.functional.to_tensor(image).to(device)

    with torch.no_grad():
        # Input to model must be a list of tensors
        predictions = model([img_tensor])[0]

    # Process predictions
    boxes = predictions["boxes"].cpu().numpy()
    labels = predictions["labels"].cpu().numpy()
    scores = predictions["scores"].cpu().numpy()

    results = []

    for box, label_id, score in zip(boxes, labels, scores):
        if score < Config.CONF_THRESHOLD:
            continue

        # Map label ID back (subtract 1)
        original_id = label_id - 1
        if original_id in id2label:
            char_unicode = id2label[original_id]

            # Calculate Center
            x1, y1, x2, y2 = box
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            results.append(f"{char_unicode} {cx} {cy}")

    # Limit predictions
    if len(results) > 1200:
        results = results[:1200]

    return results


def generate_submission(
    test_csv_path=Config.TEST_CSV, submission_path=Config.SUBMISSION_PATH
):
    seed_everything(Config.SEED)
    device = Config.DEVICE

    print("Starting submission generation...")
    test_df = pd.read_csv(test_csv_path)

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
            print(f"Error: {e}")
            labels_str = ""

        results.append({"image_id": image_id, "labels": labels_str})

        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1}/{len(test_df)} images.")

    submission_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(submission_path), exist_ok=True)
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
