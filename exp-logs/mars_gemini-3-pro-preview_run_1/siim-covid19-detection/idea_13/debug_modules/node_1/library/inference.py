import os
import torch
import pandas as pd
import numpy as np
import pydicom
from torch.utils.data import DataLoader
from library.config import Config
from library.dataset import SIIMDataset, get_transforms
from library.model import ResNet18D_UNet
from library.utils import mask2bbox


def get_test_dimensions(df, load_cached_data=True):
    """
    Retrieves original image dimensions for the test set to allow accurate
    bounding box scaling. Implements caching using Parquet.
    """
    cache_path = os.path.join(Config.WORKING_DIR, "test_dims.parquet")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            dims_df = pd.read_parquet(cache_path)
            # Convert to dict for O(1) lookup: {image_id: {'height': h, 'width': w}}
            return dims_df.set_index("image_id")[["height", "width"]].to_dict("index")
        except Exception:
            pass

    # 2. Compute from scratch if cache missing or failed
    records = []
    for _, row in df.iterrows():
        file_path = os.path.join(Config.INPUT_DIR, row["file_path"])
        try:
            # Read only the header to be fast (stop_before_pixels=True)
            dcm = pydicom.dcmread(file_path, stop_before_pixels=True)
            records.append(
                {"image_id": row["image_id"], "height": dcm.Rows, "width": dcm.Columns}
            )
        except Exception:
            # Fallback for corrupt files (unlikely in test, but safe)
            records.append(
                {
                    "image_id": row["image_id"],
                    "height": Config.IMG_SIZE,
                    "width": Config.IMG_SIZE,
                }
            )

    dims_df = pd.DataFrame(records)

    # 3. Save to cache
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    dims_df.to_parquet(cache_path)

    return dims_df.set_index("image_id")[["height", "width"]].to_dict("index")


def format_study_prediction(probs):
    """
    Formats the study-level prediction string with all class probabilities.
    Classes: 0:Negative, 1:Typical, 2:Indeterminate, 3:Atypical
    """
    classes = ["negative", "typical", "indeterminate", "atypical"]
    pred_strings = []
    for i, prob in enumerate(probs):
        # Format: class_name confidence 0 0 1 1
        pred_strings.append(f"{classes[i]} {prob:.6f} 0 0 1 1")
    return " ".join(pred_strings)


def predict_and_submit(load_cached_data=True):
    """
    Performs inference on the test set, applies TTA and Gating,
    and generates the submission file.
    """
    device = torch.device(Config.DEVICE)
    print(f"Starting inference on device: {device}")

    # 1. Load Data
    test_dataset = SIIMDataset(
        "test", load_cached_data=load_cached_data, transform=get_transforms("test")
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Get original dimensions for scaling boxes
    dims_map = get_test_dimensions(test_dataset.df, load_cached_data=load_cached_data)

    # 2. Load Model
    model = ResNet18D_UNet(num_classes=Config.NUM_CLASSES, pretrained=False)

    if os.path.exists(Config.BEST_MODEL_PATH):
        state_dict = torch.load(Config.BEST_MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
        print("Loaded best model checkpoint.")
    else:
        print(
            f"Warning: Model checkpoint not found at {Config.BEST_MODEL_PATH}. Using random weights."
        )

    model.to(device)
    model.eval()

    results = []

    # 3. Inference Loop
    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            batch_study_ids = batch["study_id"]
            batch_image_ids = batch["image_id"]

            # --- TTA: Horizontal Flip ---
            # Forward Pass 1: Original
            cls_logits_1, seg_logits_1 = model(images)

            # Forward Pass 2: Flipped
            images_flipped = torch.flip(images, dims=[3])
            cls_logits_2, seg_logits_2 = model(images_flipped)

            # Combine Predictions
            # Classification: Average probabilities (Softmax)
            cls_probs = (
                torch.softmax(cls_logits_1, dim=1) + torch.softmax(cls_logits_2, dim=1)
            ) / 2.0

            # Segmentation: Average probabilities (Sigmoid)
            seg_probs_1 = torch.sigmoid(seg_logits_1)
            seg_probs_2 = torch.sigmoid(seg_logits_2)
            seg_probs_2 = torch.flip(
                seg_probs_2, dims=[3]
            )  # Flip back to original orientation
            seg_probs = (seg_probs_1 + seg_probs_2) / 2.0

            # Move to CPU for processing
            cls_probs = cls_probs.cpu().numpy()
            seg_probs = seg_probs.cpu().numpy()

            # Process Batch
            for i in range(len(images)):
                study_id = batch_study_ids[i]
                image_id = batch_image_ids[i]

                # --- Study Level Prediction ---
                # We output probabilities for ALL classes to maximize mAP
                study_pred_str = format_study_prediction(cls_probs[i])
                results.append(
                    {"Id": f"{study_id}_study", "PredictionString": study_pred_str}
                )

                # --- Image Level Prediction ---
                # Strategy: Logical Gating
                # If the strongest study prediction is "Negative" (Class 0),
                # we force the image prediction to "none".
                pred_class_idx = np.argmax(cls_probs[i])

                if pred_class_idx == 0:
                    image_pred_str = "none 1 0 0 1 1"
                else:
                    # Get original dimensions
                    if image_id in dims_map:
                        orig_h = dims_map[image_id]["height"]
                        orig_w = dims_map[image_id]["width"]
                    else:
                        orig_h, orig_w = Config.IMG_SIZE, Config.IMG_SIZE

                    scale_x = orig_w / Config.IMG_SIZE
                    scale_y = orig_h / Config.IMG_SIZE

                    # Extract boxes from mask
                    boxes = mask2bbox(seg_probs[i], threshold=0.5)

                    if not boxes:
                        image_pred_str = "none 1 0 0 1 1"
                    else:
                        box_strs = []
                        for box in boxes:
                            # mask2bbox returns [xmin, ymin, xmax, ymax]
                            x1, y1, x2, y2 = box

                            # Scale to original image size
                            x1_s = x1 * scale_x
                            y1_s = y1 * scale_y
                            x2_s = x2 * scale_x
                            y2_s = y2 * scale_y

                            # Calculate confidence score (mean probability within the box)
                            # We slice the 512x512 mask using unscaled coordinates
                            # Clamp coordinates to be safe
                            mx1, my1 = max(0, x1), max(0, y1)
                            mx2, my2 = min(Config.IMG_SIZE, x2), min(
                                Config.IMG_SIZE, y2
                            )

                            mask_slice = seg_probs[i, 0, my1:my2, mx1:mx2]
                            if mask_slice.size > 0:
                                conf = np.mean(mask_slice)
                            else:
                                conf = 0.0

                            # Format: opacity confidence xmin ymin xmax ymax
                            box_strs.append(
                                f"opacity {conf:.6f} {x1_s:.6f} {y1_s:.6f} {x2_s:.6f} {y2_s:.6f}"
                            )

                        image_pred_str = " ".join(box_strs)

                results.append(
                    {"Id": f"{image_id}_image", "PredictionString": image_pred_str}
                )

    # 4. Save Submission
    submission_df = pd.DataFrame(results)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
