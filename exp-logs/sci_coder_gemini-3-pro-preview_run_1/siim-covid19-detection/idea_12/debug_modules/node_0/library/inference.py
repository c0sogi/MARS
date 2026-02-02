import os
import torch
import numpy as np
import pandas as pd
from library.config import Config
from library.dataset import get_loaders
from library.model import ResNet18D_UNet
from library.utils import mask2bbox, seed_everything


def predict_and_submit(model_path=None):
    """
    Runs inference on the test dataset using the trained ResNet18-D U-Net model.
    Applies Test-Time Augmentation (TTA) and Gated Prediction logic.
    Generates the submission.csv file.

    Args:
        model_path (str, optional): Path to the trained model weights.
                                    Defaults to None (will attempt to load from working dir if needed).
    """
    # 1. Setup
    seed_everything(Config.seed)
    device = Config.device

    print(f"Starting inference on device: {device}")

    # 2. Data Loading
    # We only need the test loader. get_loaders handles caching and preprocessing.
    print("Loading test data...")
    _, _, test_loader = get_loaders(debug=Config.debug, load_cached_data=True)

    # 3. Model Initialization
    print("Initializing model...")
    # pretrained=False because we are loading custom weights
    model = ResNet18D_UNet(num_classes=Config.num_study_classes, pretrained=False)

    if model_path is None:
        # Default fallback to best_model.pth in the current working directory or idea folder
        possible_paths = ["./working/idea_12/best_model.pth", "best_model.pth"]
        for p in possible_paths:
            if os.path.exists(p):
                model_path = p
                break

    if model_path and os.path.exists(model_path):
        print(f"Loading weights from {model_path}...")
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(
            f"WARNING: Model path '{model_path}' not found. Using random initialization."
        )

    model.to(device)
    model.eval()

    # 4. Inference Loop
    study_probs_map = {}  # study_id -> list of probability vectors
    image_preds_map = {}  # image_id -> prediction string

    # Mapping for study labels
    # 0: Negative, 1: Typical, 2: Indeterminate, 3: Atypical
    class_name_map = {0: "negative", 1: "typical", 2: "indeterminate", 3: "atypical"}

    print("Running prediction loop with TTA...")

    with torch.no_grad():
        for i, (images, image_ids, study_ids) in enumerate(test_loader):
            images = images.to(device)

            # --- TTA: Horizontal Flip ---
            # 1. Original Forward
            logits, pred_masks = model(images)

            # 2. Flipped Forward
            images_flip = torch.flip(images, dims=[3])
            logits_flip, pred_masks_flip = model(images_flip)

            # 3. Un-flip and Average
            pred_masks_flip = torch.flip(pred_masks_flip, dims=[3])

            logits_avg = (logits + logits_flip) / 2.0
            pred_masks_avg = (pred_masks + pred_masks_flip) / 2.0

            # --- Post-Processing ---

            # Study Probabilities (Softmax)
            probs = torch.softmax(logits_avg, dim=1).cpu().numpy()

            # Mask Probabilities (Sigmoid)
            mask_probs = torch.sigmoid(pred_masks_avg).cpu().numpy()

            batch_size = images.size(0)

            for b in range(batch_size):
                img_id = image_ids[b]
                std_id = study_ids[b]

                # Store study probabilities
                if std_id not in study_probs_map:
                    study_probs_map[std_id] = []
                study_probs_map[std_id].append(probs[b])

                # --- Gated Prediction Logic ---
                # Determine dominant study class
                pred_class_idx = np.argmax(probs[b])

                image_pred_str = ""

                # If predicted "Negative for Pneumonia" (Class 0), force no finding
                if pred_class_idx == 0:
                    image_pred_str = "none 1 0 0 1 1"
                else:
                    # Extract bounding boxes
                    # Threshold mask
                    mask_binary = (mask_probs[b, 0] > 0.5).astype(np.uint8)
                    bboxes = mask2bbox(mask_binary)

                    if not bboxes:
                        image_pred_str = "none 1 0 0 1 1"
                    else:
                        box_strings = []
                        for bbox in bboxes:
                            x1, y1, x2, y2 = map(int, bbox)

                            # Clip to image bounds
                            x1 = max(0, x1)
                            y1 = max(0, y1)
                            x2 = min(Config.img_size, x2)
                            y2 = min(Config.img_size, y2)

                            if x2 > x1 and y2 > y1:
                                # Confidence is mean probability inside the box
                                conf = mask_probs[b, 0, y1:y2, x1:x2].mean()
                                # Format: opacity confidence xmin ymin xmax ymax
                                box_strings.append(
                                    f"opacity {conf:.4f} {x1} {y1} {x2} {y2}"
                                )

                        if box_strings:
                            image_pred_str = " ".join(box_strings)
                        else:
                            image_pred_str = "none 1 0 0 1 1"

                image_preds_map[img_id] = image_pred_str

    # 5. Submission Formatting
    print("Formatting submission rows...")
    submission_rows = []

    # Process Study Level
    for std_id, prob_list in study_probs_map.items():
        # Average probabilities if multiple images per study
        avg_probs = np.mean(prob_list, axis=0)

        # Create string with all classes and probabilities
        # Format: "negative {conf} 0 0 1 1 typical {conf} 0 0 1 1 ..."
        parts = []
        for cls_idx in range(4):
            cls_name = class_name_map[cls_idx]
            conf = avg_probs[cls_idx]
            parts.append(f"{cls_name} {conf:.6f} 0 0 1 1")

        pred_string = " ".join(parts)
        submission_rows.append(
            {"Id": f"{std_id}_study", "PredictionString": pred_string}
        )

    # Process Image Level
    for img_id, pred_string in image_preds_map.items():
        submission_rows.append(
            {"Id": f"{img_id}_image", "PredictionString": pred_string}
        )

    # 6. Save to CSV
    df_sub = pd.DataFrame(submission_rows)
    # Sort for consistency
    df_sub = df_sub.sort_values("Id").reset_index(drop=True)

    os.makedirs(Config.submission_dir, exist_ok=True)
    df_sub.to_csv(Config.submission_path, index=False)

    print(f"Submission saved to {Config.submission_path}")
    print(f"Total rows: {len(df_sub)}")
