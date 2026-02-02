import os
import torch
import pandas as pd
import numpy as np
import pydicom
from torch.utils.data import DataLoader

from library.config import Config
from library.dataset import prepare_data, SIIMDataset, get_transforms
from library.model import ResNet18UNetMultiScale
from library.utils import mask2bbox


def inference(debug=False):
    """
    Runs inference on the test set with TTA, Gated Logic, and generates the submission file.

    Args:
        debug (bool): If True, runs on a subset of data for debugging.
    """
    # 1. Setup
    device = torch.device(Config.DEVICE)
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    # Mapping for submission class names (Full -> Short)
    class_map = {
        "Negative for Pneumonia": "negative",
        "Typical Appearance": "typical",
        "Indeterminate Appearance": "indeterminate",
        "Atypical Appearance": "atypical",
    }

    print("Loading test metadata and mapping dimensions...")
    df_test = pd.read_csv(Config.TEST_CSV)

    # Create maps:
    #   image_id -> study_id
    #   image_id -> (original_width, original_height)
    image_to_study = {}
    image_size_map = {}

    # We need to read original dimensions to rescale boxes later
    # This is fast if we only read headers
    for idx, row in df_test.iterrows():
        img_id = row["image_id"]
        study_id = row["study_id"]
        file_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        image_to_study[img_id] = study_id

        try:
            dcm = pydicom.dcmread(file_path, stop_before_pixels=True)
            image_size_map[img_id] = (dcm.Columns, dcm.Rows)  # (Width, Height)
        except Exception as e:
            # Fallback if read fails
            image_size_map[img_id] = (Config.IMG_SIZE, Config.IMG_SIZE)

    if debug:
        print(f"Debug mode: Mapped {len(image_size_map)} images.")

    # 2. Data Preparation
    # prepare_data handles caching. We use it to get processed images (512x512)
    print("Preparing test data pipeline...")
    test_images, _, _, test_ids = prepare_data(
        "test", load_cached_data=True, debug=debug
    )

    test_transforms = get_transforms("test")
    test_dataset = SIIMDataset(test_images, ids=test_ids, transforms=test_transforms)

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Loading
    print(f"Loading model from {Config.MODEL_PATH}...")
    model = ResNet18UNetMultiScale(num_classes=Config.NUM_CLASSES, pretrained=False)

    if os.path.exists(Config.MODEL_PATH):
        state_dict = torch.load(Config.MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(
            f"WARNING: Checkpoint {Config.MODEL_PATH} not found. Using random weights."
        )

    model.to(device)
    model.eval()

    # 4. Inference Loop with TTA
    study_probs_map = {}  # study_id -> list of probability vectors
    image_data_map = {}  # image_id -> {mask: np.array, study_id: str}

    print("Running inference...")

    with torch.no_grad():
        for images, indices in test_loader:
            images = images.to(device)

            # --- TTA: Horizontal Flip ---
            # 1. Original Forward
            logit_mask, logit_cls = model(images)
            prob_mask = torch.sigmoid(logit_mask)
            prob_cls = torch.softmax(logit_cls, dim=1)

            # 2. Flipped Forward
            images_flip = torch.flip(images, dims=[3])  # Flip width
            logit_mask_flip, logit_cls_flip = model(images_flip)
            prob_mask_flip = torch.sigmoid(logit_mask_flip)
            prob_cls_flip = torch.softmax(logit_cls_flip, dim=1)

            # 3. Un-flip Mask
            prob_mask_flip = torch.flip(prob_mask_flip, dims=[3])

            # 4. Average
            avg_prob_mask = (prob_mask + prob_mask_flip) / 2.0
            avg_prob_cls = (prob_cls + prob_cls_flip) / 2.0

            # Move to CPU
            avg_prob_mask = avg_prob_mask.cpu().numpy()
            avg_prob_cls = avg_prob_cls.cpu().numpy()

            # Store results
            batch_ids = test_ids[indices]

            for i, img_id in enumerate(batch_ids):
                study_id = image_to_study.get(img_id, f"{img_id}_study")

                # Accumulate study probs
                if study_id not in study_probs_map:
                    study_probs_map[study_id] = []
                study_probs_map[study_id].append(avg_prob_cls[i])

                # Store image mask
                image_data_map[img_id] = {
                    "mask": avg_prob_mask[i, 0],  # (H, W)
                    "study_id": study_id,
                }

    # 5. Submission Generation
    print("Generating submission strings...")
    submission_rows = []

    # A. Determine Study Labels
    final_study_labels = {}  # study_id -> short_class_name

    for study_id, probs_list in study_probs_map.items():
        # Average probs across all images in the study
        mean_probs = np.mean(probs_list, axis=0)
        best_idx = np.argmax(mean_probs)
        confidence = mean_probs[best_idx]

        full_class_name = Config.CLASSES[best_idx]
        short_class_name = class_map[full_class_name]

        final_study_labels[study_id] = short_class_name

        # Study Row: id, PredictionString
        # Format: class conf 0 0 1 1
        submission_rows.append(
            {
                "id": f"{study_id}_study",
                "PredictionString": f"{short_class_name} {confidence:.6f} 0 0 1 1",
            }
        )

    # B. Determine Image Labels (Gated)
    for img_id, data in image_data_map.items():
        mask = data["mask"]
        study_id = data["study_id"]

        # Get the decided label for this study
        study_label = final_study_labels.get(study_id, "negative")

        # Gating Logic: If study is negative, image must be none
        if study_label == "negative":
            pred_string = "none 1 0 0 1 1"
        else:
            # Extract boxes from mask
            boxes = mask2bbox(mask, threshold=0.5)

            if len(boxes) == 0:
                pred_string = "none 1 0 0 1 1"
            else:
                # We have boxes. We need to scale them to original image size.
                orig_w, orig_h = image_size_map.get(
                    img_id, (Config.IMG_SIZE, Config.IMG_SIZE)
                )
                scale_x = orig_w / Config.IMG_SIZE
                scale_y = orig_h / Config.IMG_SIZE

                box_strings = []
                for box in boxes:
                    # box is [xmin, ymin, xmax, ymax] in 512x512 coords
                    x1, y1, x2, y2 = box

                    # Calculate confidence: mean probability inside the box
                    # Clip coords to mask size to avoid index errors
                    mx1 = max(0, min(x1, Config.IMG_SIZE - 1))
                    my1 = max(0, min(y1, Config.IMG_SIZE - 1))
                    mx2 = max(0, min(x2, Config.IMG_SIZE))
                    my2 = max(0, min(y2, Config.IMG_SIZE))

                    if mx2 > mx1 and my2 > my1:
                        box_conf = np.mean(mask[my1:my2, mx1:mx2])
                    else:
                        box_conf = 0.0

                    # Scale to original dimensions
                    sx1 = x1 * scale_x
                    sy1 = y1 * scale_y
                    sx2 = x2 * scale_x
                    sy2 = y2 * scale_y

                    box_strings.append(
                        f"opacity {box_conf:.6f} {sx1:.1f} {sy1:.1f} {sx2:.1f} {sy2:.1f}"
                    )

                if box_strings:
                    pred_string = " ".join(box_strings)
                else:
                    pred_string = "none 1 0 0 1 1"

        submission_rows.append(
            {"id": f"{img_id}_image", "PredictionString": pred_string}
        )

    # 6. Save Submission
    df_sub = pd.DataFrame(submission_rows)
    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH} with {len(df_sub)} rows.")

    return df_sub
